"""Phase B (online) / round-0 entry point: DrQ-v2 policy + VideoPredictor WM.

Single entry. Two modes:

  --mode round0   Train (WM, policy, real buffer) from scratch on the full env.
                  No bias_goal, no probe eval. Output saved as M_0 for Phase B.

  --mode online --condition {collapse_prone,balanced_replay,frozen_wm}
                  Load M_0 + Policy_0 + D_real_R0 from round-0. Run online MBPO
                  with the condition-specific WM data source. Every
                  `eval_every_frames` frames: probe-bank eval, coverage,
                  goal-shift behavior eval.

The DrQ-v2 policy training is unchanged across conditions
(real_ratio=0.5 real env + WM-rolled imagination). What varies is the
distribution the WM is trained on — that's the world-collapse knob.
"""

from __future__ import annotations

# Env vars must be set BEFORE torch/mujoco imports.
import os
os.environ.setdefault("MUJOCO_GL", "egl")
# mujoco 3.x reads MUJOCO_EGL_DEVICE_ID, not EGL_DEVICE_ID. With
# CUDA_VISIBLE_DEVICES restricting visibility, the visible GPU is at index 0.
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# Pre-load NVIDIA EGL ICD so glvnd dispatches to it (in-process registry).
import ctypes as _ctypes
import glob as _glob
for _libpath in sorted(_glob.glob("/opt/nvidia-*/lib64/libEGL_nvidia.so.0")):
    try:
        _ctypes.cdll.LoadLibrary(_libpath)
        print(f"[boot] preloaded NVIDIA EGL ICD: {_libpath}", flush=True)
        break
    except OSError as _exc:
        print(f"[boot] WARN preload {_libpath} failed: {_exc}", flush=True)

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Bootstrap iVideoGPT submodule imports (same code as wmcollapse vendors).
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
_IVG = _REPO / "iVideoGPT"
if str(_IVG / "mbrl") not in sys.path:
    sys.path.insert(0, str(_IVG / "mbrl"))
if str(_IVG) not in sys.path:
    sys.path.insert(0, str(_IVG))
os.environ.setdefault("IVIDEOGPT_ROOT", str(_IVG))

import numpy as np
import torch
from dm_env import specs
from omegaconf import OmegaConf, DictConfig
from tqdm import trange

# wmcollapse imports
import drq_utils
import replay_buffer as _rb
from replay_buffer import (
    ReplayBufferStorage,
    make_replay_loader,
    make_segment_replay_loader,
)
from video_predictor import VideoPredictor
from drqv2 import DrQV2Agent

# Patch upstream worker init: it does `np.random.get_state()[1][0] + worker_id`
# which yields a numpy uint32 that Python 3.12's random.seed() rejects.
import random as _random
def _worker_init_fn(worker_id: int) -> None:
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    _random.seed(seed)
_rb._worker_init_fn = _worker_init_fn

from wcollapse.envs.metaworld_dmenv import make as make_dmenv
from wcollapse.data.probe_bank import load_probe_bank, ProbeBank
from wcollapse.eval.wm_eval_drqv2 import probe_eval
from wcollapse.eval.coverage_drqv2 import coverage_metrics
from wcollapse.eval.behavior_drqv2 import goal_shift_eval

torch.backends.cudnn.benchmark = True


# ---- helpers ----------------------------------------------------------------


def _drill_inner(env):
    """Find the innermost MetaWorldV3 (has set_goal_subregion + goal_low/high)."""
    cur = env
    while not hasattr(cur, "set_goal_subregion"):
        cur = cur._env
    return cur


def _make_agent(obs_spec, action_spec, agent_cfg) -> DrQV2Agent:
    return DrQV2Agent(
        obs_shape=tuple(obs_spec.shape),
        action_shape=tuple(action_spec.shape),
        device=str(agent_cfg.device),
        lr=float(agent_cfg.lr),
        critic_target_tau=float(agent_cfg.critic_target_tau),
        update_every_steps=int(agent_cfg.update_every_steps),
        use_tb=False,
        num_expl_steps=int(agent_cfg.num_expl_steps),
        hidden_dim=int(agent_cfg.hidden_dim),
        feature_dim=int(agent_cfg.feature_dim),
        stddev_schedule=str(agent_cfg.stddev_schedule),
        stddev_clip=float(agent_cfg.stddev_clip),
        beta=float(agent_cfg.get("beta", 0.0)),
        delay_steps=int(agent_cfg.get("delay_steps", 1)),
    )


def _make_data_specs(env) -> tuple:
    """Replay-buffer data_specs incl. raw 39-d state for coverage extraction."""
    return (
        env.observation_spec(),
        env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
        specs.Array((39,), np.float64, "state"),
    )


def _record(storage: ReplayBufferStorage, time_step):
    """Adapter: wmcollapse storage `add` indexes by spec.name on the time_step.

    Our ExtendedTimeStep already exposes .observation/.action/.reward/.discount.
    `state` is set on the inner _StepWithExtras. We monkey-attach so the spec
    name lookup finds it.
    """
    # ExtendedTimeStep is a NamedTuple — to avoid mutating it, build a tiny
    # wrapper dict-style object that `for spec in data_specs: time_step[spec.name]`
    # can index into.
    class _Adapter:
        def __init__(self, ts):
            self._ts = ts
        def __getitem__(self, k):
            if k == "state":
                state = getattr(self._ts, "state", None)
                if state is None:
                    return np.zeros(39, dtype=np.float64)
                return np.asarray(state, dtype=np.float64)
            return getattr(self._ts, k)
        def last(self):
            return self._ts.last()
    return storage.add(_Adapter(time_step))


def _eval_policy(env, agent, global_step: int, n_episodes: int) -> tuple[float, float]:
    total_reward, total_success = 0.0, 0.0
    for _ in range(n_episodes):
        ts = env.reset()
        ep_success = 0.0
        while not ts.last():
            with torch.no_grad():
                action = agent.act(ts.observation, global_step, eval_mode=True)
            ts = env.step(action)
            total_reward += ts.reward
            ep_success = max(ep_success, ts.success)
        total_success += float(ep_success >= 1.0)
    return total_reward / n_episodes, total_success / n_episodes


# ---- main loop --------------------------------------------------------------


def run(
    mode: str,
    cfg: DictConfig,
    output_dir: Path,
    condition: str | None = None,
    round0_dir: Path | None = None,
    probe_bank_path: Path | None = None,
) -> None:
    assert mode in {"round0", "online"}
    if mode == "online":
        assert condition in {"collapse_prone", "balanced_replay", "frozen_wm"}
        assert round0_dir is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    drq_utils.set_seed_everywhere(int(cfg.seed))
    device = torch.device(cfg.device)
    action_repeat = int(cfg.action_repeat)

    train_env = make_dmenv(
        cfg.task_name, int(cfg.frame_stack), action_repeat,
        int(cfg.seed), str(cfg.camera), int(cfg.duration), float(cfg.succ_bonus),
    )
    eval_env = make_dmenv(
        cfg.task_name, int(cfg.frame_stack), action_repeat,
        int(cfg.seed) + 1000, str(cfg.camera), int(cfg.duration), float(cfg.succ_bonus),
    )

    # ---- bias_goal: trained vs holdout sub-region split (Phase B only) ----
    trained_sub = None
    holdout_sub = None
    if mode == "online" and bool(cfg.get("bias_goal", False)):
        inner = _drill_inner(train_env)
        gl, gh = inner.goal_low, inner.goal_high
        frac = float(cfg.get("bias_fraction", 0.5))
        split_x = float(gl[0] + (gh[0] - gl[0]) * frac)
        trained_lo = gl.copy(); trained_hi = gh.copy(); trained_hi[0] = split_x
        holdout_lo = gl.copy(); holdout_hi = gh.copy(); holdout_lo[0] = split_x
        trained_sub = (trained_lo, trained_hi)
        holdout_sub = (holdout_lo, holdout_hi)
        inner.set_goal_subregion(trained_lo, trained_hi)
        # Static partition: probes with goal_x < split_x are "trained subregion".
        OmegaConf.set_struct(cfg, False)
        cfg.static_goal_split = split_x
        print(
            f"[bias_goal] trained x ∈ [{trained_lo[0]:.3f}, {split_x:.3f}], "
            f"holdout x ∈ [{split_x:.3f}, {holdout_hi[0]:.3f}]",
            flush=True,
        )

    # ---- agent + WM ----
    agent_cfg = OmegaConf.create({
        "device": cfg.device,
        "lr": cfg.lr,
        "critic_target_tau": cfg.agent.critic_target_tau,
        "update_every_steps": cfg.agent.update_every_steps,
        "num_expl_steps": cfg.agent.num_expl_steps,
        "hidden_dim": cfg.agent.hidden_dim,
        "feature_dim": cfg.agent.feature_dim,
        "stddev_schedule": cfg.agent.stddev_schedule,
        "stddev_clip": cfg.agent.stddev_clip,
        "beta": cfg.agent.get("beta", 0.0),
        "delay_steps": cfg.agent.get("delay_steps", 1),
    })
    agent = _make_agent(train_env.observation_spec(), train_env.action_spec(), agent_cfg)
    video_predictor = VideoPredictor(device, cfg.world_model)
    # iVideoGPT instantiates torch.cuda.amp.GradScaler() but trains under
    # bfloat16 autocast — newer torch rejects unscale on BF16 grads. Replace
    # both scalers with no-op equivalents (correct behavior for BF16).
    class _NoOpScaler:
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
    video_predictor.tok_scaler = _NoOpScaler()
    video_predictor.model_scaler = _NoOpScaler()

    # Load round-0 checkpoint for Phase B; also load M_0 baseline (separate instance).
    wm_baseline = None
    if mode == "online":
        video_predictor.load_snapshot(str(round0_dir))
        snap = torch.load(str(round0_dir / "snapshot.pt"), map_location=device, weights_only=False)
        agent = snap["agent"]
        print(f"[online] loaded WM + DrQ-v2 from {round0_dir}", flush=True)
        wm_baseline = VideoPredictor(device, cfg.world_model)
        wm_baseline.load_snapshot(str(round0_dir))
        torch.nn.Module.train(wm_baseline, False)

    # ---- replay storages (this run) ----
    data_specs = _make_data_specs(train_env)
    round_buffer_dir = output_dir / "buffer"
    imag_buffer_dir = output_dir / "imag_buffer"
    replay_storage = ReplayBufferStorage(data_specs, round_buffer_dir)
    imag_specs = (
        train_env.observation_spec(),
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    imag_replay_storage = ReplayBufferStorage(imag_specs, imag_buffer_dir)

    real_batch_size = int(cfg.batch_size * cfg.real_ratio)
    imag_batch_size = int(cfg.batch_size - real_batch_size)
    replay_loader = make_replay_loader(
        round_buffer_dir, int(cfg.replay_buffer_size), real_batch_size,
        int(cfg.replay_buffer_num_workers), bool(cfg.save_snapshot),
        int(cfg.nstep), float(cfg.discount),
    )
    imag_replay_loader = make_replay_loader(
        imag_buffer_dir, int(cfg.replay_buffer_size), imag_batch_size,
        int(cfg.replay_buffer_num_workers), False, int(cfg.nstep), float(cfg.discount),
    )

    # ---- WM segment loaders (condition-dependent) ----
    seg_len = int(cfg.world_model.segment_length)
    wm_bs = int(cfg.world_model.batch_size)
    seg_loaders: dict[str, torch.utils.data.DataLoader] = {}

    if mode == "round0":
        # Round-0: WM trains on its own growing buffer (standard MBPO).
        seg_loaders["online"] = make_segment_replay_loader(
            round_buffer_dir, int(cfg.replay_buffer_size), wm_bs,
            int(cfg.replay_buffer_num_workers), bool(cfg.save_snapshot),
            int(cfg.nstep), float(cfg.discount), seg_len,
        )
    elif condition in {"collapse_prone", "balanced_replay"}:
        online_max = int(cfg.recent_window) if condition == "collapse_prone" else int(cfg.replay_buffer_size)
        seg_loaders["online"] = make_segment_replay_loader(
            round_buffer_dir, online_max, wm_bs,
            int(cfg.replay_buffer_num_workers), bool(cfg.save_snapshot),
            int(cfg.nstep), float(cfg.discount), seg_len,
        )
        if condition == "balanced_replay":
            seg_loaders["pretrain"] = make_segment_replay_loader(
                round0_dir / "buffer", int(cfg.replay_buffer_size), wm_bs,
                int(cfg.replay_buffer_num_workers), bool(cfg.save_snapshot),
                int(cfg.nstep), float(cfg.discount), seg_len,
            )
    # frozen_wm: no segment loaders needed.

    iters: dict[str, Any] = {k: None for k in list(seg_loaders.keys()) + ["real_pol", "imag_pol"]}
    loaders = {**seg_loaders, "real_pol": replay_loader, "imag_pol": imag_replay_loader}

    def get_next(key: str):
        if iters[key] is None:
            iters[key] = iter(loaders[key])
        try:
            return next(iters[key])
        except StopIteration:
            iters[key] = iter(loaders[key])
            return next(iters[key])

    def get_wm_batch():
        if condition == "balanced_replay" and np.random.random() < 0.5:
            return get_next("pretrain")
        return get_next("online")

    def get_policy_batch(global_step: int):
        real_b = get_next("real_pol")
        use_imag = global_step * action_repeat >= int(cfg.start_mbpo)
        if use_imag:
            # imag buffer may be empty (e.g., NaN-skipped rollouts in early
            # training); fall back to real samples to keep policy training alive.
            try:
                fake_b = get_next("imag_pol")
            except (IndexError, RuntimeError):
                fake_b = get_next("real_pol")
        else:
            fake_b = get_next("real_pol")
        return [torch.cat([r, f], 0) for r, f in zip(real_b, fake_b)]

    def generate_inloop(global_step: int):
        batch = get_next("real_pol")
        policy_fn = lambda obs, _t: agent.act2(obs, max(global_step - 1, 0), eval_mode=False)
        # bf16 autocast generation can produce NaN logits → torch.multinomial
        # CUDA assert. Skip this rollout on failure rather than killing the run;
        # the next call after more WM training usually succeeds.
        try:
            with torch.no_grad():
                obss, actions, rewards = video_predictor.rollout(
                    batch[0][: int(cfg.gen_batch)], policy_fn, int(cfg.gen_horizon),
                )
        except RuntimeError as e:
            if "CUDA error" in str(e) or "multinomial" in str(e):
                print(f"[generate_inloop] skipped at step {global_step}: {e}", flush=True)
                return
            raise
        for i in range(len(obss)):
            imag_replay_storage._store_episode({
                "action": actions[i].detach().cpu().numpy(),
                "observation": (obss[i] * 255).detach().cpu().numpy().astype(np.uint8),
                "reward": rewards[i].detach().cpu().numpy(),
                "discount": np.ones_like(rewards[i].detach().cpu().numpy()),
            })

    # ---- probe bank (online mode only) ----
    probe_bank: ProbeBank | None = None
    if mode == "online" and probe_bank_path is not None and Path(probe_bank_path).exists():
        probe_bank = load_probe_bank(probe_bank_path)
        print(f"[online] probe bank: {len(probe_bank)} probes", flush=True)

    # ---- eval driver ----
    def do_eval(global_step: int) -> dict:
        metrics: dict = {}
        # Policy SR on the trained sub-region (set during bias_goal handshake).
        avg_r, sr = _eval_policy(eval_env, agent, global_step, int(cfg.num_eval_episodes))
        metrics["avg_reward"] = float(avg_r)
        metrics["success_rate"] = float(sr)

        if mode == "online" and probe_bank is not None:
            cov = coverage_metrics(
                pretrain_buffer_dir=round0_dir / "buffer",
                active_buffer_dir=round_buffer_dir,
                recent_n_episodes=max(1, int(cfg.recent_window) // int(cfg.duration)),
                probe_bank=probe_bank,
                cfg=cfg,
            )
            wm = probe_eval(
                video_predictor=video_predictor,
                probe_bank=probe_bank,
                wm_baseline=wm_baseline,
                device=device,
                cfg=cfg,
                visited_mask=cov["visited_mask"],
                static_visited_mask=cov["static_visited_mask"],
                frame_stack=int(cfg.frame_stack),
            )
            for k, v in cov["scalar"].items():
                metrics[f"cov/{k}"] = v
            for k, v in wm.items():
                metrics[f"wm/{k}"] = v

            if trained_sub is not None and holdout_sub is not None:
                # Use the eval env with sub-region swapping.
                beh = goal_shift_eval(
                    eval_env=eval_env, agent=agent,
                    trained_subregion=trained_sub, holdout_subregion=holdout_sub,
                    n_eval_episodes=int(cfg.num_eval_episodes), global_step=global_step,
                )
                for k, v in beh.items():
                    metrics[f"beh/{k}"] = v
                # Restore trained subregion on train_env (eval_env was reset internally).
                _drill_inner(train_env).set_goal_subregion(*trained_sub)
        return metrics

    # ---- main loop ----
    num_train_frames = int(cfg.num_train_frames)
    num_seed_frames = int(cfg.num_seed_frames)
    seed_until = drq_utils.Until(num_seed_frames, action_repeat)
    train_until = drq_utils.Until(num_train_frames, action_repeat, bar_name=f"{mode}_{condition or ''}")
    eval_every = drq_utils.Every(int(cfg.eval_every_frames), action_repeat)
    gen_every = drq_utils.Every(int(cfg.gen_every_steps), action_repeat)
    update_gen_every = drq_utils.Every(int(cfg.update_gen_every_step), action_repeat)

    global_step = 0
    global_episode = 0
    t0 = time.time()
    init_model = False
    init_gen = False

    time_step = train_env.reset()
    _record(replay_storage, time_step)

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")  # truncate

    while train_until(global_step):
        if time_step.last():
            global_episode += 1
            time_step = train_env.reset()
            _record(replay_storage, time_step)

        if eval_every(global_step) and (mode != "round0" or global_step > 0):
            m = do_eval(global_step)
            m["frame"] = global_step * action_repeat
            m["step"] = global_step
            m["mode"] = mode
            m["condition"] = condition or "_"
            with metrics_path.open("a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"[{mode}/{condition or '_'}] frame={m['frame']} sr={m.get('success_rate',-1):.2f}", flush=True)

        with torch.no_grad():
            action = agent.act(time_step.observation, global_step, eval_mode=False)

        if not seed_until(global_step):
            wm_active = (mode == "round0") or condition in {"collapse_prone", "balanced_replay"}
            if wm_active:
                if not init_model:
                    print(f"[{mode}] WM init: {cfg.init_update_gen_steps} steps", flush=True)
                    for _ in trange(int(cfg.init_update_gen_steps), desc="WM init"):
                        video_predictor.train(get_wm_batch())
                    init_model = True
                elif update_gen_every(global_step):
                    update_tok = (global_step % (int(cfg.update_tokenizer_every_step) // action_repeat) == 0)
                    for _ in range(int(cfg.update_gen_times)):
                        video_predictor.train(get_wm_batch(), update_tokenizer=update_tok)
            else:
                init_model = True

            if global_step * action_repeat >= int(cfg.start_mbpo) and not init_gen:
                print(f"[{mode}] init gen: {cfg.init_gen_times} batches", flush=True)
                for _ in trange(int(cfg.init_gen_times), desc="Init gen"):
                    generate_inloop(global_step)
                init_gen = True

            for _ in range(int(cfg.agent_update_times)):
                batch = get_policy_batch(global_step)
                agent.update(iter([batch]), global_step)

            if global_step * action_repeat >= int(cfg.start_mbpo) and gen_every(global_step):
                generate_inloop(global_step)

        time_step = train_env.step(action)
        _record(replay_storage, time_step)
        global_step += 1

    # ---- save final ----
    video_predictor.save_snapshot(str(output_dir))
    torch.save(
        {"agent": agent, "_global_step": global_step, "_global_episode": global_episode},
        str(output_dir / "snapshot.pt"),
    )

    # Final eval
    final = do_eval(global_step)
    final["frame"] = global_step * action_repeat
    final["final"] = True
    final["wall_time_sec"] = time.time() - t0
    with metrics_path.open("a") as f:
        f.write(json.dumps(final) + "\n")
    print(f"[{mode}/{condition or '_'}] DONE in {final['wall_time_sec']:.1f}s; sr={final.get('success_rate',-1):.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["round0", "online"], required=True)
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/drqv2_base.yaml).")
    p.add_argument("--overrides", nargs="*", default=[], help="dotlist overrides, e.g. seed=1 condition=collapse_prone")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--condition", default=None)
    p.add_argument("--round0_dir", default=None)
    p.add_argument("--probe_bank_path", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    run(
        mode=args.mode,
        cfg=cfg,
        output_dir=Path(args.output_dir),
        condition=args.condition,
        round0_dir=Path(args.round0_dir) if args.round0_dir else None,
        probe_bank_path=Path(args.probe_bank_path) if args.probe_bank_path else None,
    )


if __name__ == "__main__":
    main()
