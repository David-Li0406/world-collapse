"""Phase B: closed-loop online training.

This file implements the three experimental conditions from the plan:

  * ``collapse_prone``  : WM updated on a recency-biased buffer (FIFO window).
  * ``balanced_replay`` : WM updated on uniform mix of D_pre + D_online.
  * ``frozen_wm``       : WM not updated at all; actor/critic still train on
                          imagined rollouts from the frozen WM.

Each iteration:
  1. collect ``collect_episodes`` real episodes with the current actor,
  2. append to the active buffer,
  3. run ``wm_steps`` world-model updates (skipped under frozen_wm),
  4. run ``head_steps`` reward+semantic head updates,
  5. run ``actor_steps`` imagination updates,
  6. every ``eval_every`` iterations: full evaluation against the probe bank
     + goal-shift task. Snapshot checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wcollapse.data.buffer import ReplayBuffer
from wcollapse.data.probe_bank import ProbeBank
from wcollapse.envs.metaworld_env import MetaworldVisualEnv
from wcollapse.eval.behavior import goal_shift_eval
from wcollapse.eval.coverage import coverage_metrics
from wcollapse.eval.wm_eval import probe_eval
from wcollapse.models.actor import Actor
from wcollapse.models.critic import Critic
from wcollapse.models.ivideogpt_wrapper import MiniWorldModel
from wcollapse.models.reward_head import RewardHead
from wcollapse.models.semantic_head import SemanticHead
from wcollapse.training.collection import rollout_actor, collect_seed_dataset
from wcollapse.training.sac import build_sac_optimizers, sac_step
from wcollapse.utils.checkpoint import save_checkpoint
from wcollapse.utils.logging import MetricsLogger


def make_active_buffer(
    pretrain_buffer: ReplayBuffer,
    condition: str,
    cfg: DictConfig,
) -> ReplayBuffer:
    """Return the buffer that WM updates draw from this run.

    For ``balanced_replay`` we reuse ``pretrain_buffer`` (which already holds
    D_pre) and append D_online to the same store. For ``collapse_prone`` we
    spin up a fresh recency-windowed buffer that *does not* include D_pre at all
    — that's what the proposal calls "small replay buffer or no replay".
    """
    if condition == "balanced_replay":
        pretrain_buffer.mode = "uniform"
        return pretrain_buffer
    elif condition in {"collapse_prone", "frozen_wm"}:
        return ReplayBuffer(
            capacity=int(cfg.recent_window),
            window_size=int(cfg.recent_window),
            seq_len=pretrain_buffer.seq_len,
            image_size=pretrain_buffer.image_size,
            mode="recent",
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def online_loop(
    *,
    condition: str,
    env: MetaworldVisualEnv,
    world_model: MiniWorldModel,
    reward_head: RewardHead,
    semantic_head: SemanticHead,
    actor: Actor,
    critic: Critic,
    pretrain_buffer: ReplayBuffer,
    pretrain_wm_state: dict,
    probe_bank: ProbeBank,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
) -> None:
    metrics_logger = MetricsLogger(output_dir / "metrics.jsonl")

    wm_opt = torch.optim.AdamW(world_model.parameters(), lr=float(cfg.lr_wm))
    rew_opt = torch.optim.AdamW(reward_head.parameters(), lr=float(cfg.lr_head))
    sem_opt = torch.optim.AdamW(semantic_head.parameters(), lr=float(cfg.lr_head))
    sac_opts = build_sac_optimizers(actor, critic, cfg, device)

    active_buffer = make_active_buffer(pretrain_buffer, condition, cfg)

    # Goal-region restriction during data collection. The trained sub-region
    # occupies the first `goal_bias_fraction` of the goal-x range; the
    # complement is reserved for goal-shift evaluation. Smaller fractions
    # give a sharper visitation gradient (variant `aggressive_bias`).
    goal_low_full = env.goal_low
    goal_high_full = env.goal_high
    bias_fraction = float(cfg.get("goal_bias_fraction", 0.5))
    split_x = float(goal_low_full[0] + (goal_high_full[0] - goal_low_full[0]) * bias_fraction)
    trained_lo = goal_low_full.copy()
    trained_hi = goal_high_full.copy()
    trained_hi[0] = split_x
    holdout_lo = goal_low_full.copy()
    holdout_hi = goal_high_full.copy()
    holdout_lo[0] = split_x
    trained_subregion = (trained_lo, trained_hi)
    holdout_subregion = (holdout_lo, holdout_hi)
    print(
        f"[online] goal bias: trained x ∈ [{trained_lo[0]:.3f}, {split_x:.3f}], "
        f"holdout x ∈ [{split_x:.3f}, {holdout_hi[0]:.3f}] (fraction={bias_fraction:.2f})",
        flush=True,
    )
    # Align the coverage module's static partition split with the bias split,
    # so static_visited == "probes whose goal_x is in the trained subregion".
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(cfg, False)
    cfg.static_goal_split = split_x

    # Pre-pretrain WM snapshot is M_0 for the forgetting score.
    save_checkpoint(
        output_dir / "ckpts" / "wm_M0.pt",
        world_model=world_model,
    )

    # Optional seed phase: random + scripted rollouts so SAC has off-policy
    # diversity before the first agent update. iVideoGPT mbrl uses
    # num_seed_frames=4000; we scale this to a small number of episodes.
    seed_episodes = int(cfg.get("seed_episodes", 0))
    if seed_episodes > 0:
        print(f"[online] seed phase: collecting {seed_episodes} scripted+random episodes", flush=True)
        seed_trajs = collect_seed_dataset(
            env=env,
            n_episodes=seed_episodes,
            scripted_fraction=0.5,
            scripted_noise=float(cfg.get("seed_noise", 0.5)),
            goal_subregion=trained_subregion if bool(cfg.bias_goal) else None,
            seed=int(cfg.get("seed", 0)) + 7777,
        )
        for tr in seed_trajs:
            active_buffer.add_trajectory(tr)
            pretrain_buffer.add_trajectory(tr)

    # Optional imitation warmup (variant B): behaviour-clone the SAC actor on
    # noiseless scripted demos before the first SAC update so the actor has a
    # non-trivial starting policy instead of pure Gaussian noise. This gives
    # the WM-policy feedback something to feed back on.
    imitation_episodes = int(cfg.get("imitation_warmup_episodes", 0))
    if imitation_episodes > 0:
        print(
            f"[online] imitation warmup: {imitation_episodes} scripted demos + "
            f"{int(cfg.get('imitation_warmup_steps', 1000))} BC steps",
            flush=True,
        )
        demo_trajs = collect_seed_dataset(
            env=env,
            n_episodes=imitation_episodes,
            scripted_fraction=1.0,
            scripted_noise=0.0,
            goal_subregion=trained_subregion if bool(cfg.bias_goal) else None,
            seed=int(cfg.get("seed", 0)) + 31337,
        )
        for tr in demo_trajs:
            active_buffer.add_trajectory(tr)
            pretrain_buffer.add_trajectory(tr)

        # Flatten (rgb, action) pairs across demos.
        import numpy as _np
        demo_rgbs = _np.concatenate([tr.rgb[:-1] for tr in demo_trajs], axis=0)
        demo_actions = _np.concatenate([tr.actions for tr in demo_trajs], axis=0)
        bc_opt = torch.optim.AdamW(actor.parameters(), lr=float(cfg.get("lr_actor_bc", cfg.lr_actor)))
        bc_steps = int(cfg.get("imitation_warmup_steps", 1000))
        bc_batch = int(cfg.get("imitation_batch_size", 128))
        rng = np.random.default_rng(int(cfg.get("seed", 0)) + 4242)
        N = demo_actions.shape[0]
        for s in range(bc_steps):
            idx = rng.integers(0, N, size=bc_batch)
            rgb_batch = demo_rgbs[idx]
            act_batch = demo_actions[idx]
            with torch.no_grad():
                x = torch.from_numpy(_np.ascontiguousarray(rgb_batch)).to(device).float() / 127.5 - 1.0
                x = x.permute(0, 3, 1, 2)
                z = world_model.encoder(x)
            target = torch.from_numpy(act_batch).to(device).clamp(-0.999, 0.999)
            sample = actor.sample(z)
            # Maximize log p(target | z) under the actor's tanh-squashed Gaussian.
            # Decompose: u = atanh(target), then -0.5*((u - mu)/std)^2 - log_std - log(1 - target^2)
            u_target = torch.atanh(target)
            mu, log_std = actor._dist_params(z)
            log_p = (
                -0.5 * (((u_target - mu) / log_std.exp()) ** 2)
                - log_std
                - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=device))
            )
            log_p = log_p.sum(-1) - torch.log(1 - target.pow(2) + 1e-6).sum(-1)
            bc_loss = -log_p.mean()
            bc_opt.zero_grad(set_to_none=True)
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            bc_opt.step()
            if s == 0 or (s + 1) % max(1, bc_steps // 5) == 0:
                print(f"[online] BC step {s+1}/{bc_steps} loss={bc_loss.item():.3f}", flush=True)

    for it in range(int(cfg.iterations)):
        # 1) collect real trajectories with the current actor on the trained subregion
        new_trajs = rollout_actor(
            env=env,
            world_model=world_model,
            actor=actor,
            n_episodes=int(cfg.collect_episodes),
            goal_subregion=trained_subregion if bool(cfg.bias_goal) else None,
            exploration_noise=float(cfg.exploration_noise),
            deterministic=False,
            device=device,
            seed=int(cfg.get("seed", 0)) + it,
        )
        for tr in new_trajs:
            active_buffer.add_trajectory(tr)
            if condition == "balanced_replay":
                # Already added to pretrain_buffer (same object); no double-write.
                pass
            else:
                # Also tee into pretrain_buffer so coverage analysis can still
                # see the full visitation history, but DO NOT use this for WM.
                pretrain_buffer.add_trajectory(tr)

        # 2) WM updates
        wm_metrics = {}
        if condition != "frozen_wm":
            world_model.train()
            for _ in range(int(cfg.wm_steps)):
                batch = active_buffer.sample_sequences(batch_size=int(cfg.batch_size))
                wm_opt.zero_grad(set_to_none=True)
                loss = world_model.compute_loss(batch)
                loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
                wm_opt.step()
                wm_metrics = {
                    "wm_total": float(loss["total"].item()),
                    "wm_recon": float(loss["recon"].item()),
                    "wm_dyn": float(loss["dynamics"].item()),
                }

        # 3) reward head update (always) + semantic head update (unless frozen).
        # Variant A (proposal §plan-1+2) freezes the semantic head during Phase B
        # so the probe-error metric isolates WM drift from head drift. The
        # reward head must keep training because SAC's MBPO rollouts score
        # imagined transitions via reward_head — freezing it would stop
        # the actor from being able to track the changing reward distribution.
        freeze_sem = bool(cfg.get("freeze_semantic_head", False))
        head_metrics: dict[str, float] = {}
        for _ in range(int(cfg.head_steps)):
            batch = active_buffer.sample_sequences(batch_size=int(cfg.batch_size))
            with torch.no_grad():
                rgb = torch.from_numpy(batch["rgb"]).to(device)
                x = (rgb.float() / 127.5 - 1.0).permute(0, 1, 4, 2, 3)
                B, Lp1 = x.shape[:2]
                z = world_model.encoder(x.reshape(B * Lp1, *x.shape[2:])).view(B, Lp1, -1)
            target_rewards = torch.from_numpy(batch["rewards"]).to(device)
            target_semantic = torch.from_numpy(batch["semantic"]).to(device)[:, 1:]
            z_post = z[:, 1:]
            rew_opt.zero_grad(set_to_none=True)
            rew_loss = F.mse_loss(
                reward_head(z_post.reshape(-1, z_post.shape[-1])),
                target_rewards.reshape(-1),
            )
            rew_loss.backward()
            rew_opt.step()
            if not freeze_sem:
                sem_opt.zero_grad(set_to_none=True)
                sem_loss = F.mse_loss(
                    semantic_head(z_post.reshape(-1, z_post.shape[-1])),
                    target_semantic.reshape(-1, target_semantic.shape[-1]),
                )
                sem_loss.backward()
                sem_opt.step()
                sem_value = float(sem_loss.item())
            else:
                # Still report the (no-grad) loss so the metric stays comparable across variants.
                with torch.no_grad():
                    sem_value = float(
                        F.mse_loss(
                            semantic_head(z_post.reshape(-1, z_post.shape[-1])),
                            target_semantic.reshape(-1, target_semantic.shape[-1]),
                        ).item()
                    )
            head_metrics = {
                "reward_mse": float(rew_loss.item()),
                "semantic_mse": sem_value,
                "freeze_semantic_head": float(freeze_sem),
            }

        # 4) actor + twin-Q critic via SAC. Mixes real transitions from the
        #    active buffer with short-horizon imagined transitions from the
        #    WM (MBPO recipe — see iVideoGPT/mbrl/train_metaworld_mbpo.py).
        sac_metrics: dict[str, float] = {}
        use_imag = bool(cfg.get("use_imagination", True))
        for _ in range(int(cfg.actor_steps)):
            sac_metrics = sac_step(
                world_model=world_model,
                reward_head=reward_head,
                actor=actor,
                critic=critic,
                opts=sac_opts,
                buffer=active_buffer,
                cfg=cfg,
                device=device,
                use_imagination=use_imag,
            )

        # 5) periodic evaluation
        if it % int(cfg.eval_every) == 0 or it == int(cfg.iterations) - 1:
            coverage = coverage_metrics(
                pretrain_buffer=pretrain_buffer,
                active_buffer=active_buffer,
                probe_bank=probe_bank,
                cfg=cfg,
            )
            wm_eval = probe_eval(
                world_model=world_model,
                semantic_head=semantic_head,
                probe_bank=probe_bank,
                wm_baseline_path=output_dir / "ckpts" / "wm_M0.pt",
                device=device,
                cfg=cfg,
                visited_mask=coverage["visited_mask"],
                static_visited_mask=coverage.get("static_visited_mask"),
            )
            behavior = goal_shift_eval(
                env=env,
                world_model=world_model,
                actor=actor,
                trained_subregion=trained_subregion,
                holdout_subregion=holdout_subregion,
                n_eval_episodes=int(cfg.n_eval_episodes),
                device=device,
                seed=int(cfg.get("seed", 0)) + 10_000 + it,
            )
            metrics = {
                **wm_metrics,
                **head_metrics,
                **sac_metrics,
                **{f"cov/{k}": v for k, v in coverage["scalar"].items()},
                **{f"wm/{k}": v for k, v in wm_eval.items()},
                **{f"beh/{k}": v for k, v in behavior.items()},
                "condition": condition,
                "iteration": it,
            }
            metrics_logger.log(step=it, metrics=metrics)
            save_checkpoint(
                output_dir / "ckpts" / f"ckpt_{it:06d}.pt",
                world_model=world_model,
                reward_head=reward_head,
                semantic_head=semantic_head,
                actor=actor,
                critic=critic,
                iteration=it,
            )

    metrics_logger.close()
