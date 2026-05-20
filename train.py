"""Entry point for the world-collapse experiment.

Modes are dispatched by ``config.mode``:

  collect_pretrain  -> collect D_pre + write trajectories + build probe bank
  pretrain          -> train WM + heads on D_pre (uses cached D_pre + probe bank)
  online            -> Phase B closed loop (loads pretrain checkpoint)
  full              -> collect_pretrain + pretrain + online, end-to-end
  debug             -> a smoke variant of ``full`` with tiny sizes

The same script is invoked by the GitHub Actions workflow on Machine B:
    uv run python train.py --config configs/debug.yaml --output_dir runs/debug-001
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from wcollapse.data.probe_bank import (
    build_probe_bank,
    load_probe_bank,
    save_probe_bank,
)
from wcollapse.data.trajectory import (
    load_trajectories,
    save_trajectories,
)
from wcollapse.envs.metaworld_env import make_env
from wcollapse.models.actor import Actor
from wcollapse.models.critic import Critic
from wcollapse.models.ivideogpt_wrapper import WMConfig, build_world_model
from wcollapse.models.reward_head import RewardHead
from wcollapse.models.semantic_head import SemanticHead
from wcollapse.training.collection import collect_seed_dataset
from wcollapse.training.online import online_loop
from wcollapse.training.pretrain import build_pretrain_buffer, pretrain
from wcollapse.utils.checkpoint import load_checkpoint, save_checkpoint
from wcollapse.utils.logging import MetricsLogger
from wcollapse.utils.seeding import seed_everything


def _load_config(path: Path) -> DictConfig:
    base = OmegaConf.create({})
    if path.parent.joinpath("_base.yaml").exists():
        base = OmegaConf.load(path.parent / "_base.yaml")
    overlay = OmegaConf.load(path)
    return OmegaConf.merge(base, overlay)


def _snapshot_config(cfg: DictConfig, output_dir: Path, config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "config.yaml")
    OmegaConf.save(cfg, output_dir / "config_resolved.yaml")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "mode": str(cfg.mode),
                "condition": str(cfg.get("condition", "")),
                "task": str(cfg.env.task),
            },
            indent=2,
        )
    )


def _build_models(cfg: DictConfig, device: torch.device):
    wm_cfg = WMConfig(
        image_size=int(cfg.env.image_size),
        latent_dim=int(cfg.wm.latent_dim),
        action_dim=4,
        encoder_channels=tuple(cfg.wm.encoder_channels),
        decoder_channels=tuple(cfg.wm.decoder_channels),
        dynamics_hidden=int(cfg.wm.dynamics_hidden),
        dynamics_layers=int(cfg.wm.dynamics_layers),
        backbone=str(cfg.wm.backbone),
    )
    wm = build_world_model(wm_cfg).to(device)
    reward_head = RewardHead(latent_dim=wm.latent_dim).to(device)
    semantic_head = SemanticHead(latent_dim=wm.latent_dim).to(device)
    actor = Actor(latent_dim=wm.latent_dim, action_dim=4).to(device)
    critic = Critic(latent_dim=wm.latent_dim).to(device)
    return wm, reward_head, semantic_head, actor, critic


def _collect_phase(cfg: DictConfig, output_dir: Path) -> Path:
    seed_everything(int(cfg.seed))
    env = make_env(
        task_name=str(cfg.env.task),
        seed=int(cfg.seed),
        image_size=int(cfg.env.image_size),
        camera_name=str(cfg.env.camera_name),
        max_episode_steps=int(cfg.env.max_episode_steps),
    )
    trajs = collect_seed_dataset(
        env=env,
        n_episodes=int(cfg.pretrain_data.n_episodes),
        scripted_fraction=float(cfg.pretrain_data.scripted_fraction),
        scripted_noise=float(cfg.pretrain_data.scripted_noise),
        goal_subregion=None,
        seed=int(cfg.seed),
    )
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "d_pre.hdf5"
    n = save_trajectories(out_path, trajs)
    (output_dir / "metadata_pretrain_data.json").write_text(
        json.dumps({"n_episodes": n, "path": str(out_path)}, indent=2)
    )

    # Probe bank.
    probe_bank = build_probe_bank(
        env=env,
        seed_trajectories=trajs,
        n_probes=int(cfg.probe.n_probes),
        horizon=int(cfg.probe.horizon),
        action_source=str(cfg.probe.action_source),
        seed=int(cfg.seed) + 1,
    )
    save_probe_bank(data_dir / "probes.hdf5", probe_bank)
    return out_path


def _pretrain_phase(cfg: DictConfig, output_dir: Path, device: torch.device) -> None:
    seed_everything(int(cfg.seed))
    trajs = load_trajectories(output_dir / "data" / "d_pre.hdf5")
    buffer = build_pretrain_buffer(
        trajs,
        image_size=int(cfg.env.image_size),
        seq_len=int(cfg.wm.seq_len),
    )
    wm, reward_head, semantic_head, actor, critic = _build_models(cfg, device)
    logger = MetricsLogger(output_dir / "metrics_pretrain.jsonl")
    pretrain(
        wm,
        reward_head,
        semantic_head,
        buffer,
        cfg.pretrain,
        device,
        log_fn=lambda step, m: logger.log(step, m),
    )
    save_checkpoint(
        output_dir / "ckpts" / "pretrained.pt",
        world_model=wm,
        reward_head=reward_head,
        semantic_head=semantic_head,
        actor=actor,
        critic=critic,
    )


def _online_phase(cfg: DictConfig, output_dir: Path, device: torch.device) -> None:
    seed_everything(int(cfg.seed) + 1000)
    env = make_env(
        task_name=str(cfg.env.task),
        seed=int(cfg.seed) + 1000,
        image_size=int(cfg.env.image_size),
        camera_name=str(cfg.env.camera_name),
        max_episode_steps=int(cfg.env.max_episode_steps),
    )
    trajs = load_trajectories(output_dir / "data" / "d_pre.hdf5")
    pretrain_buffer = build_pretrain_buffer(
        trajs,
        image_size=int(cfg.env.image_size),
        seq_len=int(cfg.wm.seq_len),
    )
    probe_bank = load_probe_bank(output_dir / "data" / "probes.hdf5")
    wm, reward_head, semantic_head, actor, critic = _build_models(cfg, device)
    ckpt = load_checkpoint(output_dir / "ckpts" / "pretrained.pt", map_location=str(device))
    wm.load_state_dict(ckpt["world_model"])
    reward_head.load_state_dict(ckpt["reward_head"])
    semantic_head.load_state_dict(ckpt["semantic_head"])
    # Actor and critic start fresh — Dreamer recipe.
    online_loop(
        condition=str(cfg.condition),
        env=env,
        world_model=wm,
        reward_head=reward_head,
        semantic_head=semantic_head,
        actor=actor,
        critic=critic,
        pretrain_buffer=pretrain_buffer,
        pretrain_wm_state=ckpt,
        probe_bank=probe_bank,
        cfg=cfg.online,
        device=device,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help=(
            "OmegaConf dotlist override, repeatable. Example: "
            "--override seed=1 --override online.iterations=50"
        ),
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_config(cfg, output_dir, args.config)

    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    mode = str(cfg.mode)

    print(f"[train] device={device} mode={mode} condition={cfg.get('condition', '')} task={cfg.env.task}", flush=True)

    if mode == "collect_pretrain":
        print("[train] starting collect_pretrain", flush=True)
        _collect_phase(cfg, output_dir)
    elif mode == "pretrain":
        if not (output_dir / "data" / "d_pre.hdf5").exists():
            print("[train] starting collect_pretrain", flush=True)
            _collect_phase(cfg, output_dir)
        print("[train] starting pretrain", flush=True)
        _pretrain_phase(cfg, output_dir, device)
    elif mode == "online":
        if not (output_dir / "data" / "d_pre.hdf5").exists():
            print("[train] starting collect_pretrain", flush=True)
            _collect_phase(cfg, output_dir)
        if not (output_dir / "ckpts" / "pretrained.pt").exists():
            print("[train] starting pretrain", flush=True)
            _pretrain_phase(cfg, output_dir, device)
        print("[train] starting online", flush=True)
        _online_phase(cfg, output_dir, device)
    elif mode in {"full", "debug"}:
        print("[train] starting collect_pretrain", flush=True)
        _collect_phase(cfg, output_dir)
        print("[train] starting pretrain", flush=True)
        _pretrain_phase(cfg, output_dir, device)
        print("[train] starting online", flush=True)
        _online_phase(cfg, output_dir, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print("[train] all phases complete", flush=True)

    # Always try to render plots at the end. Don't fail the whole run if it errors.
    try:
        from wcollapse.eval.plots import make_plots

        make_plots(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plotting failed: {e}")

    # Minimal metrics.json so the workflow's summary step can read something
    # even on truncated runs.
    summary = {
        "mode": mode,
        "condition": str(cfg.get("condition", "")),
        "task": str(cfg.env.task),
        "output_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
