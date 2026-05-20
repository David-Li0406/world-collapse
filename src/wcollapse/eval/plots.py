"""Render Figures 1, 2, 4, 5 from a run's metrics.jsonl.

Importable from train.py so the end-of-run plotting works without making
scripts/ a python package; the CLI lives in scripts/make_plots.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        if key in r:
            xs.append(r["step"])
            ys.append(r[key])
    return np.asarray(xs), np.asarray(ys, dtype=float)


def make_plots(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return
    rows = _load_jsonl(metrics_path)
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots()
    xs, ys = _extract(rows, "cov/visitation_entropy")
    if ys.size:
        ax1.plot(xs, ys, label="visitation entropy", color="C0")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("entropy", color="C0")
    ax2 = ax1.twinx()
    xs2, ys2 = _extract(rows, "cov/support_gap")
    if ys2.size:
        ax2.plot(xs2, ys2, label="support gap", color="C1")
        ax2.set_ylabel("support gap", color="C1")
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_visitation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for h, c in zip((1, 5, 10, 15), ("C0", "C1", "C2", "C3")):
        xs_v, ys_v = _extract(rows, f"wm/sem_err_h{h}_visited")
        xs_u, ys_u = _extract(rows, f"wm/sem_err_h{h}_underv")
        if ys_v.size:
            ax.plot(xs_v, ys_v, color=c, linestyle="-", label=f"visited h={h}")
        if ys_u.size:
            ax.plot(xs_u, ys_u, color=c, linestyle="--", label=f"under-visited h={h}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("semantic L2 error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_probe_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for k, color, label in (
        ("beh/trained_success_rate", "C0", "trained subregion"),
        ("beh/holdout_success_rate", "C1", "holdout subregion"),
    ):
        xs, ys = _extract(rows, k)
        if ys.size:
            ax.plot(xs, ys, color=color, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_goal_shift.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for h, c in zip((1, 5, 10, 15), ("C0", "C1", "C2", "C3")):
        xs, ys = _extract(rows, f"wm/gap_h{h}")
        if ys.size:
            ax.plot(xs, ys, color=c, label=f"horizon {h}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("E_out[err] - E_in[err]")
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure5_off_support_gap.png", dpi=150)
    plt.close(fig)
