"""Cross-condition × cross-seed analysis for the world-collapse experiment.

Given a directory of per-run outputs (one subdir per ``<condition>-seed<N>``),
this script:

  1. Loads every ``metrics.jsonl`` and parses condition+seed from the path.
  2. Groups by condition, computes mean ± SEM across seeds at each step.
  3. Plots the four required §4.5 figures with one line per condition and a
     shaded ±SEM band:
       Figure 1  visitation entropy + support gap
       Figure 2  visited vs under-visited probe semantic error (horizons 1,5,10,15)
       Figure 4  trained vs holdout subregion success rate
       Figure 5  off-support generalization gap E_out − E_in
  4. Writes a Markdown summary (collapse_signature.md) that scores each
     condition against the §4.7 three-way world-collapse signature:
       (i) visitation entropy decreasing,
       (ii) under-visited probe error growing faster than visited,
       (iii) holdout success dropping while trained stays flat.

Usage:
    python scripts/aggregate_analysis.py --runs_dir downloaded_runs/conditions-pretrain-v1 \
                                          --out_dir analysis/<batch_name>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# Color/style per condition. Stable so multi-figure comparisons stay readable.
CONDITION_STYLE = {
    "collapse_prone":  {"color": "#d62728", "label": "collapse-prone (recent-only)"},
    "frozen_wm":       {"color": "#1f77b4", "label": "frozen pretrained WM"},
    "balanced_replay": {"color": "#2ca02c", "label": "balanced replay"},
}

NAME_RE = re.compile(r"^(?P<cond>[a-z_]+)-seed(?P<seed>\d+)$")


def _find_runs(runs_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Return {condition_name: [(seed, run_dir), ...]} sorted by seed.

    Handles both layouts that `gh run download` produces:
      runs_dir/<run-name>/metrics.jsonl
      runs_dir/<artifact-name>/<run-name>/metrics.jsonl
    """
    out: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for jsonl_path in runs_dir.rglob("metrics.jsonl"):
        run_dir = jsonl_path.parent
        m = NAME_RE.match(run_dir.name)
        if m is None:
            continue
        cond = m.group("cond")
        seed = int(m.group("seed"))
        out[cond].append((seed, run_dir))
    for cond in out:
        out[cond].sort(key=lambda x: x[0])
    return out


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["step"])
            ys.append(float(r[key]))
    return np.asarray(xs, dtype=int), np.asarray(ys, dtype=float)


def _align_on_steps(per_seed: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take a list of (xs, ys) per seed; return (xs, mean_ys, sem_ys) on the
    intersection of step indices. SEM = std/sqrt(n)."""
    if not per_seed:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    common = set(per_seed[0][0].tolist())
    for xs, _ in per_seed[1:]:
        common &= set(xs.tolist())
    if not common:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    common_xs = np.array(sorted(common))
    stacked = []
    for xs, ys in per_seed:
        idx = np.array([np.where(xs == x)[0][0] for x in common_xs])
        stacked.append(ys[idx])
    arr = np.stack(stacked, axis=0)  # (n_seeds, n_steps)
    n = arr.shape[0]
    mean = arr.mean(0)
    sem = arr.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return common_xs, mean, sem


def _plot_curve(ax, runs_by_cond: dict[str, list[Path]], key: str, **plot_kwargs) -> None:
    """For each condition: load metric `key` from every seed, plot mean±SEM."""
    for cond, paths in runs_by_cond.items():
        style = CONDITION_STYLE.get(cond, {"color": None, "label": cond})
        per_seed = []
        for p in paths:
            rows = _load_rows(p / "metrics.jsonl")
            xs, ys = _series(rows, key)
            if xs.size > 0:
                per_seed.append((xs, ys))
        if not per_seed:
            continue
        common_xs, mean, sem = _align_on_steps(per_seed)
        if common_xs.size == 0:
            continue
        ax.plot(common_xs, mean, color=style["color"], label=style["label"], **plot_kwargs)
        if sem.size and sem.sum() > 0:
            ax.fill_between(common_xs, mean - sem, mean + sem,
                            color=style["color"], alpha=0.2, linewidth=0)


def figure1_visitation(runs_by_cond: dict[str, list[Path]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    _plot_curve(axes[0], runs_by_cond, "cov/visitation_entropy")
    axes[0].set_title("Visitation entropy (recent window)")
    axes[0].set_xlabel("online iteration")
    axes[0].set_ylabel("entropy (nats)")
    axes[0].legend(fontsize=8)
    _plot_curve(axes[1], runs_by_cond, "cov/support_gap")
    axes[1].set_title("Support gap |S_ref \\ S_t| / |S_ref|")
    axes[1].set_xlabel("online iteration")
    axes[1].set_ylabel("fraction of pretrain support no longer visited")
    axes[1].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure2_probe_error(runs_by_cond: dict[str, list[Path]], out: Path) -> None:
    horizons = (1, 5, 10, 15)
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4), sharey=True)
    for ax, h in zip(axes, horizons):
        _plot_curve(ax, runs_by_cond, f"wm/sem_err_h{h}_visited", linestyle="-")
        _plot_curve(ax, runs_by_cond, f"wm/sem_err_h{h}_underv", linestyle="--")
        ax.set_title(f"horizon H={h}")
        ax.set_xlabel("online iteration")
    axes[0].set_ylabel("semantic L2 error\n(solid = visited probes, dashed = under-visited)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Figure 2 — fixed-probe semantic rollout error", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure4_goal_shift(runs_by_cond: dict[str, list[Path]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    _plot_curve(axes[0], runs_by_cond, "beh/trained_success_rate")
    axes[0].set_title("Trained subregion success")
    _plot_curve(axes[1], runs_by_cond, "beh/holdout_success_rate")
    axes[1].set_title("Holdout subregion success (goal shift)")
    for ax in axes:
        ax.set_xlabel("online iteration")
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure5_off_support_gap(runs_by_cond: dict[str, list[Path]], out: Path) -> None:
    horizons = (1, 5, 10, 15)
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4), sharey=True)
    for ax, h in zip(axes, horizons):
        _plot_curve(ax, runs_by_cond, f"wm/gap_h{h}")
        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_title(f"horizon H={h}")
        ax.set_xlabel("online iteration")
    axes[0].set_ylabel("Gap_H = E_out[err] − E_in[err]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 5 — off-support generalization gap", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _trend(xs: np.ndarray, ys: np.ndarray) -> float:
    """Sign-encoded slope of a least-squares line. Returns slope coefficient."""
    if xs.size < 2:
        return 0.0
    slope, _ = np.polyfit(xs.astype(float), ys.astype(float), 1)
    return float(slope)


def _per_condition_summary(runs_by_cond: dict[str, list[Path]]) -> dict[str, dict[str, float]]:
    """Compute the §4.7 three-way collapse signature per condition."""
    out: dict[str, dict[str, float]] = {}
    for cond, paths in runs_by_cond.items():
        entropy_slopes = []
        gap_h5_finals = []
        visited_slopes = []
        underv_slopes = []
        trained_finals = []
        holdout_finals = []
        for p in paths:
            rows = _load_rows(p / "metrics.jsonl")
            xs, ys = _series(rows, "cov/visitation_entropy")
            if xs.size:
                entropy_slopes.append(_trend(xs, ys))
            xs, ys = _series(rows, "wm/gap_h5")
            if xs.size:
                gap_h5_finals.append(ys[-1])
            xs, ys = _series(rows, "wm/sem_err_h5_visited")
            if xs.size:
                visited_slopes.append(_trend(xs, ys))
            xs, ys = _series(rows, "wm/sem_err_h5_underv")
            if xs.size:
                underv_slopes.append(_trend(xs, ys))
            xs, ys = _series(rows, "beh/trained_success_rate")
            if xs.size:
                trained_finals.append(ys[-1])
            xs, ys = _series(rows, "beh/holdout_success_rate")
            if xs.size:
                holdout_finals.append(ys[-1])

        def stat(xs: list[float]) -> tuple[float, float]:
            if not xs:
                return float("nan"), float("nan")
            return float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0)

        out[cond] = {
            "n_seeds": len(paths),
            "entropy_slope_mean": stat(entropy_slopes)[0],
            "entropy_slope_sem": stat(entropy_slopes)[1],
            "gap_h5_final_mean": stat(gap_h5_finals)[0],
            "gap_h5_final_sem": stat(gap_h5_finals)[1],
            "visited_h5_slope_mean": stat(visited_slopes)[0],
            "underv_h5_slope_mean": stat(underv_slopes)[0],
            "trained_final_mean": stat(trained_finals)[0],
            "holdout_final_mean": stat(holdout_finals)[0],
            "success_drop_mean": stat(trained_finals)[0] - stat(holdout_finals)[0],
        }
    return out


def write_summary(summary: dict[str, dict[str, float]], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# World-collapse signature summary\n")
    lines.append("§4.7 says collapse is the JOINT appearance of three effects. ")
    lines.append("This table reports each condition's value for each effect; the headline ")
    lines.append("comparison is collapse_prone vs the two baselines (frozen_wm, balanced_replay).\n\n")
    lines.append("| Condition | n seeds | entropy slope (Δ/iter) | gap_h5 (final) | E_visited slope | E_underv slope | trained success (final) | holdout success (final) | shift drop |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cond, s in summary.items():
        lines.append(
            f"| {cond} | {int(s['n_seeds'])} "
            f"| {s['entropy_slope_mean']:+.4f} ± {s['entropy_slope_sem']:.4f} "
            f"| {s['gap_h5_final_mean']:+.3f} ± {s['gap_h5_final_sem']:.3f} "
            f"| {s['visited_h5_slope_mean']:+.4f} "
            f"| {s['underv_h5_slope_mean']:+.4f} "
            f"| {s['trained_final_mean']:.3f} "
            f"| {s['holdout_final_mean']:.3f} "
            f"| {s['success_drop_mean']:+.3f} |\n"
        )

    lines.append("\n### Reading the signature\n\n")
    lines.append("- **entropy slope** < 0 → visitation narrowing (proposal §4.5.1).\n")
    lines.append("- **gap_h5** > 0 and growing → world model is selectively worse off-support at horizon 5 (proposal §4.5.2).\n")
    lines.append("- **E_visited slope** ≈ 0 while **E_underv slope** > 0 → forgetting is localized, not uniform.\n")
    lines.append("- **shift drop** > 0 → behavioral consequence of goal shift (proposal §4.5.3).\n\n")
    lines.append("Collapse claim is supported iff collapse_prone shows all four AND the baselines do not.\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True, type=Path,
                        help="Directory containing per-run subdirs (or one nesting level deeper)")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    found = _find_runs(args.runs_dir)
    if not found:
        raise SystemExit(f"No <condition>-seed<N>/metrics.jsonl found under {args.runs_dir}")

    print("Found runs:")
    for cond, items in found.items():
        seeds = ", ".join(str(s) for s, _ in items)
        print(f"  {cond:>16}  seeds=[{seeds}]")

    runs_by_cond = {c: [p for _, p in items] for c, items in found.items()}

    figure1_visitation(runs_by_cond, args.out_dir / "figure1_visitation.png")
    figure2_probe_error(runs_by_cond, args.out_dir / "figure2_probe_error.png")
    figure4_goal_shift(runs_by_cond, args.out_dir / "figure4_goal_shift.png")
    figure5_off_support_gap(runs_by_cond, args.out_dir / "figure5_off_support_gap.png")

    summary = _per_condition_summary(runs_by_cond)
    write_summary(summary, args.out_dir / "collapse_signature.md")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nFigures + collapse_signature.md written to {args.out_dir}")


if __name__ == "__main__":
    main()
