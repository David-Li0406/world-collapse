"""Render figures from a run's metrics.jsonl. Thin CLI shim.

Usage:
    python scripts/make_plots.py --run_dir runs/<run_name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wcollapse.eval.plots import make_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    args = parser.parse_args()
    make_plots(args.run_dir)


if __name__ == "__main__":
    main()
