# world-collapse

Reproduce the **world collapse** experiment from `world_collapse_experiment.docx` (§4):
during closed-loop online training of a visual world model on Metaworld, expose the joint signature of
(i) shrinking policy visitation, (ii) selective off-support world-model degradation, and
(iii) downstream control failure on previously-known regions.

See `world_collapse_experiment.docx` for the proposal and `CLAUDE.md` for the two-machine
GitHub-Actions-driven dev workflow.

## Quickstart

```bash
uv sync --frozen
uv run python train.py --config configs/debug.yaml --output_dir runs/local-smoke
```

## Layout

```
src/wcollapse/   # package: envs, data, models, training, eval, utils
configs/         # YAML configs for each experimental condition
scripts/         # prepare_data.sh, make_plots.py
third_party/     # vendored upstream code (e.g., iVideoGPT) — added later
tests/           # unit tests (probe-bank determinism, etc.)
runs/            # experiment outputs (gitignored)
```
