Use GPU 4-7 in the Machine B

# GitHub Actions GPU Runner Instructions

This project uses two machines:

- Machine A: the development machine. It has the coding agent. Use it to write code, edit configs, trigger remote experiments, download results, and inspect logs.
- Machine B: the GPU machine. It is configured as a GitHub self-hosted runner. Use it to execute GitHub Actions workflows, prepare the Python environment, download or check datasets, run training jobs, and upload logs/results.

Machine A does not need direct SSH access to Machine B. Machine A controls remote execution through GitHub CLI (`gh`) and GitHub Actions.

---

## Core Principles

1. Machine A edits code and triggers workflows.
2. Machine B executes workflows.
3. Do not manually edit project code on Machine B.
4. Use `uv` for Python environment management.
5. Track Python dependencies with `pyproject.toml` and `uv.lock`.
6. Do not commit `.venv/` to Git.
7. Do not commit datasets to Git, except very small debug samples.
8. Machine B should download or check datasets inside the workflow.
9. Machine B should cache datasets in a persistent local directory.
10. Every experiment must write outputs to `runs/<run_name>/`.
11. Upload logs, metrics, config snapshots, and small result files as GitHub Actions artifacts.
12. Do not upload large checkpoints, datasets, or cache directories to GitHub artifacts unless explicitly requested.

---

## Required Commands on Machine A

Before running remote experiments, confirm that `gh` is the official GitHub CLI:

```bash
which gh
gh --version
```

Expected output should look like:

```text
/home/<user>/.local/bin/gh
gh version 2.x.x
```

Check GitHub authentication:

```bash
gh auth status
```

Check available workflows:

```bash
gh workflow list
```

Check whether the self-hosted runner is online:

```bash
gh api repos/David-Li0406/world-collapse/actions/runners \
  --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

If the runner is not `online`, do not trigger a new experiment. Tell the user to check Machine B and start the runner.

---

## Recommended Project Structure

```text
.
├── pyproject.toml
├── uv.lock
├── train.py
├── configs/
│   └── debug.yaml
├── scripts/
│   ├── prepare_data.sh
│   └── gpu_run.sh
├── .github/
│   └── workflows/
│       └── remote-gpu-exp.yml
└── runs/
```

The `runs/` directory is for experiment outputs and should not be committed to Git.

---

## Python Environment Management

This project uses `uv`.

Machine A may modify dependencies, but dependency changes must update the lock file.

To add a package:

```bash
uv add <package>
uv lock
```

If `pyproject.toml` is edited manually, run:

```bash
uv lock
```

Then commit the dependency files:

```bash
git add pyproject.toml uv.lock
git commit -m "Update Python environment"
git push
```

Machine B should use this command inside the workflow:

```bash
uv sync --frozen
```

This means Machine B must strictly follow `uv.lock` and should not re-resolve dependencies.

Never commit:

```text
.venv/
```

---

## Dataset Management

Datasets should not be stored in the Git repository, except for very small debug datasets.

Machine B should use a persistent dataset cache directory:

```bash
$HOME/datasets/world-collapse
```

The workflow should set:

```bash
export DATA_ROOT="$HOME/datasets/world-collapse"
```

For Hugging Face datasets and models, use persistent cache directories:

```bash
export HF_HOME="$HOME/hf_cache"
export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"
export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"
```

The dataset preparation script must be idempotent. If the dataset is already prepared, it should skip downloading.

Recommended `scripts/prepare_data.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/datasets/world-collapse}"
DATA_VERSION="${DATA_VERSION:-v1}"
DONE_FILE="$DATA_ROOT/.prepared-$DATA_VERSION"

mkdir -p "$DATA_ROOT"

if [ -f "$DONE_FILE" ]; then
  echo "Dataset already prepared: $DATA_ROOT, version: $DATA_VERSION"
  exit 0
fi

echo "Preparing dataset at $DATA_ROOT"

# If the project has a dataset download script, call it here:
# uv run python scripts/download_data.py --output "$DATA_ROOT"

# If the project uses Hugging Face datasets, the training script may download
# datasets automatically into the Hugging Face cache.
# Optional pre-download code can also be placed here.

rm -f "$DATA_ROOT"/.prepared-*
touch "$DONE_FILE"

echo "Dataset prepared."
```

If the dataset download or preprocessing logic changes, update `DATA_VERSION`, for example from `v1` to `v2`, so Machine B prepares the dataset again.

---

## Remote Experiment Workflow

Recommended workflow path:

```text
.github/workflows/remote-gpu-exp.yml
```

Recommended workflow name:

```text
remote-gpu-exp
```

The workflow must support manual triggering:

```yaml
on:
  workflow_dispatch:
```

Recommended input parameters:

- `config`: config file path, for example `configs/debug.yaml`
- `run_name`: experiment name, for example `debug-001`
- `data_version`: dataset version, for example `v1`

Example workflow:

```yaml
name: remote-gpu-exp

on:
  workflow_dispatch:
    inputs:
      config:
        description: "Config path"
        required: true
        default: "configs/debug.yaml"
      run_name:
        description: "Run name"
        required: true
        default: "debug"
      data_version:
        description: "Dataset version"
        required: true
        default: "v1"

jobs:
  run:
    runs-on: self-hosted

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Show machine info
        run: |
          hostname
          nvidia-smi || true

      - name: Setup uv environment
        run: |
          export UV_CACHE_DIR="$HOME/.cache/uv"
          uv sync --frozen

      - name: Prepare data
        run: |
          export DATA_ROOT="$HOME/datasets/world-collapse"
          export DATA_VERSION="${{ inputs.data_version }}"
          export HF_HOME="$HOME/hf_cache"
          export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"
          export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"

          ./scripts/prepare_data.sh

      - name: Run experiment
        run: |
          export DATA_ROOT="$HOME/datasets/world-collapse"
          export HF_HOME="$HOME/hf_cache"
          export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"
          export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"

          mkdir -p "runs/${{ inputs.run_name }}"

          uv run python train.py \
            --config "${{ inputs.config }}" \
            --data_root "$DATA_ROOT" \
            --output_dir "runs/${{ inputs.run_name }}" \
            2>&1 | tee "runs/${{ inputs.run_name }}/train.log"

      - name: Write summary
        if: always()
        run: |
          echo "## Experiment Summary" >> "$GITHUB_STEP_SUMMARY"
          echo "" >> "$GITHUB_STEP_SUMMARY"
          echo "- Config: ${{ inputs.config }}" >> "$GITHUB_STEP_SUMMARY"
          echo "- Run name: ${{ inputs.run_name }}" >> "$GITHUB_STEP_SUMMARY"
          echo "- Data version: ${{ inputs.data_version }}" >> "$GITHUB_STEP_SUMMARY"
          echo "- Commit: $(git rev-parse HEAD)" >> "$GITHUB_STEP_SUMMARY"
          echo "" >> "$GITHUB_STEP_SUMMARY"

          if [ -f "runs/${{ inputs.run_name }}/metrics.json" ]; then
            echo '```json' >> "$GITHUB_STEP_SUMMARY"
            cat "runs/${{ inputs.run_name }}/metrics.json" >> "$GITHUB_STEP_SUMMARY"
            echo '```' >> "$GITHUB_STEP_SUMMARY"
          fi

      - name: Upload outputs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: ${{ inputs.run_name }}
          path: |
            runs/${{ inputs.run_name }}/train.log
            runs/${{ inputs.run_name }}/metrics.json
            runs/${{ inputs.run_name }}/config.yaml
            runs/${{ inputs.run_name }}/*.txt
            runs/${{ inputs.run_name }}/plots/**
```

If the project does not yet generate `metrics.json`, the training script or `gpu_run.sh` should generate a minimal one so the agent can determine run status.

---

## Triggering a Remote Experiment

On Machine A, make sure the latest code is pushed:

```bash
git status
git add .
git commit -m "Update experiment"
git push
```

Then trigger the remote GPU workflow:

```bash
gh workflow run remote-gpu-exp.yml \
  -f config=configs/debug.yaml \
  -f run_name=debug-001 \
  -f data_version=v1
```

If the same dataset version should be reused, keep:

```bash
-f data_version=v1
```

If dataset download or preprocessing logic changed, use a new version:

```bash
gh workflow run remote-gpu-exp.yml \
  -f config=configs/debug.yaml \
  -f run_name=debug-002 \
  -f data_version=v2
```

---

## Checking Run Status

List recent workflow runs:

```bash
gh run list --workflow remote-gpu-exp.yml
```

Watch the current run:

```bash
gh run watch
```

View logs:

```bash
gh run view --log
```

View logs for a specific run:

```bash
gh run view <run-id> --log
```

---

## Downloading Experiment Results

List recent runs:

```bash
gh run list --workflow remote-gpu-exp.yml
```

Download artifacts for a specific run:

```bash
gh run download <run-id> --dir downloaded_runs/
```

Inspect downloaded files:

```bash
find downloaded_runs -type f
```

Read training logs:

```bash
tail -n 100 downloaded_runs/<run_name>/train.log
```

Read metrics:

```bash
cat downloaded_runs/<run_name>/metrics.json
```

---

## Recommended Debugging Workflow

During early development, prefer a small debug config:

```text
configs/debug.yaml
```

A debug config should:

- run very few training steps
- use a small batch size
- use a small dataset subset
- avoid saving large checkpoints
- finish within a few minutes

Typical loop:

```bash
git add .
git commit -m "Debug remote run"
git push

gh workflow run remote-gpu-exp.yml \
  -f config=configs/debug.yaml \
  -f run_name=debug-001 \
  -f data_version=v1

gh run watch
gh run view --log
```

If the run fails, download artifacts:

```bash
gh run download <run-id> --dir downloaded_runs/
```

Then inspect logs:

```bash
tail -n 200 downloaded_runs/<run_name>/train.log
```

Use the logs and metrics to modify code, then repeat the loop.

If there are any missing packages/tools in Machine B, report it to the user.

---

## Cache Strategy

Machine B is a self-hosted runner with persistent disk. Do not reinstall the environment or redownload datasets unnecessarily.

Environment cache:

```text
$HOME/.cache/uv
project .venv
```

Dataset cache:

```text
$HOME/datasets/world-collapse
```

Hugging Face cache:

```text
$HOME/hf_cache
```

Every workflow may run:

```bash
uv sync --frozen
./scripts/prepare_data.sh
```

If `uv.lock` has not changed, `uv sync --frozen` should be fast.

If the dataset has already been prepared, `prepare_data.sh` should skip downloading.

---

## Things Not To Do

Do not commit:

```text
.venv/
runs/
downloaded_runs/
__pycache__/
*.pt
*.pth
*.ckpt
*.safetensors
wandb/
data/
datasets/
```

Do not manually edit project code on Machine B.

Do not print tokens, API keys, or secrets in workflow logs.

Do not upload large checkpoints to GitHub artifacts unless explicitly requested.

Do not modify GitHub runner settings, repository secrets, or repository permissions unless explicitly requested by the user.

---

## If the Workflow Is Not Picked Up by Machine B

Check runner status:

```bash
gh api repos/David-Li0406/world-collapse/actions/runners \
  --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

If the runner is offline, ask the user to start it on Machine B:

```bash
cd ~/actions-runner
./run.sh
```

If the runner was installed as a service, ask the user to check or start the service:

```bash
sudo ./svc.sh status
sudo ./svc.sh start
```

If the workflow stays queued, check whether `runs-on` in `.github/workflows/remote-gpu-exp.yml` matches the runner labels.

The safest initial value is:

```yaml
runs-on: self-hosted
```

---

## Success Criteria

A successful remote experiment should satisfy all of the following:

1. Machine A has pushed the latest code to GitHub.
2. Machine A can trigger the workflow with `gh workflow run`.
3. Machine B self-hosted runner receives the job.
4. Machine B completes `uv sync --frozen`.
5. Machine B completes `scripts/prepare_data.sh`.
6. Machine B runs the training script.
7. The workflow uploads `train.log` and `metrics.json`.
8. Machine A can download the artifact with `gh run download`.
9. The coding agent can inspect logs and metrics, then continue improving the code.
