#!/usr/bin/env bash
# Launch all (condition × seed) jobs concurrently, packed onto N GPUs.
#
# Each job:
#   1) inherits a shared pretrain output (D_pre, probe bank, pretrained.pt)
#      via symlink — no copy, instant
#   2) gets its own runs/<condition>-seed<N>/ output dir
#   3) is pinned to a GPU via CUDA_VISIBLE_DEVICES = job_index mod NUM_GPUS
#
# With #jobs > NUM_GPUS, some GPU hosts two jobs concurrently. H20 has 96GB
# so two PyTorch processes per card fit comfortably; total wall-clock matches
# the theoretical optimum for an evenly-sized batch.
#
# Usage:
#   run_conditions.sh <shared_pretrain_dir> <seeds_csv> [conditions_csv] [num_gpus]
# Example:
#   run_conditions.sh ~/wcollapse-shared/pretrain-v1 0,1,2

set -euo pipefail

SHARED_PRETRAIN="${1:?shared pretrain dir required}"
SEEDS_CSV="${2:?seeds csv required}"
CONDITIONS_CSV="${3:-collapse_prone,frozen_wm,balanced_replay}"
NUM_GPUS="${4:-8}"

for f in ckpts/pretrained.pt data/d_pre.hdf5 data/probes.hdf5; do
  if [ ! -f "$SHARED_PRETRAIN/$f" ]; then
    echo "ERROR: missing $SHARED_PRETRAIN/$f" >&2
    exit 1
  fi
done

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra CONDS <<< "$CONDITIONS_CSV"

mkdir -p runs

# Pre-stage everyone's input dirs synchronously, then launch in one shot.
JOBS=()
for COND in "${CONDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_NAME="${COND}-seed${SEED}"
    OUT="runs/${RUN_NAME}"
    mkdir -p "$OUT/data" "$OUT/ckpts"
    ln -sfn "$SHARED_PRETRAIN/data/d_pre.hdf5"     "$OUT/data/d_pre.hdf5"
    ln -sfn "$SHARED_PRETRAIN/data/probes.hdf5"    "$OUT/data/probes.hdf5"
    ln -sfn "$SHARED_PRETRAIN/ckpts/pretrained.pt" "$OUT/ckpts/pretrained.pt"
    JOBS+=("$COND|$SEED|$OUT|$RUN_NAME")
  done
done

NUM_JOBS=${#JOBS[@]}
echo "[run_conditions] $NUM_JOBS jobs over $NUM_GPUS GPUs (conditions=${CONDS[*]}, seeds=${SEEDS[*]})"
echo "[run_conditions] shared pretrain: $SHARED_PRETRAIN"

PIDS=()
NAMES=()
INDEX=0
for SPEC in "${JOBS[@]}"; do
  IFS='|' read -r COND SEED OUT RUN_NAME <<< "$SPEC"
  GPU=$(( INDEX % NUM_GPUS ))
  echo "[run_conditions] launch $RUN_NAME on GPU $GPU"
  (
    CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python -u train.py \
      --config "configs/${COND}.yaml" \
      --output_dir "$OUT" \
      --override "seed=${SEED}" \
      --override "condition=${COND}" \
      > "$OUT/train.log" 2>&1
  ) &
  PIDS+=($!)
  NAMES+=("$RUN_NAME")
  INDEX=$((INDEX + 1))
done

# Wait for everyone. Collect failures into FAIL, but do NOT short-circuit —
# we want every run's train.log for diagnostics.
FAIL=0
for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  NAME="${NAMES[$i]}"
  if wait "$PID"; then
    echo "[run_conditions] $NAME OK"
  else
    EXIT=$?
    echo "[run_conditions] $NAME FAILED (exit=$EXIT)"
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "[run_conditions] one or more runs failed; see runs/<name>/train.log" >&2
  exit 1
fi

echo "[run_conditions] all $NUM_JOBS runs completed successfully"
