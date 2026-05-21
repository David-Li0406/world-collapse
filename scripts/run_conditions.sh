#!/usr/bin/env bash
# Launch all (variant × condition × seed) jobs concurrently, packed onto N GPUs.
#
# Variants are short names that expand to OmegaConf override flags:
#   base          (no overrides)
#   frozen_head   freeze the semantic head during Phase B (plan 1+2)
#   warmstart     50 imitation-warmup episodes + 2000 BC steps before SAC (plan 3)
#   unfreeze      unfreeze iVideoGPT tokenizer except codebook (plan 4)
#
# Each job inherits the shared pretrain output (D_pre, probe bank,
# pretrained.pt) via symlink, gets its own output dir, and is pinned to a
# GPU via CUDA_VISIBLE_DEVICES round-robin.
#
# Usage:
#   run_conditions.sh <shared_pretrain_dir> <seeds_csv> [conditions_csv] [num_gpus] [variants_csv]
# Example:
#   run_conditions.sh ~/wcollapse-shared/pretrain-v1 0 collapse_prone,frozen_wm,balanced_replay 8 \
#                     frozen_head,warmstart,unfreeze

set -euo pipefail

SHARED_PRETRAIN="${1:?shared pretrain dir required}"
SEEDS_CSV="${2:?seeds csv required}"
CONDITIONS_CSV="${3:-collapse_prone,frozen_wm,balanced_replay}"
NUM_GPUS="${4:-8}"
VARIANTS_CSV="${5:-base}"

for f in ckpts/pretrained.pt data/d_pre.hdf5 data/probes.hdf5; do
  if [ ! -f "$SHARED_PRETRAIN/$f" ]; then
    echo "ERROR: missing $SHARED_PRETRAIN/$f" >&2
    exit 1
  fi
done

# Map variant name -> space-separated --override args.
variant_overrides() {
  case "$1" in
    base)            echo "" ;;
    frozen_head)     echo "online.freeze_semantic_head=true" ;;
    warmstart)       echo "online.imitation_warmup_episodes=50 online.imitation_warmup_steps=2000" ;;
    unfreeze)        echo "wm.unfreeze_tokenizer=true wm.freeze_codebook_only=true" ;;
    # Variant `longer`: bigger sample budget — 150 online iter, 30 seed eps,
    # plus a light BC warmup so the actor starts from a non-trivial policy.
    longer)          echo "online.iterations=150 online.seed_episodes=30 online.imitation_warmup_episodes=20 online.imitation_warmup_steps=1000" ;;
    # Variant `aggressive_bias`: shrink the trained subregion to the first
    # 25% of the goal-x range (vs 50% default), and tighten the recency
    # window so the WM trains on a smaller slice of the actor's visitation.
    aggressive_bias) echo "online.goal_bias_fraction=0.25 online.recent_window=3000" ;;
    *) echo "ERROR: unknown variant: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra CONDS <<< "$CONDITIONS_CSV"
IFS=',' read -ra VARIANTS <<< "$VARIANTS_CSV"

mkdir -p runs

# Pre-stage all run dirs synchronously, then launch in one shot.
JOBS=()
for VAR in "${VARIANTS[@]}"; do
  for COND in "${CONDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if [ "$VAR" = "base" ]; then
        RUN_NAME="${COND}-seed${SEED}"
      else
        RUN_NAME="${VAR}-${COND}-seed${SEED}"
      fi
      OUT="runs/${RUN_NAME}"
      mkdir -p "$OUT/data" "$OUT/ckpts"
      ln -sfn "$SHARED_PRETRAIN/data/d_pre.hdf5"     "$OUT/data/d_pre.hdf5"
      ln -sfn "$SHARED_PRETRAIN/data/probes.hdf5"    "$OUT/data/probes.hdf5"
      ln -sfn "$SHARED_PRETRAIN/ckpts/pretrained.pt" "$OUT/ckpts/pretrained.pt"
      JOBS+=("$VAR|$COND|$SEED|$OUT|$RUN_NAME")
    done
  done
done

NUM_JOBS=${#JOBS[@]}
echo "[run_conditions] $NUM_JOBS jobs over $NUM_GPUS GPUs"
echo "[run_conditions]   variants=${VARIANTS[*]}, conditions=${CONDS[*]}, seeds=${SEEDS[*]}"
echo "[run_conditions]   shared pretrain: $SHARED_PRETRAIN"

PIDS=()
NAMES=()
INDEX=0
for SPEC in "${JOBS[@]}"; do
  IFS='|' read -r VAR COND SEED OUT RUN_NAME <<< "$SPEC"
  GPU=$(( INDEX % NUM_GPUS ))
  # Build the --override flag list for this variant.
  EXTRA_OVERRIDES=()
  for OV in $(variant_overrides "$VAR"); do
    EXTRA_OVERRIDES+=("--override" "$OV")
  done
  echo "[run_conditions] launch $RUN_NAME on GPU $GPU  (variant=$VAR, cond=$COND, seed=$SEED)"
  (
    CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python -u train.py \
      --config "configs/${COND}.yaml" \
      --output_dir "$OUT" \
      --override "seed=${SEED}" \
      --override "condition=${COND}" \
      "${EXTRA_OVERRIDES[@]}" \
      > "$OUT/train.log" 2>&1
  ) &
  PIDS+=($!)
  NAMES+=("$RUN_NAME")
  INDEX=$((INDEX + 1))
done

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
