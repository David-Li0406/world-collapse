#!/usr/bin/env bash
# Round-0 entry point for the DrQ-v2 + VideoPredictor pipeline.
# Trains (WM, DrQ-v2 policy, real buffer) from scratch on full push-v3 distribution.
# Output: runs/${RUN_NAME}/  containing model.pt, tokenizer.pt, snapshot.pt, buffer/, metrics.jsonl
set -euo pipefail

RUN_NAME="${RUN_NAME:-round0-v1}"
SEED="${SEED:-1}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"

mkdir -p "${OUT_DIR}"

# Merge base + round0 overlay (config has dotlist override at the CLI).
TMP_CFG="$(mktemp -t drqv2_round0.XXXXXX.yaml)"
python -c "
from omegaconf import OmegaConf
base = OmegaConf.load('configs/drqv2_base.yaml')
overlay = OmegaConf.load('configs/round0_drqv2.yaml')
OmegaConf.save(OmegaConf.merge(base, overlay), '${TMP_CFG}')
"

export PYTHONPATH="${PYTHONPATH:-}:src:iVideoGPT/mbrl:iVideoGPT"
export IVIDEOGPT_ROOT="${IVIDEOGPT_ROOT:-$PWD/iVideoGPT}"
export MUJOCO_GL=osmesa

python -m wcollapse.training.online_drqv2 \
    --mode round0 \
    --config "${TMP_CFG}" \
    --overrides seed=${SEED} \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/train.log"

rm -f "${TMP_CFG}"
echo "Round-0 done: ${OUT_DIR}"
