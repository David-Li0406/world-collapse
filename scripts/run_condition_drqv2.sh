#!/usr/bin/env bash
# Phase B entry point for the DrQ-v2 + VideoPredictor pipeline.
# Loads (WM_0, Policy_0, D_real_R0) from a round-0 dir, runs one condition × seed.
#
# Required env:
#   CONDITION   collapse_prone | balanced_replay | frozen_wm
#   ROUND0_DIR  path to round-0 outputs
#   PROBE_BANK  path to shared probe_bank.h5
# Optional:
#   SEED, RUN_NAME, OUT_DIR
set -euo pipefail

: "${CONDITION:?CONDITION must be set}"
: "${ROUND0_DIR:?ROUND0_DIR must be set}"
: "${PROBE_BANK:?PROBE_BANK must be set}"

SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-${CONDITION}-seed${SEED}}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"

mkdir -p "${OUT_DIR}"

TMP_CFG="$(mktemp -t drqv2_cond.XXXXXX.yaml)"
python -c "
from omegaconf import OmegaConf
base = OmegaConf.load('configs/drqv2_base.yaml')
overlay = OmegaConf.load('configs/${CONDITION}_drqv2.yaml')
OmegaConf.save(OmegaConf.merge(base, overlay), '${TMP_CFG}')
"

export PYTHONPATH="${PYTHONPATH:-}:src:iVideoGPT/mbrl:iVideoGPT"
export IVIDEOGPT_ROOT="${IVIDEOGPT_ROOT:-$PWD/iVideoGPT}"
export MUJOCO_GL=osmesa

python -m wcollapse.training.online_drqv2 \
    --mode online \
    --condition "${CONDITION}" \
    --config "${TMP_CFG}" \
    --overrides seed=${SEED} \
    --output_dir "${OUT_DIR}" \
    --round0_dir "${ROUND0_DIR}" \
    --probe_bank_path "${PROBE_BANK}" \
    2>&1 | tee "${OUT_DIR}/train.log"

rm -f "${TMP_CFG}"
echo "Condition ${CONDITION} seed=${SEED} done: ${OUT_DIR}"
