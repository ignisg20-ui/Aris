#!/usr/bin/env bash
# Convenience script to launch a multi-GPU pretraining run via torchrun.
set -euo pipefail

CONFIG=${1:-llm/configs/7b.yaml}
DATA=${2:-data/pretrain}
OUT=${3:-checkpoints/aris-7b}
NPROC=${NPROC:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)}

torchrun --standalone --nproc_per_node="${NPROC}" \
    -m llm.training.pretrain \
    --config "${CONFIG}" \
    --data "${DATA}" \
    --out "${OUT}" \
    --micro-batch "${MICRO_BATCH:-4}" \
    --accum "${ACCUM:-4}" \
    --steps "${STEPS:-100000}" \
    --tp "${TP:-${NPROC}}"
