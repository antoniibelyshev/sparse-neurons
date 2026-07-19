#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-auto}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cifar10_vgg}"
SEED="${SEED:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-100}"
ARD_EPOCHS="${ARD_EPOCHS:-300}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
BASELINE_LR="${BASELINE_LR:-0.001}"
ARD_LR="${ARD_LR:-0.001}"
BASELINE_WEIGHT_DECAY="${BASELINE_WEIGHT_DECAY:-0.0005}"
EMA_DECAY="${EMA_DECAY:-0.999}"
MIXTURE_SPIKE_VARIANCE="${MIXTURE_SPIKE_VARIANCE:-0.0001}"

BASELINE_DIR="${OUTPUT_ROOT}/baseline"
ARD_DIR="${OUTPUT_ROOT}/ard"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT_OVERRIDE:-${BASELINE_DIR}/final_ema_model.pt}"

rm -rf "${BASELINE_DIR}" "${ARD_DIR}"
mkdir -p "${OUTPUT_ROOT}/cache/matplotlib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/cache/matplotlib}"

if [[ -z "${PRETRAINED_CHECKPOINT_OVERRIDE:-}" ]]; then
  uv run train-cifar10-vgg-baseline \
    --epochs "${BASELINE_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${BASELINE_LR}" \
    --weight-decay "${BASELINE_WEIGHT_DECAY}" \
    --ema-decay "${EMA_DECAY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${BASELINE_DIR}" \
    --num-workers "${NUM_WORKERS}"
fi

uv run train-cifar10-vgg-ard \
  --epochs "${ARD_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${ARD_LR}" \
  --ema-decay "${EMA_DECAY}" \
  --initial-relative-std 0.01 \
  --mixture-spike-variance "${MIXTURE_SPIKE_VARIANCE}" \
  --kl-warmup-epochs "${KL_WARMUP_EPOCHS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${ARD_DIR}" \
  --num-workers "${NUM_WORKERS}" \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --checkpoint-every "${CHECKPOINT_EVERY}"

uv run evaluate-cifar10-vgg \
  "${ARD_DIR}/model.pt" \
  --output-dir "${ARD_DIR}/diagnostics" \
  --device "${DEVICE}"
