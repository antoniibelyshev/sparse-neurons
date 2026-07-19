#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-auto}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mnist_long}"
SEED="${SEED:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-100}"
ARD_EPOCHS="${ARD_EPOCHS:-300}"
KL_ZERO_EPOCHS="${KL_ZERO_EPOCHS:-0}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
BASELINE_LR="${BASELINE_LR:-0.01}"
BASELINE_WEIGHT_DECAY="${BASELINE_WEIGHT_DECAY:-0.001}"
ARD_LR="${ARD_LR:-0.01}"
EMA_DECAY="${EMA_DECAY:-0.999}"
MIXTURE_SPIKE_VARIANCE="${MIXTURE_SPIKE_VARIANCE:-0.0001}"

BASELINE_DIR="${OUTPUT_ROOT}/baseline"
ARD_DIR="${OUTPUT_ROOT}/ard_learned_spike_variance"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${BASELINE_DIR}/final_ema_model.pt}"

rm -rf "${BASELINE_DIR}" "${ARD_DIR}"
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/cache/matplotlib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/cache/matplotlib}"

if [[ -z "${PRETRAINED_CHECKPOINT_OVERRIDE:-}" ]]; then
  uv run train-mnist-baseline \
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
else
  PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT_OVERRIDE}"
fi

uv run train-mnist-ard \
  --epochs "${ARD_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${ARD_LR}" \
  --ema-decay "${EMA_DECAY}" \
  --initial-relative-std 0.01 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${ARD_DIR}" \
  --num-workers "${NUM_WORKERS}" \
  --kl-zero-epochs "${KL_ZERO_EPOCHS}" \
  --kl-warmup-epochs "${KL_WARMUP_EPOCHS}" \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --mixture-spike-variance "${MIXTURE_SPIKE_VARIANCE}" \
  --checkpoint-every "${CHECKPOINT_EVERY}"

uv run evaluate-neuron-importance \
  "${ARD_DIR}/model.pt" \
  --output-dir "${ARD_DIR}/neuron_importance" \
  --device "${DEVICE}"
