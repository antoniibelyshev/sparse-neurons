#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-artifacts/mnist_long/ard_learned_spike_variance/model.pt}"
OUTPUT_DIR="${2:-$(dirname "${CHECKPOINT}")/diagnostics}"
DEVICE="${DEVICE:-auto}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Usage: scripts/plot_trained_run.sh [CHECKPOINT] [OUTPUT_DIR]" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/matplotlib-cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/matplotlib-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${OUTPUT_DIR}/uv-cache}"

uv run --no-sync evaluate-neuron-importance \
  "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"

echo "Diagnostics written to: ${OUTPUT_DIR}"
