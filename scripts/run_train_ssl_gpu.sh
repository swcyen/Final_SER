#!/usr/bin/env bash
set -euo pipefail

# run_train_ssl_gpu.sh
# Usage:
#   ./scripts/run_train_ssl_gpu.sh [DATA_DIR] [EPOCHS] [BATCH_SIZE] [SAVE_DIR]
# Example:
#   ./scripts/run_train_ssl_gpu.sh ./local_datasets 10 64 ./checkpoints

DATA=${1:-./local_datasets}
EPOCHS=${2:-10}
BATCH_SIZE=${3:-64}
SAVE_DIR=${4:-./checkpoints}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-4}

echo "Running SSL training"
echo "  data: ${DATA}"
echo "  epochs: ${EPOCHS}"
echo "  batch-size: ${BATCH_SIZE}"
echo "  save-dir: ${SAVE_DIR}"
echo "  device: ${DEVICE}"

if [ "${DEVICE}" = "cuda" ] || [ "${DEVICE}" = "gpu" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi available, GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
  else
    echo "Warning: nvidia-smi not found. Ensure CUDA drivers are installed if you expect GPUs."
  fi
fi

# Run the training script. You can set environment variables to customize runtime:
#   PYTHON=/path/to/python  DEVICE=cuda  ./scripts/run_train_ssl_gpu.sh /data 20 128 /out

"${PYTHON}" train_ssl.py \
  --data "${DATA}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --save-dir "${SAVE_DIR}"
