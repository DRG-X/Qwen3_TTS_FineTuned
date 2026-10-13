#!/usr/bin/env bash
# Reproduces the notebook's stages 4-6 outside Kaggle.  UNVERIFIED outside Kaggle (the original run used 2x T4).
# Usage: HTTS_CONFIG=configs/train_hindi_female.local.json bash scripts/run_pipeline.sh
set -euo pipefail
export HTTS_CONFIG="${HTTS_CONFIG:-configs/train_hindi_female.local.json}"
cd "$(dirname "$0")/../training"
export HTTS_CONFIG="$(cd .. && realpath "$HTTS_CONFIG")"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(nvidia-smi -L | wc -l)
python -u htts_prepare_data.py                                   # ~20-40 min on one GPU (notebook estimate)
if [ "$NGPU" -gt 1 ]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" htts_train.py
else
  CUDA_VISIBLE_DEVICES=0 python -u htts_train.py
fi
CUDA_VISIBLE_DEVICES=0 python -u htts_export.py                  # merges adapter_best.pt, writes + reloads + evaluates
