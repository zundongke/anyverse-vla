#!/usr/bin/env bash
# Revo2 relative-pose fine-tuning on the offline DINO + SAM2 dataset.
set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
export REPO_ID=${REPO_ID:-ego_pico_manus_20260917_revo2_dinosam}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$PROJECT_ROOT/datasets}
export DATASET_DIR=${DATASET_DIR:-$HF_LEROBOT_HOME/$REPO_ID}
export OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-$PROJECT_ROOT/outputs/revo2-dinosam-relative-8gpu-40k}
export ENABLE_SHORT_TERM_MEMORY=true

# Use this single dataset, without inherited multi-dataset or task selection.
export TRAINING_DATASET_CONFIG=""
export DATASET_TASK_ID_LIST=""

# Match the 50-step relative-normalization statistics in the conversion manifest.
# Additional training CLI arguments (including --resume) are passed through.
exec bash "$PROJECT_ROOT/src/core/wj_lerobot/launch_revo2_wj_pretrain_relative_8gpu_40k.sh" \
  --policy.chunk_size=50 --policy.n_action_steps=50 "$@"
