#!/usr/bin/env bash
# Single-node, eight-GPU cloud entrypoint. Forward all training CLI arguments.
set -euo pipefail

# Change these paths to match the training container's persistent mounts.
# Environment variables supplied by the job take precedence over these defaults.
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
export LEROBOT_PYTHON=${LEROBOT_PYTHON:-$PROJECT_ROOT/.venv-revo2/bin/python}
export REPO_ID=${REPO_ID:-ego_pico_manus_20260917_revo2_dinosam}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$PROJECT_ROOT/datasets}
export DATASET_DIR=${DATASET_DIR:-$HF_LEROBOT_HOME/$REPO_ID}
export PI05_BASE_DIR=${PI05_BASE_DIR:-/wj-dataset/vla_pretrain_model/WJ-Pretrain/RynnBrain-Backbone-2B/20260529_100046_pi05_finetune/checkpoints/060000/pretrained_model}
export RYNNBRAIN_PATH=${RYNNBRAIN_PATH:-/wj-dataset/vla_pretrain_model/RynnBrain-2B}
export PALIGEMMA_TOKENIZER_PATH=${PALIGEMMA_TOKENIZER_PATH:-/wj-dataset/lerobot/vla_pretrain_model/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}
export OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-$PROJECT_ROOT/outputs/revo2-dinosam-relative-8gpu-40k}

# Launch this entrypoint once per job; it starts all eight workers itself.
export NNODES=1
export NODE_RANK=0
export NUM_GPUS=8
export NPROC_PER_NODE=8
export PYTHONUNBUFFERED=1

for name in PROJECT_ROOT DATASET_DIR PI05_BASE_DIR RYNNBRAIN_PATH PALIGEMMA_TOKENIZER_PATH; do
  if [[ ! -d "${!name}" ]]; then
    printf '[ERROR] %s directory not found: %s\n' "$name" "${!name}" >&2
    exit 2
  fi
done
if [[ ! -x "$LEROBOT_PYTHON" ]]; then
  printf '[ERROR] Python executable not found: %s\n' "$LEROBOT_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
  printf '[ERROR] Dataset metadata not found: %s/meta/info.json\n' "$DATASET_DIR" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [[ "${1:-}" == "--check-paths" ]]; then
  echo 'Path checks passed (Python dependencies and GPUs have not been checked).'
  exit 0
fi

exec bash src/core/wj_lerobot/launch_revo2_dinosam_relative_8gpu_40k.sh "$@"
