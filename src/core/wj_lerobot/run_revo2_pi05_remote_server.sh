#!/usr/bin/env bash
set -euo pipefail
REVO2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
REVO2_ENV_DIR="${REVO2_ENV_DIR:-$REVO2_PROJECT_ROOT/.venv-revo2}"
: "${REVO2_MODEL_DIR:?Set REVO2_MODEL_DIR to the trained Revo2 checkpoint directory}"
REVO2_DATASET_ROOT=${REVO2_DATASET_ROOT:-$REVO2_PROJECT_ROOT/datasets/ego_pico_manus_20260917_revo2_dino}
REVO2_DINO_MODEL=${REVO2_DINO_MODEL:-/qichen_zhang/test_runs/Anyverse-VLA/models/grounding-dino-tiny}
export PYTHONPATH="$REVO2_PROJECT_ROOT/src/open_source/lerobot/src:$REVO2_PROJECT_ROOT/src/open_source/transformers_4.57.1${PYTHONPATH:+:$PYTHONPATH}"
exec "$REVO2_ENV_DIR/bin/python" "$REVO2_PROJECT_ROOT/src/core/wj_lerobot/eval/pi05_remote_server.py" \
  --model-dir "$REVO2_MODEL_DIR" \
  --dataset-root "$REVO2_DATASET_ROOT" \
  --dataset-repo-id "${REVO2_DATASET_REPO_ID:-$(basename -- "$REVO2_DATASET_ROOT")}" \
  --dino-model "$REVO2_DINO_MODEL" \
  --dino-prompt "${REVO2_DINO_PROMPT:-robot arm . robot gripper .}" \
  --dino-threshold "${REVO2_DINO_THRESHOLD:-0.30}" \
  --dino-max-boxes "${REVO2_DINO_MAX_BOXES:-4}" \
  --dino-short-edge "${REVO2_DINO_SHORT_EDGE:-480}" \
  --default-task "${REVO2_TASK:-pick the cup into the box}" \
  --device "${REVO2_DEVICE:-cuda}" \
  --host "${REVO2_HOST:-0.0.0.0}" --port "${REVO2_PORT:-6007}" "$@"
