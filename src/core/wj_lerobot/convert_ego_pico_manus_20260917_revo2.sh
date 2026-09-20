#!/usr/bin/env bash
set -euo pipefail
REVO2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REVO2_PROJECT_ROOT/src/open_source/lerobot/src:$REVO2_PROJECT_ROOT/src/open_source/transformers_4.57.1${PYTHONPATH:+:$PYTHONPATH}"
exec "${LEROBOT_PYTHON:-$REVO2_PROJECT_ROOT/.venv-revo2/bin/python}" "$REVO2_PROJECT_ROOT/src/core/wj_lerobot/dataset_tools/convert_revo2_rrd_to_lerobot_dino.py" \
  --source "${REVO2_RRD_SOURCE:-/zundong_ke/datasets/ego_pico_manus_2026-09-17_100}" \
  --output "${DATASET_DIR:-$REVO2_PROJECT_ROOT/datasets/ego_pico_manus_20260917_revo2_dino}" \
  --cache-dir "$REVO2_PROJECT_ROOT/outputs/revo2_conversion/cache" \
  --dino-model "${REVO2_DINO_MODEL:-/qichen_zhang/test_runs/Anyverse-VLA/models/grounding-dino-tiny}" \
  --dino-short-edge "${REVO2_DINO_SHORT_EDGE:-480}" \
  --task "${REVO2_TASK:-pick the cup into the box}" \
  --device "${REVO2_CONVERSION_DEVICE:-cpu}" \
  --workers "${REVO2_CONVERSION_WORKERS:-4}" \
  --cpu-threads "${REVO2_CONVERSION_CPU_THREADS:-4}" --batch-size "${REVO2_CONVERSION_BATCH_SIZE:-4}" "$@"
