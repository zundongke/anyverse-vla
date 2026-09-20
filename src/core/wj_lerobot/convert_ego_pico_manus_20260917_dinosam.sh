#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SAM2_VENDOR=${SAM2_VENDOR:-$PROJECT_DIR/outputs/revo2_dinosam/vendor}
export PYTHONPATH="$SAM2_VENDOR:$SAM2_VENDOR/hydra:$SAM2_VENDOR/omegaconf:$SAM2_VENDOR/portalocker${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_DIR=${DATASET_DIR:-$PROJECT_DIR/datasets/ego_pico_manus_20260917_revo2_dinosam}
exec bash "$PROJECT_DIR/src/core/wj_lerobot/convert_ego_pico_manus_20260917_revo2.sh" \
  --mask-backend sam2 \
  --sam2-checkpoint "${SAM2_CHECKPOINT:-/qichen_zhang/test_runs/Anyverse-VLA/models/sam2_hiera_tiny.pt}" \
  --detection-cache-dir "${DINO_DETECTION_CACHE:-$PROJECT_DIR/outputs/revo2_conversion/cache}" \
  --cache-dir "$PROJECT_DIR/outputs/revo2_dinosam/cache" "$@"
