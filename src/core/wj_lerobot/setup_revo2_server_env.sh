#!/usr/bin/env bash
set -euo pipefail
REVO2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REVO2_PROJECT_ROOT"
REVO2_ENV_DIR="${REVO2_ENV_DIR:-$REVO2_PROJECT_ROOT/.venv-revo2}"
REVO2_BASE_PYTHON="${REVO2_BASE_PYTHON:-python3}"
if [[ ! -x "$REVO2_ENV_DIR/bin/python" ]]; then
  "$REVO2_BASE_PYTHON" -m venv "$REVO2_ENV_DIR"
fi
if [[ -n "${REVO2_TORCH_INDEX_URL:-}" ]]; then
  "$REVO2_ENV_DIR/bin/python" -m pip install --force-reinstall --index-url "$REVO2_TORCH_INDEX_URL" 'torch==2.7.1' 'torchvision==0.22.1'
fi
"$REVO2_ENV_DIR/bin/python" -m pip install --index-url "${REVO2_PIP_INDEX_URL:-https://pypi.org/simple}" -r src/core/wj_lerobot/requirements_revo2_server.txt
export PYTHONPATH="$REVO2_PROJECT_ROOT/src/open_source/lerobot/src:$REVO2_PROJECT_ROOT/src/open_source/transformers_4.57.1${PYTHONPATH:+:$PYTHONPATH}"
"$REVO2_ENV_DIR/bin/python" -m pip check
"$REVO2_ENV_DIR/bin/python" -c 'import torch, torchvision, transformers, fastapi; from lerobot.policies.pi05.processor_pi05 import Revo2RelativePoseProcessorStep; print("Environment ready; CUDA available:", torch.cuda.is_available())'
