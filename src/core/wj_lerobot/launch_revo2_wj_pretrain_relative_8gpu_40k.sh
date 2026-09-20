#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
export LEROBOT_PYTHON=${LEROBOT_PYTHON:-$PROJECT_ROOT/.venv-revo2/bin/python}
export PYTHONPATH="$PROJECT_ROOT/src/open_source/lerobot/src:$PROJECT_ROOT/src/open_source/transformers_4.57.1${PYTHONPATH:+:$PYTHONPATH}"
export REPO_ID=${REPO_ID:-ego_pico_manus_20260917_revo2_dino}
REVO2_WJ_STORAGE=${REVO2_WJ_STORAGE:-/wj-dataset}
REVO2_DATA_STORAGE=${REVO2_DATA_STORAGE:-/qichen_zhang}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$PROJECT_ROOT/datasets}
export DATASET_DIR=${DATASET_DIR:-$HF_LEROBOT_HOME/$REPO_ID}
# Selected Anyverse/WJ pretrained checkpoint; this initializes Revo2 training.
export PI05_BASE_DIR=${PI05_BASE_DIR:-$REVO2_WJ_STORAGE/vla_pretrain_model/WJ-Pretrain/RynnBrain-Backbone-2B/20260529_100046_pi05_finetune/checkpoints/060000/pretrained_model}
export PALIGEMMA_TOKENIZER_PATH=${PALIGEMMA_TOKENIZER_PATH:-$REVO2_WJ_STORAGE/lerobot/vla_pretrain_model/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}
export OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-$PROJECT_ROOT/outputs/revo2-wj-pretrain-relative-8gpu-40k}
[[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -eq 8 ]] || { echo "[ERROR] Expected 8 visible GPUs; refusing to start." >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export STEPS=40000
export SAVE_FREQ=10000
export BATCH_SIZE=20
export NUM_WORKERS=8
export LEROBOT_PARQUET_LOAD_WORKERS=4
export DTYPE=bfloat16
export TOLERANCE_S=0.01
export OPTIMIZER_LR=7.5e-5
export OPTIMIZER_WEIGHT_DECAY=0.001
export OPTIMIZER_GRAD_CLIP_NORM=1.0
export SCHEDULER_WARMUP_STEPS=500
export SCHEDULER_DECAY_STEPS=40000
export SCHEDULER_DECAY_LR=2.5e-6
export ACTION_SPACE=revo2_eef_pose
export ACTION_TARGET_MODE=relative_pose
export CAMERA_KEYS=top_head
export NORMALIZATION_MAPPING='{"ACTION":"MIN_MAX","STATE":"MIN_MAX","VISUAL":"IDENTITY"}'

# Match the KITT 40k run's auxiliary-loss and relative-pose settings.
export TASK_AUX_ENABLE=false
export BOX_AUX_ENABLE=false
export CROSS_CENTER_AUX_ENABLE=false
export FUTURE_ACTION_AUX_ENABLE=true
export FUTURE_ACTION_AUX_DIM=128
export FUTURE_ACTION_AUX_LOSS_WEIGHT=1.0
export FLOW_SOURCE_MODE=gaussian
export EE_POSITION_LOSS_WEIGHT=0.0
export EE_ORIENTATION_LOSS_WEIGHT=0.0
export POSE_TRANSLATION_LOSS_WEIGHT=0.0
export POSE_ROTATION_LOSS_WEIGHT=0.0
export POSE_LOSS_ACTIVE_SIDES='["left","right"]'
export SPLIT_ACTION_HEADS_ENABLE=true
# Arm poses: 0:18; all finger-joint targets: 18:30.
export GRIPPER_ACTION_INDICES='[18,19,20,21,22,23,24,25,26,27,28,29]'
export ACTION_LOSS_ACTIVE_INDICES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]'
export JOINT_ACTION_LOSS_WEIGHT=1.0
export GRIPPER_ACTION_LOSS_WEIGHT=1.0
export GRADIENT_CHECKPOINTING_ENABLE=true
export STATE_IN_PREFIX_TOKENS_ENABLE=true
export STATE_PREFIX_USE_HISTORY=false
export SHORT_TERM_MEMORY_NUM_FRAMES='[6]'
export SHORT_TERM_MEMORY_STRIDE='[5]'
export SHORT_TERM_MEMORY_INFERENCE_NUM_FRAMES=6
export SHORT_TERM_MEMORY_INCLUDE_PROPRIO=true
export SHORT_TERM_MEMORY_TEMPORAL_EVERY_N_LAYERS=999
export SHORT_TERM_MEMORY_DROP_HISTORY_LAST_N_LAYERS=4
export SHORT_TERM_MEMORY_CURRENT_TOKEN_KEEP_RATIO=1.0
export USE_RYNNBRAIN=true
export RYNNBRAIN_PATH=${RYNNBRAIN_PATH:-$REVO2_WJ_STORAGE/vla_pretrain_model/RynnBrain-2B}
export LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR=true
export ACTION_EXPERT_VARIANT=gemma_300m
export FREEZE_VISION_ENCODER=false
export TRAIN_EXPERT_ONLY=false
export IMAGE_TF_ENABLE=true
export IMAGE_TF_MAX_NUM_TRANSFORMS=3
export IMAGE_TF_RANDOM_ORDER=false
export WANDB_ENABLE=false
export DDP_FIND_UNUSED_PARAMETERS=true

cd "$PROJECT_ROOT/src/core/wj_lerobot"
exec bash ./training_wjvla.sh "$@"
