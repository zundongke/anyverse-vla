#!/usr/bin/env bash
set -euo pipefail

# Root of the mounted project directory.
# In your container image, mount Anyverse-VLA here and override PROJECT_ROOT if needed.
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/dataset/chao-liang/Anyverse-VLA}

# ====== User-configurable (override via env) ======
REPO_ID=${REPO_ID:-coffe_step1}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/mnt/dataset/wj-dataset/dataset_coffee/lerobot}
DATASET_DIR=${DATASET_DIR:-}
TRAINING_DATASET_CONFIG=${TRAINING_DATASET_CONFIG:-}
LEROBOT_PYTHON=${LEROBOT_PYTHON:-/opt/conda/envs/lerobot/bin/python}
DATASET_TASK_ID_LIST=${DATASET_TASK_ID_LIST:-}

DATASET_REPO_TO_ROOT_ARG=""
DATASET_REPO_TO_TASK_ID_ARG=""
DATASET_EPISODES_ARG=""
DATASET_TASK_NUM_CLASSES=""
DATASET_SELECTED_TASKS_ARG=""
DATASET_SELECTED_EPISODE_COUNT=""
DATASET_CONFIG_ENTRIES=()
if [[ -n "$TRAINING_DATASET_CONFIG" ]]; then
  if [[ ! -f "$TRAINING_DATASET_CONFIG" ]]; then
    echo "[ERROR] TRAINING_DATASET_CONFIG does not exist: $TRAINING_DATASET_CONFIG" >&2
    exit 1
  fi
  readarray -t DATASET_CONFIG_LINES < <("$LEROBOT_PYTHON" - "$TRAINING_DATASET_CONFIG" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
cfg = json.loads(cfg_path.read_text())
datasets = cfg["datasets"] if isinstance(cfg, dict) else cfg
if not isinstance(datasets, list) or len(datasets) == 0:
    raise ValueError("`training_dataset.json` must contain a non-empty dataset list.")

repo_ids = []
repo_to_root = {}
repo_to_task_id = {}
max_task_id = None
for item in datasets:
    if not isinstance(item, dict):
        raise ValueError("Each dataset entry must be an object.")
    repo_id = str(item.get("repo_id", "")).strip()
    root = str(item.get("root", "")).strip()
    if not repo_id or not root:
        raise ValueError("Each dataset entry requires `repo_id` and `root`.")
    repo_ids.append(repo_id)
    repo_to_root[repo_id] = root
    if "task_id" in item and item["task_id"] is not None:
        task_id = int(item["task_id"])
        if task_id < 0:
            raise ValueError(f"task_id must be non-negative for repo_id={repo_id}, got {task_id}")
        repo_to_task_id[repo_id] = task_id
        max_task_id = task_id if max_task_id is None else max(max_task_id, task_id)

if len(repo_ids) == 1:
    print(repo_ids[0])
else:
    print(json.dumps(repo_ids, separators=(",", ":")))
print(json.dumps(repo_to_root, separators=(",", ":")))
print(json.dumps(repo_to_task_id, separators=(",", ":")))
print("" if max_task_id is None else str(max_task_id + 1))
for repo_id in repo_ids:
    print(f"{repo_id}\t{repo_to_root[repo_id]}")
PY
  )
  DATASET_REPO_ARG="${DATASET_CONFIG_LINES[0]}"
  DATASET_REPO_TO_ROOT_ARG="${DATASET_CONFIG_LINES[1]}"
  DATASET_REPO_TO_TASK_ID_ARG="${DATASET_CONFIG_LINES[2]}"
  DATASET_TASK_NUM_CLASSES="${DATASET_CONFIG_LINES[3]}"
  DATASET_CONFIG_ENTRIES=("${DATASET_CONFIG_LINES[@]:4}")
  if [[ ${#DATASET_CONFIG_ENTRIES[@]} -gt 1 ]]; then
    IS_MULTI_REPO=1
    if [[ -z "$DATASET_DIR" ]]; then
      DATASET_DIR="${DATASET_CONFIG_ENTRIES[0]#*$'\t'}"
    fi
  else
    IS_MULTI_REPO=0
    SINGLE_REPO_ID="${DATASET_CONFIG_ENTRIES[0]%%$'\t'*}"
    SINGLE_REPO_ROOT="${DATASET_CONFIG_ENTRIES[0]#*$'\t'}"
    if [[ -f "$SINGLE_REPO_ROOT/meta/info.json" ]]; then
      DATASET_DIR="$SINGLE_REPO_ROOT"
    else
      DATASET_DIR="$SINGLE_REPO_ROOT/$SINGLE_REPO_ID"
    fi
  fi
else
  IFS=',' read -r -a REPO_ID_LIST_RAW <<< "$REPO_ID"
  REPO_ID_LIST=()
  for repo in "${REPO_ID_LIST_RAW[@]}"; do
    repo="${repo// /}"
    if [[ -n "$repo" ]]; then
      REPO_ID_LIST+=("$repo")
    fi
  done
  if [[ ${#REPO_ID_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] REPO_ID is empty." >&2
    exit 1
  fi
  if [[ ${#REPO_ID_LIST[@]} -gt 1 ]]; then
    IS_MULTI_REPO=1
  else
    IS_MULTI_REPO=0
    SINGLE_REPO_ID="${REPO_ID_LIST[0]}"
  fi
  if [[ -z "${DATASET_DIR:-}" ]]; then
    if [[ "$IS_MULTI_REPO" -eq 1 ]]; then
      DATASET_DIR="$HF_LEROBOT_HOME"
    else
      DATASET_DIR="$HF_LEROBOT_HOME/$SINGLE_REPO_ID"
    fi
  fi
  if [[ "$IS_MULTI_REPO" -eq 0 && -d "$DATASET_DIR/$SINGLE_REPO_ID/meta" && ! -f "$DATASET_DIR/meta/info.json" ]]; then
    DATASET_DIR="$DATASET_DIR/$SINGLE_REPO_ID"
  fi
  if [[ "$IS_MULTI_REPO" -eq 1 ]]; then
    DATASET_REPO_ARG="["
    for repo in "${REPO_ID_LIST[@]}"; do
      if [[ "$DATASET_REPO_ARG" != "[" ]]; then
        DATASET_REPO_ARG+=","
      fi
      DATASET_REPO_ARG+="\"$repo\""
    done
    DATASET_REPO_ARG+="]"
  else
    DATASET_REPO_ARG="$SINGLE_REPO_ID"
  fi
fi

# Optional: HuggingFace endpoint / proxy
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
http_proxy=${http_proxy:-http://192.168.100.117:18000}
https_proxy=${https_proxy:-http://192.168.100.117:18000}
HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}

# Local pretrained checkpoint directory (must exist)
PI05_BASE_DIR=${PI05_BASE_DIR:-$(ls -d /mnt/dataset/wj-dataset/lerobot/vla_pretrain_model/models--lerobot--pi05_base/snapshots/* 2>/dev/null | head -n 1)}

# Output directory base
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-/mnt/dataset/chao-liang/multi-task-debug}
LATEST_RUN_DIR_FILE=${LATEST_RUN_DIR_FILE:-$OUTPUT_BASE_DIR/latest_run_dir.txt}
RESUME_RUN_DIR=${RESUME_RUN_DIR:-}

# GPU / distributed selection
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
GPU_COUNT_FROM_CUDA_VISIBLE_DEVICES=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)
NPROC_PER_NODE=${NPROC_PER_NODE:-${NUM_GPUS:-$GPU_COUNT_FROM_CUDA_VISIBLE_DEVICES}}
NUM_GPUS=${NUM_GPUS:-$NPROC_PER_NODE}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# NUM_GPUS=${NUM_GPUS:-1}


# Training params
STEPS=${STEPS:-60000}
SAVE_FREQ=${SAVE_FREQ:-20000}
BATCH_SIZE=${BATCH_SIZE:-24}
NUM_WORKERS=${NUM_WORKERS:-8}
LEROBOT_PARQUET_LOAD_WORKERS=${LEROBOT_PARQUET_LOAD_WORKERS:-4}
DATASET_STREAMING=${DATASET_STREAMING:-false}
STREAMING_BUFFER_SIZE=${STREAMING_BUFFER_SIZE:-128}
DTYPE=${DTYPE:-bfloat16}
TOLERANCE_S=${TOLERANCE_S:-0.01}
STEP_PROFILE_ENABLE=${STEP_PROFILE_ENABLE:-false}
STEP_PROFILE_START=${STEP_PROFILE_START:-10}
STEP_PROFILE_STEPS=${STEP_PROFILE_STEPS:-20}
STEP_PROFILE_CUDA_SYNC=${STEP_PROFILE_CUDA_SYNC:-true}
CLIP_STATUS_CURVE_DUMP_DIR=${CLIP_STATUS_CURVE_DUMP_DIR:-}
CURRENT_PROGRESS_AS_INPUT_ENABLE=${CURRENT_PROGRESS_AS_INPUT_ENABLE:-false}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-true}
GRADIENT_CHECKPOINTING_ENABLE=${GRADIENT_CHECKPOINTING_ENABLE:-true}
OPTIMIZER_LR=${OPTIMIZER_LR:-7.5e-5}
OPTIMIZER_WEIGHT_DECAY=${OPTIMIZER_WEIGHT_DECAY:-0.001}
OPTIMIZER_GRAD_CLIP_NORM=${OPTIMIZER_GRAD_CLIP_NORM:-1.0}
OPTIMIZER_BACKBONE_LR_SCALE=${OPTIMIZER_BACKBONE_LR_SCALE:-1.0}
OPTIMIZER_HEAD_LR_SCALE=${OPTIMIZER_HEAD_LR_SCALE:-1.0}
SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS:-3000}
SCHEDULER_DECAY_STEPS=${SCHEDULER_DECAY_STEPS:-60000}
SCHEDULER_DECAY_LR=${SCHEDULER_DECAY_LR:-2.5e-6}
IMAGE_TF_ENABLE=${IMAGE_TF_ENABLE:-true}
IMAGE_TF_MAX_NUM_TRANSFORMS=${IMAGE_TF_MAX_NUM_TRANSFORMS:-3}
IMAGE_TF_RANDOM_ORDER=${IMAGE_TF_RANDOM_ORDER:-false}
STM_VIDEO_PROFILE_ENABLE=${STM_VIDEO_PROFILE_ENABLE:-false}
STM_VIDEO_PROFILE_INTERVAL=${STM_VIDEO_PROFILE_INTERVAL:-10}
QWEN_VISION_SDPA_BATCH_SEGMENTS=${QWEN_VISION_SDPA_BATCH_SEGMENTS:-1}
LEROBOT_ADAMW_FUSED=${LEROBOT_ADAMW_FUSED:-true}
NORMALIZATION_MAPPING=${NORMALIZATION_MAPPING:-'{"ACTION":"MIN_MAX","STATE":"MIN_MAX","VISUAL":"IDENTITY"}'}
ACTION_SPACE=${ACTION_SPACE:-joint}
ACTION_TARGET_MODE=${ACTION_TARGET_MODE:-absolute}
LANGUAGE_INCLUDE_STATE_IN_PROMPT=${LANGUAGE_INCLUDE_STATE_IN_PROMPT:-false}
TASK_AUX_ENABLE=${TASK_AUX_ENABLE:-true}
TASK_AUX_NUM_CLASSES=${TASK_AUX_NUM_CLASSES:-}
TASK_AUX_LOSS_WEIGHT=${TASK_AUX_LOSS_WEIGHT:-1.0}
BOX_AUX_ENABLE=${BOX_AUX_ENABLE:-false}
BOX_AUX_LOSS_WEIGHT=${BOX_AUX_LOSS_WEIGHT:-1.0}
CROSS_CENTER_AUX_ENABLE=${CROSS_CENTER_AUX_ENABLE:-false}
CROSS_CENTER_AUX_LOSS_WEIGHT=${CROSS_CENTER_AUX_LOSS_WEIGHT:-1.0}
FUTURE_ACTION_AUX_ENABLE=${FUTURE_ACTION_AUX_ENABLE:-true}
FUTURE_ACTION_AUX_DIM=${FUTURE_ACTION_AUX_DIM:-128}
FUTURE_ACTION_AUX_LOSS_WEIGHT=${FUTURE_ACTION_AUX_LOSS_WEIGHT:-1.0}
GROUP_CONSISTENCY_ENABLE=${GROUP_CONSISTENCY_ENABLE:-false}
GROUP_CONSISTENCY_CAMERA_KEYS=${GROUP_CONSISTENCY_CAMERA_KEYS:-}
GROUP_CONSISTENCY_ANCHOR_CAMERA_KEY=${GROUP_CONSISTENCY_ANCHOR_CAMERA_KEY:-}
GROUP_CONSISTENCY_PROJECT_DIM=${GROUP_CONSISTENCY_PROJECT_DIM:-256}
GROUP_CONSISTENCY_LOSS_WEIGHT=${GROUP_CONSISTENCY_LOSS_WEIGHT:-1.0}
BOX_CROSS_DEBUG_DUMP_DIR=${BOX_CROSS_DEBUG_DUMP_DIR:-}
BOX_CROSS_DEBUG_DUMP_INTERVAL=${BOX_CROSS_DEBUG_DUMP_INTERVAL:-0}
BOX_CROSS_DEBUG_CAMERA_KEY=${BOX_CROSS_DEBUG_CAMERA_KEY:-}
BOX_CROSS_DEBUG_MAX_SAMPLES_PER_BATCH=${BOX_CROSS_DEBUG_MAX_SAMPLES_PER_BATCH:-0}
INPUT_IMAGE_DEBUG_DUMP_DIR=${INPUT_IMAGE_DEBUG_DUMP_DIR:-}
INPUT_IMAGE_DEBUG_DUMP_INTERVAL=${INPUT_IMAGE_DEBUG_DUMP_INTERVAL:-0}
INPUT_IMAGE_DEBUG_MAX_SAMPLES_PER_BATCH=${INPUT_IMAGE_DEBUG_MAX_SAMPLES_PER_BATCH:-1}
INPUT_IMAGE_DEBUG_MAX_CAMERAS=${INPUT_IMAGE_DEBUG_MAX_CAMERAS:-3}
EE_POSITION_LOSS_WEIGHT=${EE_POSITION_LOSS_WEIGHT:-0.000005}
EE_ORIENTATION_LOSS_WEIGHT=${EE_ORIENTATION_LOSS_WEIGHT:-0.001}
POSE_TRANSLATION_LOSS_WEIGHT=${POSE_TRANSLATION_LOSS_WEIGHT:-0.0}
POSE_ROTATION_LOSS_WEIGHT=${POSE_ROTATION_LOSS_WEIGHT:-0.0}
POSE_LOSS_ACTIVE_SIDES=${POSE_LOSS_ACTIVE_SIDES:-'["left","right"]'}
SPLIT_ACTION_HEADS_ENABLE=${SPLIT_ACTION_HEADS_ENABLE:-true}
BEHAVIOR_B1K_SEMANTIC_ACTION_HEADS_ENABLE=${BEHAVIOR_B1K_SEMANTIC_ACTION_HEADS_ENABLE:-false}
GRIPPER_ACTION_INDICES=${GRIPPER_ACTION_INDICES:-[6,13]}
ACTION_LOSS_ACTIVE_INDICES=${ACTION_LOSS_ACTIVE_INDICES:-}
JOINT_ACTION_LOSS_WEIGHT=${JOINT_ACTION_LOSS_WEIGHT:-1.0}
GRIPPER_ACTION_LOSS_WEIGHT=${GRIPPER_ACTION_LOSS_WEIGHT:-1.0}
GRIPPER_TRANSITION_OVERSAMPLE_WEIGHT=${GRIPPER_TRANSITION_OVERSAMPLE_WEIGHT:-1.0}
GRIPPER_TRANSITION_THRESHOLD=${GRIPPER_TRANSITION_THRESHOLD:-10000}
GRIPPER_TRANSITION_WINDOW_BEFORE=${GRIPPER_TRANSITION_WINDOW_BEFORE:-30}
GRIPPER_TRANSITION_MIN_BEFORE=${GRIPPER_TRANSITION_MIN_BEFORE:-5}
GRIPPER_TRANSITION_WINDOW_AFTER=${GRIPPER_TRANSITION_WINDOW_AFTER:-0}
GRIPPER_TRANSITION_EXTRA_AUG_REPEATS=${GRIPPER_TRANSITION_EXTRA_AUG_REPEATS:-0}
GRIPPER_TRANSITION_DEBUG_DUMP_DIR=${GRIPPER_TRANSITION_DEBUG_DUMP_DIR:-}
GRIPPER_TRANSITION_DEBUG_MAX_SAMPLES=${GRIPPER_TRANSITION_DEBUG_MAX_SAMPLES:-0}
GRIPPER_TRANSITION_DEBUG_FUTURE_STEPS=${GRIPPER_TRANSITION_DEBUG_FUTURE_STEPS:-50}
FIRST_CLIP_ONLY_PER_DATA=${FIRST_CLIP_ONLY_PER_DATA:-false}
CAMERA_KEYS=${CAMERA_KEYS:-cam_high,wrist_left,wrist_right}
EGO_REPO_IDS=${EGO_REPO_IDS:-}
# Ego camera remapping is opt-in. Keep these empty unless the caller explicitly
# enables remapping for specific repos via EGO_REPO_IDS.
EGO_SOURCE_CAMERA_KEYS=${EGO_SOURCE_CAMERA_KEYS:-}
EGO_TARGET_CAMERA_KEY=${EGO_TARGET_CAMERA_KEY:-}
EGO_BLACK_CAMERA_KEYS=${EGO_BLACK_CAMERA_KEYS:-}
FLOW_SOURCE_MODE=${FLOW_SOURCE_MODE:-blend}
FLOW_SOURCE_STATE_NUM_FRAMES=${FLOW_SOURCE_STATE_NUM_FRAMES:-1}
FLOW_SOURCE_BLEND_ALPHA=${FLOW_SOURCE_BLEND_ALPHA:-0.3}
TRAIN_ROLLOUT_DEBUG_INTERVAL=${TRAIN_ROLLOUT_DEBUG_INTERVAL:-200}
STATE_IN_ACTION_TIME_EMB_ENABLE=${STATE_IN_ACTION_TIME_EMB_ENABLE:-false}
STATE_IN_ACTION_TIME_EMB_DEBUG_INTERVAL=${STATE_IN_ACTION_TIME_EMB_DEBUG_INTERVAL:-100}
STATE_IN_PREFIX_TOKENS_ENABLE=${STATE_IN_PREFIX_TOKENS_ENABLE:-true}
LANGUAGE_IN_PREFIX_TOKEN_ENABLE=${LANGUAGE_IN_PREFIX_TOKEN_ENABLE:-false}
STATE_PREFIX_USE_HISTORY=${STATE_PREFIX_USE_HISTORY:-false}
USE_RYNNBRAIN=${USE_RYNNBRAIN:-true}
RYNNBRAIN_PATH=${RYNNBRAIN_PATH:-/mnt/dataset/wj-dataset/vla_pretrain_model/RynnBrain-2B}
LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR=${LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR:-true}
RYNNBRAIN_LORA_ENABLE=${RYNNBRAIN_LORA_ENABLE:-false}
RYNNBRAIN_LORA_R=${RYNNBRAIN_LORA_R:-64}
RYNNBRAIN_LORA_ALPHA=${RYNNBRAIN_LORA_ALPHA:-128}
RYNNBRAIN_LORA_DROPOUT=${RYNNBRAIN_LORA_DROPOUT:-0.01}
RYNNBRAIN_LORA_TARGET_MODULES=${RYNNBRAIN_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}
ACTION_EXPERT_VARIANT=${ACTION_EXPERT_VARIANT:-gemma_300m}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-false}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-false}
VGGT_OMEGA_ENABLE=${VGGT_OMEGA_ENABLE:-false}
VGGT_OMEGA_CHECKPOINT=${VGGT_OMEGA_CHECKPOINT:-/mnt/dataset/wj-dataset/vla_pretrain_model/VGGT-OMEGA/VGGT-Omega-1B-256-Text-Alignment/model.pt}
VGGT_OMEGA_FREEZE=${VGGT_OMEGA_FREEZE:-true}
VGGT_OMEGA_CAMERA_KEYS=${VGGT_OMEGA_CAMERA_KEYS:-cam_high,wrist_left,wrist_right}
SUFFIX_ATTENTION_MASK=${SUFFIX_ATTENTION_MASK:-block}
PALIGEMMA_TOKENIZER_PATH=${PALIGEMMA_TOKENIZER_PATH:-/mnt/dataset/wj-dataset/lerobot/vla_pretrain_model/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}
ENABLE_SHORT_TERM_MEMORY=${ENABLE_SHORT_TERM_MEMORY:-true}
SHORT_TERM_MEMORY_NUM_FRAMES=${SHORT_TERM_MEMORY_NUM_FRAMES:-[2,2,6,6,6]}
SHORT_TERM_MEMORY_STRIDE=${SHORT_TERM_MEMORY_STRIDE:-[5,10,5,10,15]}
SHORT_TERM_MEMORY_INFERENCE_NUM_FRAMES=${SHORT_TERM_MEMORY_INFERENCE_NUM_FRAMES:-6}
SHORT_TERM_MEMORY_INCLUDE_PROPRIO=${SHORT_TERM_MEMORY_INCLUDE_PROPRIO:-true}
SHORT_TERM_MEMORY_TEMPORAL_EVERY_N_LAYERS=${SHORT_TERM_MEMORY_TEMPORAL_EVERY_N_LAYERS:-999}
SHORT_TERM_MEMORY_DROP_HISTORY_LAST_N_LAYERS=${SHORT_TERM_MEMORY_DROP_HISTORY_LAST_N_LAYERS:-4}
SHORT_TERM_MEMORY_DEBUG_INTERVAL=${SHORT_TERM_MEMORY_DEBUG_INTERVAL:-0}
SHORT_TERM_MEMORY_CURRENT_TOKEN_KEEP_RATIO=${SHORT_TERM_MEMORY_CURRENT_TOKEN_KEEP_RATIO:-1.0}
WANDB_ENABLE=${WANDB_ENABLE:-true}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_DISABLE_ARTIFACT=${WANDB_DISABLE_ARTIFACT:-true}
LANG_MISMATCH_ENABLE=${LANG_MISMATCH_ENABLE:-false}
LANG_MISMATCH_RATIO=${LANG_MISMATCH_RATIO:-0.3}
LANG_MISMATCH_MODE=${LANG_MISMATCH_MODE:-action_rank}
LANG_MISMATCH_MARGIN=${LANG_MISMATCH_MARGIN:-0.01}
LANG_MISMATCH_WEIGHT=${LANG_MISMATCH_WEIGHT:-0.3}
USE_BEHAVIOR_B1K_ADAPTER=${USE_BEHAVIOR_B1K_ADAPTER:-false}
BEHAVIOR_B1K_DEBUG_METRICS=${BEHAVIOR_B1K_DEBUG_METRICS:-false}

# ====== Environment exports ======
export HF_LEROBOT_HOME
export DATASET_DIR
export HF_ENDPOINT
export PYTORCH_CUDA_ALLOC_CONF
export QWEN_VISION_SDPA_BATCH_SEGMENTS
export LEROBOT_ADAMW_FUSED
export STM_VIDEO_PROFILE_ENABLE
export STM_VIDEO_PROFILE_INTERVAL
export PYTHONUNBUFFERED=1

if [[ -n "$http_proxy" ]]; then export http_proxy; fi
if [[ -n "$https_proxy" ]]; then export https_proxy; fi
if [[ -n "$HTTP_PROXY" ]]; then export HTTP_PROXY; fi
if [[ -n "$HTTPS_PROXY" ]]; then export HTTPS_PROXY; fi

if [[ -z "$TASK_AUX_NUM_CLASSES" ]]; then
  if [[ -n "$DATASET_TASK_NUM_CLASSES" ]]; then
    TASK_AUX_NUM_CLASSES="$DATASET_TASK_NUM_CLASSES"
  else
    TASK_AUX_NUM_CLASSES=100
  fi
fi

if [[ "$TASK_AUX_ENABLE" == "true" && "$TASK_AUX_NUM_CLASSES" =~ ^[0-9]+$ && "$TASK_AUX_NUM_CLASSES" -le 1 ]]; then
  echo "[WARN] TASK_AUX_NUM_CLASSES=$TASK_AUX_NUM_CLASSES, disabling task aux because task classification needs at least 2 classes."
  TASK_AUX_ENABLE=false
fi

if [[ -n "$DATASET_TASK_ID_LIST" ]]; then
  if [[ "$IS_MULTI_REPO" -eq 1 ]]; then
    echo "[ERROR] DATASET_TASK_ID_LIST currently only supports single-repo training." >&2
    exit 1
  fi

  readarray -t DATASET_TASK_SELECTION_LINES < <("$LEROBOT_PYTHON" - "$DATASET_DIR" "$DATASET_TASK_ID_LIST" <<'PY'
import json
import pathlib
import sys

import pandas as pd


def parse_task_ids(raw: str) -> list[int]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]

    if not isinstance(parsed, list):
        parsed = [parsed]

    task_ids = sorted({int(item) for item in parsed})
    if len(task_ids) == 0:
        raise ValueError("DATASET_TASK_ID_LIST is empty after parsing.")
    return task_ids


dataset_dir = pathlib.Path(sys.argv[1])
task_ids = parse_task_ids(sys.argv[2])
meta_dir = dataset_dir / "meta"
tasks_path = meta_dir / "tasks.parquet"
episode_paths = sorted((meta_dir / "episodes").glob("chunk-*/file-*.parquet"))

if not tasks_path.is_file():
    raise FileNotFoundError(f"Missing tasks metadata: {tasks_path}")
if len(episode_paths) == 0:
    raise FileNotFoundError(f"Missing episode metadata shards: {meta_dir / 'episodes'}")

task_df = pd.read_parquet(tasks_path).reset_index()
task_name_column = "task" if "task" in task_df.columns else task_df.columns[0]
task_index_to_name = {
    int(task_index): str(task_name)
    for task_index, task_name in zip(task_df["task_index"], task_df[task_name_column], strict=False)
}

missing_task_ids = sorted(set(task_ids).difference(task_index_to_name))
if missing_task_ids:
    raise ValueError(f"Unknown task ids: {missing_task_ids}")

selected_task_names = {task_index_to_name[task_id] for task_id in task_ids}
selected_episode_ids: list[int] = []
for episode_path in episode_paths:
    episode_df = pd.read_parquet(episode_path, columns=["episode_index", "tasks"])
    matched_episode_ids = episode_df.loc[
        episode_df["tasks"].apply(lambda tasks: any(str(task_name) in selected_task_names for task_name in tasks)),
        "episode_index",
    ].tolist()
    selected_episode_ids.extend(int(episode_id) for episode_id in matched_episode_ids)

selected_episode_ids = sorted(set(selected_episode_ids))
if len(selected_episode_ids) == 0:
    raise ValueError(f"No episodes matched task ids: {task_ids}")

print(json.dumps(task_ids, separators=(",", ":")))
print(json.dumps(selected_episode_ids, separators=(",", ":")))
print(str(len(selected_episode_ids)))
PY
  )
  DATASET_TASK_ID_LIST="${DATASET_TASK_SELECTION_LINES[0]}"
  DATASET_SELECTED_TASKS_ARG="${DATASET_TASK_SELECTION_LINES[0]}"
  DATASET_EPISODES_ARG="${DATASET_TASK_SELECTION_LINES[1]}"
  DATASET_SELECTED_EPISODE_COUNT="${DATASET_TASK_SELECTION_LINES[2]}"
fi

if [[ "$FIRST_CLIP_ONLY_PER_DATA" == "true" ]]; then
  if [[ -n "$DATASET_EPISODES_ARG" ]]; then
    readarray -t DATASET_FIRST_CLIP_LINES < <("$LEROBOT_PYTHON" - "$IS_MULTI_REPO" "$DATASET_EPISODES_ARG" <<'PY'
import json
import sys

is_multi = sys.argv[1] == "1"
episodes = json.loads(sys.argv[2])

if is_multi:
    first_clip_episodes = {}
    first_clip_counts = {}
    total_count = 0
    for repo_id, repo_episodes in episodes.items():
        if len(repo_episodes) == 0:
            raise ValueError(f"FIRST_CLIP_ONLY_PER_DATA=true but no selected episodes found for repo_id={repo_id}")
        first_clip_episodes[str(repo_id)] = [int(repo_episodes[0])]
        first_clip_counts[str(repo_id)] = 1
        total_count += 1
    first_clip_counts["__total__"] = total_count
    print(json.dumps(first_clip_episodes, separators=(",", ":")))
    print(json.dumps(first_clip_counts, separators=(",", ":")))
else:
    if len(episodes) == 0:
        raise ValueError("FIRST_CLIP_ONLY_PER_DATA=true but selected episode list is empty.")
    print(json.dumps([int(episodes[0])], separators=(",", ":")))
    print("1")
PY
    )
    DATASET_EPISODES_ARG="${DATASET_FIRST_CLIP_LINES[0]}"
    DATASET_SELECTED_EPISODE_COUNT="${DATASET_FIRST_CLIP_LINES[1]}"
  else
    if [[ "$IS_MULTI_REPO" -eq 1 ]]; then
      DATASET_EPISODES_ARG="{"
      DATASET_SELECTED_EPISODE_COUNT="{"
      FIRST_CLIP_REPO_COUNT=0
      if [[ -n "$TRAINING_DATASET_CONFIG" ]]; then
        for entry in "${DATASET_CONFIG_ENTRIES[@]}"; do
          repo="${entry%%$'\t'*}"
          if [[ "$DATASET_EPISODES_ARG" != "{" ]]; then
            DATASET_EPISODES_ARG+=","
          fi
          if [[ "$DATASET_SELECTED_EPISODE_COUNT" != "{" ]]; then
            DATASET_SELECTED_EPISODE_COUNT+=","
          fi
          DATASET_EPISODES_ARG+="\"$repo\":[0]"
          DATASET_SELECTED_EPISODE_COUNT+="\"$repo\":1"
          FIRST_CLIP_REPO_COUNT=$((FIRST_CLIP_REPO_COUNT + 1))
        done
      else
        for repo in "${REPO_ID_LIST[@]}"; do
          if [[ "$DATASET_EPISODES_ARG" != "{" ]]; then
            DATASET_EPISODES_ARG+=","
          fi
          if [[ "$DATASET_SELECTED_EPISODE_COUNT" != "{" ]]; then
            DATASET_SELECTED_EPISODE_COUNT+=","
          fi
          DATASET_EPISODES_ARG+="\"$repo\":[0]"
          DATASET_SELECTED_EPISODE_COUNT+="\"$repo\":1"
          FIRST_CLIP_REPO_COUNT=$((FIRST_CLIP_REPO_COUNT + 1))
        done
      fi
      DATASET_EPISODES_ARG+="}"
      if [[ "$DATASET_SELECTED_EPISODE_COUNT" != "{" ]]; then
        DATASET_SELECTED_EPISODE_COUNT+=","
      fi
      DATASET_SELECTED_EPISODE_COUNT+="\"__total__\":$FIRST_CLIP_REPO_COUNT}"
    else
      DATASET_EPISODES_ARG="[0]"
      DATASET_SELECTED_EPISODE_COUNT="1"
    fi
  fi
fi

if [[ "$RYNNBRAIN_LORA_ENABLE" == "true" ]]; then
  if [[ "$USE_RYNNBRAIN" != "true" ]]; then
    echo "[ERROR] RYNNBRAIN_LORA_ENABLE=true requires USE_RYNNBRAIN=true" >&2
    exit 1
  fi
  if [[ "$TRAIN_EXPERT_ONLY" == "true" ]]; then
    echo "[ERROR] RYNNBRAIN_LORA_ENABLE=true is incompatible with TRAIN_EXPERT_ONLY=true" >&2
    exit 1
  fi
  if [[ "$LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR" != "false" ]]; then
    echo "[WARN] RYNNBRAIN_LORA_ENABLE=true: forcing LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR=false to keep the open-source RynnBrain base."
    LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR=false
  fi
  if ! "$LEROBOT_PYTHON" -c "import peft" >/dev/null 2>&1; then
    echo "[ERROR] RYNNBRAIN_LORA_ENABLE=true requires the Python package 'peft' in $LEROBOT_PYTHON" >&2
    exit 1
  fi
fi

NORMALIZED_CAMERA_KEYS=""
if [[ -n "$CAMERA_KEYS" ]]; then
  if [[ "$CAMERA_KEYS" =~ ^\[.*\]$ ]]; then
    NORMALIZED_CAMERA_KEYS="$CAMERA_KEYS"
  else
    IFS=',' read -r -a CAMERA_KEY_LIST_RAW <<< "$CAMERA_KEYS"
    CAMERA_KEY_LIST=()
    for camera_key in "${CAMERA_KEY_LIST_RAW[@]}"; do
      camera_key="${camera_key// /}"
      if [[ -n "$camera_key" ]]; then
        CAMERA_KEY_LIST+=("$camera_key")
      fi
    done
    if [[ ${#CAMERA_KEY_LIST[@]} -eq 0 ]]; then
      echo "[ERROR] CAMERA_KEYS is empty after parsing: $CAMERA_KEYS" >&2
      exit 1
    fi
    NORMALIZED_CAMERA_KEYS="["
    for camera_key in "${CAMERA_KEY_LIST[@]}"; do
      if [[ "$NORMALIZED_CAMERA_KEYS" != "[" ]]; then
        NORMALIZED_CAMERA_KEYS+=","
      fi
      NORMALIZED_CAMERA_KEYS+="\"$camera_key\""
    done
    NORMALIZED_CAMERA_KEYS+="]"
  fi
fi

NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS=""
GROUP_CONSISTENCY_GROUP_SIZE=0
if [[ -n "$GROUP_CONSISTENCY_CAMERA_KEYS" ]]; then
  if [[ "$GROUP_CONSISTENCY_CAMERA_KEYS" =~ ^\[.*\]$ ]]; then
    NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS="$GROUP_CONSISTENCY_CAMERA_KEYS"
    GROUP_CONSISTENCY_KEYS_SPEC="${GROUP_CONSISTENCY_CAMERA_KEYS#[}"
    GROUP_CONSISTENCY_KEYS_SPEC="${GROUP_CONSISTENCY_KEYS_SPEC%]}"
    IFS=',' read -r -a GROUP_CONSISTENCY_KEY_LIST_RAW <<< "$GROUP_CONSISTENCY_KEYS_SPEC"
    GROUP_CONSISTENCY_KEY_LIST=()
    for camera_key in "${GROUP_CONSISTENCY_KEY_LIST_RAW[@]}"; do
      camera_key="${camera_key// /}"
      camera_key="${camera_key%\"}"
      camera_key="${camera_key#\"}"
      camera_key="${camera_key%\'}"
      camera_key="${camera_key#\'}"
      if [[ -n "$camera_key" ]]; then
        GROUP_CONSISTENCY_KEY_LIST+=("$camera_key")
      fi
    done
  else
    IFS=',' read -r -a GROUP_CONSISTENCY_KEY_LIST_RAW <<< "$GROUP_CONSISTENCY_CAMERA_KEYS"
    GROUP_CONSISTENCY_KEY_LIST=()
    for camera_key in "${GROUP_CONSISTENCY_KEY_LIST_RAW[@]}"; do
      camera_key="${camera_key// /}"
      if [[ -n "$camera_key" ]]; then
        GROUP_CONSISTENCY_KEY_LIST+=("$camera_key")
      fi
    done
    if [[ ${#GROUP_CONSISTENCY_KEY_LIST[@]} -gt 0 ]]; then
      NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS="["
      for camera_key in "${GROUP_CONSISTENCY_KEY_LIST[@]}"; do
        if [[ "$NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS" != "[" ]]; then
          NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS+=","
        fi
        NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS+="\"$camera_key\""
      done
      NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS+="]"
    fi
  fi
  GROUP_CONSISTENCY_GROUP_SIZE=${#GROUP_CONSISTENCY_KEY_LIST[@]}
fi

normalize_optional_string_list() {
  local raw_value="${1:-}"
  if [[ -z "$raw_value" ]]; then
    echo ""
    return
  fi
  if [[ "$raw_value" =~ ^\[.*\]$ ]]; then
    echo "$raw_value"
    return
  fi
  local IFS=','
  read -r -a raw_items <<< "$raw_value"
  local items=()
  local item=""
  for item in "${raw_items[@]}"; do
    item="${item// /}"
    if [[ -n "$item" ]]; then
      items+=("$item")
    fi
  done
  if [[ ${#items[@]} -eq 0 ]]; then
    echo ""
    return
  fi
  local normalized="["
  for item in "${items[@]}"; do
    if [[ "$normalized" != "[" ]]; then
      normalized+=","
    fi
    normalized+="\"$item\""
  done
  normalized+="]"
  echo "$normalized"
}

NORMALIZED_EGO_REPO_IDS="$(normalize_optional_string_list "$EGO_REPO_IDS")"
NORMALIZED_EGO_SOURCE_CAMERA_KEYS="$(normalize_optional_string_list "$EGO_SOURCE_CAMERA_KEYS")"
NORMALIZED_EGO_BLACK_CAMERA_KEYS="$(normalize_optional_string_list "$EGO_BLACK_CAMERA_KEYS")"

TRAIN_BATCH_SIZE="$BATCH_SIZE"
if [[ "$GROUP_CONSISTENCY_ENABLE" == "true" ]]; then
  if [[ "$GROUP_CONSISTENCY_GROUP_SIZE" -lt 2 ]]; then
    echo "[ERROR] GROUP_CONSISTENCY_ENABLE=true requires at least 2 GROUP_CONSISTENCY_CAMERA_KEYS" >&2
    exit 1
  fi
  if [[ "$BATCH_SIZE" -le 0 || $((BATCH_SIZE % GROUP_CONSISTENCY_GROUP_SIZE)) -ne 0 ]]; then
    echo "[ERROR] BATCH_SIZE=$BATCH_SIZE must be divisible by GROUP_CONSISTENCY group size=$GROUP_CONSISTENCY_GROUP_SIZE" >&2
    exit 1
  fi
  TRAIN_BATCH_SIZE=$((BATCH_SIZE / GROUP_CONSISTENCY_GROUP_SIZE))
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/open_source/transformers_4.57.1:$PROJECT_ROOT/src/open_source/lerobot/src:${PYTHONPATH:-}"
export NPROC_PER_NODE
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export LEROBOT_PARQUET_LOAD_WORKERS
export DDP_FIND_UNUSED_PARAMETERS

# ====== Sanity checks ======
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  echo "        Please mount Anyverse-VLA to this path or set PROJECT_ROOT." >&2
  exit 1
fi

if [[ "$IS_MULTI_REPO" -eq 1 ]]; then
  if [[ -n "$TRAINING_DATASET_CONFIG" ]]; then
    for entry in "${DATASET_CONFIG_ENTRIES[@]}"; do
      repo="${entry%%$'\t'*}"
      root="${entry#*$'\t'}"
      if [[ -f "$root/meta/info.json" ]]; then
        resolved_dir="$root"
      else
        resolved_dir="$root/$repo"
      fi
      if [[ ! -f "$resolved_dir/meta/info.json" ]]; then
        echo "[ERROR] Missing dataset metadata: $resolved_dir/meta/info.json" >&2
        exit 1
      fi
    done
  else
    if [[ ! -d "$DATASET_DIR" ]]; then
      echo "[ERROR] DATASET_DIR does not exist: $DATASET_DIR" >&2
      echo "        Set HF_LEROBOT_HOME/DATASET_DIR or mount the dataset into the container." >&2
      exit 1
    fi
    for repo in "${REPO_ID_LIST[@]}"; do
      if [[ ! -d "$DATASET_DIR/$repo" ]]; then
        echo "[ERROR] Dataset repo does not exist: $DATASET_DIR/$repo" >&2
        exit 1
      fi
      if [[ ! -f "$DATASET_DIR/$repo/meta/info.json" ]]; then
        echo "[ERROR] Missing dataset metadata: $DATASET_DIR/$repo/meta/info.json" >&2
        exit 1
      fi
    done
  fi
else
  if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[ERROR] DATASET_DIR does not exist: $DATASET_DIR" >&2
    echo "        Set HF_LEROBOT_HOME/DATASET_DIR or mount the dataset into the container." >&2
    exit 1
  fi
  if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
    echo "[ERROR] Missing dataset metadata: $DATASET_DIR/meta/info.json" >&2
    echo "        If DATASET_DIR is a parent directory, set REPO_ID correctly or unset DATASET_DIR." >&2
    exit 1
  fi
fi

if [[ -z "$PI05_BASE_DIR" || ! -d "$PI05_BASE_DIR" ]]; then
  echo "[ERROR] 找不到本地PI05 Pretrained Model,请设置 PI05_BASE_DIR" >&2
  exit 1
fi

if [[ "$VGGT_OMEGA_ENABLE" == "true" ]]; then
  if [[ -z "$VGGT_OMEGA_CHECKPOINT" ]]; then
    echo "[ERROR] VGGT_OMEGA_ENABLE=true but VGGT_OMEGA_CHECKPOINT is empty." >&2
    echo "        Please set VGGT_OMEGA_CHECKPOINT to a valid VGGT-Omega text-alignment checkpoint." >&2
    exit 1
  fi
  if [[ ! -f "$VGGT_OMEGA_CHECKPOINT" ]]; then
    echo "[ERROR] VGGT_OMEGA_ENABLE=true but VGGT_OMEGA_CHECKPOINT does not exist: $VGGT_OMEGA_CHECKPOINT" >&2
    exit 1
  fi
fi

if [[ ! -x "$LEROBOT_PYTHON" ]]; then
  echo "[ERROR] Python not found: $LEROBOT_PYTHON" >&2
  exit 1
fi

echo "Launching PI05 fine-tuning with ${NUM_GPUS} GPUs"
echo "Using project:  $PROJECT_ROOT"
echo "Using dataset:  $DATASET_DIR"
echo "Using repo_id:   $DATASET_REPO_ARG"
echo "Using PI05_BASE: $PI05_BASE_DIR"
echo "LEROBOT_PYTHON=$LEROBOT_PYTHON"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "LEROBOT_DATASET_MATERIALIZATION_CACHE=${LEROBOT_DATASET_MATERIALIZATION_CACHE:-}"
echo "LEROBOT_DATASET_MATERIALIZATION_CACHE_DIR=${LEROBOT_DATASET_MATERIALIZATION_CACHE_DIR:-}"
echo "NUM_GPUS=$NUM_GPUS"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "LEROBOT_PARQUET_LOAD_WORKERS=$LEROBOT_PARQUET_LOAD_WORKERS"
echo "DATASET_STREAMING=$DATASET_STREAMING"
echo "TOLERANCE_S=$TOLERANCE_S"
echo "STEP_PROFILE_ENABLE=$STEP_PROFILE_ENABLE"
echo "STEP_PROFILE_START=$STEP_PROFILE_START"
echo "STEP_PROFILE_STEPS=$STEP_PROFILE_STEPS"
echo "STEP_PROFILE_CUDA_SYNC=$STEP_PROFILE_CUDA_SYNC"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "http_proxy=$http_proxy"
echo "https_proxy=$https_proxy"
echo "FIRST_CLIP_ONLY_PER_DATA=$FIRST_CLIP_ONLY_PER_DATA"
if [[ -n "$NORMALIZED_CAMERA_KEYS" ]]; then
  echo "CAMERA_KEYS=$NORMALIZED_CAMERA_KEYS"
fi
if [[ -n "$NORMALIZED_EGO_REPO_IDS" ]]; then
  echo "EGO_REPO_IDS=$NORMALIZED_EGO_REPO_IDS"
fi
if [[ -n "$NORMALIZED_EGO_SOURCE_CAMERA_KEYS" ]]; then
  echo "EGO_SOURCE_CAMERA_KEYS=$NORMALIZED_EGO_SOURCE_CAMERA_KEYS"
fi
if [[ -n "$NORMALIZED_EGO_BLACK_CAMERA_KEYS" ]]; then
  echo "EGO_BLACK_CAMERA_KEYS=$NORMALIZED_EGO_BLACK_CAMERA_KEYS"
fi
echo "GROUP_CONSISTENCY_ENABLE=$GROUP_CONSISTENCY_ENABLE"
if [[ -n "$NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS" ]]; then
  echo "GROUP_CONSISTENCY_CAMERA_KEYS=$NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS"
fi
echo "GROUP_CONSISTENCY_ANCHOR_CAMERA_KEY=$GROUP_CONSISTENCY_ANCHOR_CAMERA_KEY"
echo "GROUP_CONSISTENCY_PROJECT_DIM=$GROUP_CONSISTENCY_PROJECT_DIM"
echo "GROUP_CONSISTENCY_LOSS_WEIGHT=$GROUP_CONSISTENCY_LOSS_WEIGHT"
echo "EFFECTIVE_BATCH_SIZE=$BATCH_SIZE"
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "RYNNBRAIN_PATH=$RYNNBRAIN_PATH"
echo "RYNNBRAIN_LORA_ENABLE=$RYNNBRAIN_LORA_ENABLE"
echo "RYNNBRAIN_LORA_R=$RYNNBRAIN_LORA_R"
echo "RYNNBRAIN_LORA_ALPHA=$RYNNBRAIN_LORA_ALPHA"
echo "RYNNBRAIN_LORA_DROPOUT=$RYNNBRAIN_LORA_DROPOUT"
echo "RYNNBRAIN_LORA_TARGET_MODULES=$RYNNBRAIN_LORA_TARGET_MODULES"
echo "FREEZE_VISION_ENCODER=$FREEZE_VISION_ENCODER"
echo "TRAIN_EXPERT_ONLY=$TRAIN_EXPERT_ONLY"
echo "ACTION_EXPERT_VARIANT=$ACTION_EXPERT_VARIANT"
echo "VGGT_OMEGA_ENABLE=$VGGT_OMEGA_ENABLE"
echo "VGGT_OMEGA_CHECKPOINT=$VGGT_OMEGA_CHECKPOINT"
echo "VGGT_OMEGA_FREEZE=$VGGT_OMEGA_FREEZE"
echo "VGGT_OMEGA_CAMERA_KEYS=$VGGT_OMEGA_CAMERA_KEYS"
echo "PALIGEMMA_TOKENIZER_PATH=$PALIGEMMA_TOKENIZER_PATH"
echo "ENABLE_SHORT_TERM_MEMORY=$ENABLE_SHORT_TERM_MEMORY"
echo "SHORT_TERM_MEMORY_NUM_FRAMES=$SHORT_TERM_MEMORY_NUM_FRAMES"
echo "SHORT_TERM_MEMORY_STRIDE=$SHORT_TERM_MEMORY_STRIDE"
echo "SHORT_TERM_MEMORY_INFERENCE_NUM_FRAMES=$SHORT_TERM_MEMORY_INFERENCE_NUM_FRAMES"
echo "SHORT_TERM_MEMORY_INCLUDE_PROPRIO=$SHORT_TERM_MEMORY_INCLUDE_PROPRIO"
echo "SHORT_TERM_MEMORY_TEMPORAL_EVERY_N_LAYERS=$SHORT_TERM_MEMORY_TEMPORAL_EVERY_N_LAYERS"
echo "SHORT_TERM_MEMORY_DROP_HISTORY_LAST_N_LAYERS=$SHORT_TERM_MEMORY_DROP_HISTORY_LAST_N_LAYERS"
echo "SHORT_TERM_MEMORY_DEBUG_INTERVAL=$SHORT_TERM_MEMORY_DEBUG_INTERVAL"
echo "SHORT_TERM_MEMORY_CURRENT_TOKEN_KEEP_RATIO=$SHORT_TERM_MEMORY_CURRENT_TOKEN_KEEP_RATIO"
echo "EE_POSITION_LOSS_WEIGHT=$EE_POSITION_LOSS_WEIGHT"
echo "EE_ORIENTATION_LOSS_WEIGHT=$EE_ORIENTATION_LOSS_WEIGHT"
echo "POSE_TRANSLATION_LOSS_WEIGHT=$POSE_TRANSLATION_LOSS_WEIGHT"
echo "POSE_ROTATION_LOSS_WEIGHT=$POSE_ROTATION_LOSS_WEIGHT"
echo "POSE_LOSS_ACTIVE_SIDES=$POSE_LOSS_ACTIVE_SIDES"
echo "ACTION_SPACE=$ACTION_SPACE"
echo "SPLIT_ACTION_HEADS_ENABLE=$SPLIT_ACTION_HEADS_ENABLE"
echo "BEHAVIOR_B1K_SEMANTIC_ACTION_HEADS_ENABLE=$BEHAVIOR_B1K_SEMANTIC_ACTION_HEADS_ENABLE"
echo "GRIPPER_ACTION_INDICES=$GRIPPER_ACTION_INDICES"
echo "ACTION_LOSS_ACTIVE_INDICES=$ACTION_LOSS_ACTIVE_INDICES"
echo "JOINT_ACTION_LOSS_WEIGHT=$JOINT_ACTION_LOSS_WEIGHT"
echo "GRIPPER_ACTION_LOSS_WEIGHT=$GRIPPER_ACTION_LOSS_WEIGHT"
echo "GRIPPER_TRANSITION_OVERSAMPLE_WEIGHT=$GRIPPER_TRANSITION_OVERSAMPLE_WEIGHT"
echo "GRIPPER_TRANSITION_THRESHOLD=$GRIPPER_TRANSITION_THRESHOLD"
echo "GRIPPER_TRANSITION_WINDOW_BEFORE=$GRIPPER_TRANSITION_WINDOW_BEFORE"
echo "GRIPPER_TRANSITION_MIN_BEFORE=$GRIPPER_TRANSITION_MIN_BEFORE"
echo "GRIPPER_TRANSITION_WINDOW_AFTER=$GRIPPER_TRANSITION_WINDOW_AFTER"
echo "GRIPPER_TRANSITION_EXTRA_AUG_REPEATS=$GRIPPER_TRANSITION_EXTRA_AUG_REPEATS"
echo "GRIPPER_TRANSITION_DEBUG_DUMP_DIR=$GRIPPER_TRANSITION_DEBUG_DUMP_DIR"
echo "GRIPPER_TRANSITION_DEBUG_MAX_SAMPLES=$GRIPPER_TRANSITION_DEBUG_MAX_SAMPLES"
echo "GRIPPER_TRANSITION_DEBUG_FUTURE_STEPS=$GRIPPER_TRANSITION_DEBUG_FUTURE_STEPS"
echo "FLOW_SOURCE_MODE=$FLOW_SOURCE_MODE"
echo "FLOW_SOURCE_STATE_NUM_FRAMES=$FLOW_SOURCE_STATE_NUM_FRAMES"
echo "FLOW_SOURCE_BLEND_ALPHA=$FLOW_SOURCE_BLEND_ALPHA"
echo "TRAIN_ROLLOUT_DEBUG_INTERVAL=$TRAIN_ROLLOUT_DEBUG_INTERVAL"
echo "INPUT_IMAGE_DEBUG_DUMP_DIR=$INPUT_IMAGE_DEBUG_DUMP_DIR"
echo "INPUT_IMAGE_DEBUG_DUMP_INTERVAL=$INPUT_IMAGE_DEBUG_DUMP_INTERVAL"
echo "INPUT_IMAGE_DEBUG_MAX_SAMPLES_PER_BATCH=$INPUT_IMAGE_DEBUG_MAX_SAMPLES_PER_BATCH"
echo "INPUT_IMAGE_DEBUG_MAX_CAMERAS=$INPUT_IMAGE_DEBUG_MAX_CAMERAS"
echo "FUTURE_ACTION_AUX_ENABLE=$FUTURE_ACTION_AUX_ENABLE"
echo "FUTURE_ACTION_AUX_DIM=$FUTURE_ACTION_AUX_DIM"
echo "FUTURE_ACTION_AUX_LOSS_WEIGHT=$FUTURE_ACTION_AUX_LOSS_WEIGHT"
echo "USE_BEHAVIOR_B1K_ADAPTER=$USE_BEHAVIOR_B1K_ADAPTER"
echo "BEHAVIOR_B1K_DEBUG_METRICS=$BEHAVIOR_B1K_DEBUG_METRICS"
if [[ -n "$DATASET_TASK_ID_LIST" ]]; then
  echo "DATASET_TASK_ID_LIST=$DATASET_TASK_ID_LIST"
fi
if [[ -n "$DATASET_SELECTED_TASKS_ARG" ]]; then
  echo "DATASET_SELECTED_TASKS_ARG=$DATASET_SELECTED_TASKS_ARG"
fi
if [[ -n "$DATASET_SELECTED_EPISODE_COUNT" ]]; then
  echo "DATASET_SELECTED_EPISODE_COUNT=$DATASET_SELECTED_EPISODE_COUNT"
fi
if [[ -n "$DATASET_EPISODES_ARG" ]]; then
  echo "DATASET_EPISODES_ARG=$DATASET_EPISODES_ARG"
fi
echo "PYTHONPATH=$PYTHONPATH"

# ====== Run from wj_lerobot directory ======
cd "$PROJECT_ROOT/src/core/wj_lerobot"

echo "LeRobot module path: $("$LEROBOT_PYTHON" -c 'import lerobot; print(lerobot.__file__)')"

# ====== Train ======
mkdir -p "$OUTPUT_BASE_DIR"
RUN_DIR="$OUTPUT_BASE_DIR/$(date +%Y%m%d_%H%M%S)_WJVLA_finetune"

RESUME_REQUESTED=false
HAS_CONFIG_PATH_ARG=false
HAS_OUTPUT_DIR_ARG=false
EXPECT_CONFIG_PATH_VALUE=false
EXPECT_OUTPUT_DIR_VALUE=false
CLI_ARGS=()
for arg in "$@"; do
  if [[ "$EXPECT_CONFIG_PATH_VALUE" == "true" ]]; then
    HAS_CONFIG_PATH_ARG=true
    EXPECT_CONFIG_PATH_VALUE=false
    CLI_ARGS+=("$arg")
    continue
  fi
  if [[ "$EXPECT_OUTPUT_DIR_VALUE" == "true" ]]; then
    HAS_OUTPUT_DIR_ARG=true
    EXPECT_OUTPUT_DIR_VALUE=false
    CLI_ARGS+=("$arg")
    continue
  fi
  case "$arg" in
    --resume)
      RESUME_REQUESTED=true
      CLI_ARGS+=("--resume=true")
      ;;
    --resume=true|--resume=1)
      RESUME_REQUESTED=true
      CLI_ARGS+=("--resume=true")
      ;;
    --resume=false|--resume=0)
      CLI_ARGS+=("--resume=false")
      ;;
    --config_path)
      HAS_CONFIG_PATH_ARG=true
      EXPECT_CONFIG_PATH_VALUE=true
      CLI_ARGS+=("$arg")
      ;;
    --config_path=*)
      HAS_CONFIG_PATH_ARG=true
      CLI_ARGS+=("$arg")
      ;;
    --output_dir)
      HAS_OUTPUT_DIR_ARG=true
      EXPECT_OUTPUT_DIR_VALUE=true
      CLI_ARGS+=("$arg")
      ;;
    --output_dir=*)
      HAS_OUTPUT_DIR_ARG=true
      CLI_ARGS+=("$arg")
      ;;
    *)
      CLI_ARGS+=("$arg")
      ;;
  esac
done

if [[ "$RESUME_REQUESTED" == "true" ]]; then
  if [[ -z "$RESUME_RUN_DIR" && -f "$LATEST_RUN_DIR_FILE" ]]; then
    RESUME_RUN_DIR=$(<"$LATEST_RUN_DIR_FILE")
  fi
  if [[ -z "$RESUME_RUN_DIR" ]]; then
    echo "[ERROR] --resume requested but no RESUME_RUN_DIR is set and $LATEST_RUN_DIR_FILE does not exist." >&2
    exit 1
  fi
  if [[ ! -d "$RESUME_RUN_DIR" ]]; then
    echo "[ERROR] RESUME_RUN_DIR does not exist: $RESUME_RUN_DIR" >&2
    exit 1
  fi
  RESUME_CONFIG_PATH="$RESUME_RUN_DIR/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "$RESUME_CONFIG_PATH" ]]; then
    echo "[ERROR] Resume config not found: $RESUME_CONFIG_PATH" >&2
    echo "        Expected checkpoint layout: $RESUME_RUN_DIR/checkpoints/last/pretrained_model/train_config.json" >&2
    exit 1
  fi
  RUN_DIR="$RESUME_RUN_DIR"
  if [[ "$HAS_CONFIG_PATH_ARG" == "false" ]]; then
    CLI_ARGS+=("--config_path=$RESUME_CONFIG_PATH")
  fi
  if [[ "$HAS_OUTPUT_DIR_ARG" == "false" ]]; then
    CLI_ARGS+=("--output_dir=$RUN_DIR")
  fi
else
  printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_DIR_FILE"
fi

echo "Outputs will be written to: $RUN_DIR"

DATASET_EXTRA_ARGS=()
if [[ -n "$DATASET_REPO_TO_ROOT_ARG" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.repo_id_to_root=$DATASET_REPO_TO_ROOT_ARG")
fi
if [[ -n "$DATASET_EPISODES_ARG" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.episodes=$DATASET_EPISODES_ARG")
fi
if [[ -n "$NORMALIZED_CAMERA_KEYS" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.camera_keys=$NORMALIZED_CAMERA_KEYS")
fi
if [[ -n "$NORMALIZED_EGO_REPO_IDS" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.ego_repo_ids=$NORMALIZED_EGO_REPO_IDS")
fi
if [[ -n "$NORMALIZED_EGO_SOURCE_CAMERA_KEYS" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.ego_source_camera_keys=$NORMALIZED_EGO_SOURCE_CAMERA_KEYS")
fi
if [[ -n "$EGO_TARGET_CAMERA_KEY" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.ego_target_camera_key=$EGO_TARGET_CAMERA_KEY")
fi
if [[ -n "$NORMALIZED_EGO_BLACK_CAMERA_KEYS" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.ego_black_camera_keys=$NORMALIZED_EGO_BLACK_CAMERA_KEYS")
fi
if [[ "$USE_BEHAVIOR_B1K_ADAPTER" == "true" ]]; then
  DATASET_EXTRA_ARGS+=("--dataset.use_behavior_b1k_adapter=true")
fi
DATASET_EXTRA_ARGS+=("--dataset.streaming=$DATASET_STREAMING")
DATASET_EXTRA_ARGS+=("--dataset.streaming_buffer_size=$STREAMING_BUFFER_SIZE")

POLICY_EXTRA_ARGS=()
if [[ -n "$CLIP_STATUS_CURVE_DUMP_DIR" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.clip_running_status_curve_dump_dir=$CLIP_STATUS_CURVE_DUMP_DIR")
fi
if [[ -n "$DATASET_REPO_TO_TASK_ID_ARG" && "$DATASET_REPO_TO_TASK_ID_ARG" != "{}" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.task_id_map=$DATASET_REPO_TO_TASK_ID_ARG")
fi
POLICY_EXTRA_ARGS+=("--policy.use_rynnbrain=$USE_RYNNBRAIN")
POLICY_EXTRA_ARGS+=("--policy.rynnbrain_path=$RYNNBRAIN_PATH")
POLICY_EXTRA_ARGS+=("--policy.enable_rynnbrain_lora=$RYNNBRAIN_LORA_ENABLE")
POLICY_EXTRA_ARGS+=("--policy.rynnbrain_lora_r=$RYNNBRAIN_LORA_R")
POLICY_EXTRA_ARGS+=("--policy.rynnbrain_lora_alpha=$RYNNBRAIN_LORA_ALPHA")
POLICY_EXTRA_ARGS+=("--policy.rynnbrain_lora_dropout=$RYNNBRAIN_LORA_DROPOUT")
POLICY_EXTRA_ARGS+=("--policy.rynnbrain_lora_target_modules=$RYNNBRAIN_LORA_TARGET_MODULES")
POLICY_EXTRA_ARGS+=("--policy.enable_vggt_omega_prefix_token=$VGGT_OMEGA_ENABLE")
POLICY_EXTRA_ARGS+=("--policy.vggt_omega_checkpoint_path=$VGGT_OMEGA_CHECKPOINT")
POLICY_EXTRA_ARGS+=("--policy.vggt_omega_freeze=$VGGT_OMEGA_FREEZE")
POLICY_EXTRA_ARGS+=("--policy.vggt_omega_camera_keys=$VGGT_OMEGA_CAMERA_KEYS")
POLICY_EXTRA_ARGS+=("--policy.suffix_attention_mask=$SUFFIX_ATTENTION_MASK")
POLICY_EXTRA_ARGS+=("--policy.enable_short_term_memory=$ENABLE_SHORT_TERM_MEMORY")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_num_frames=$SHORT_TERM_MEMORY_NUM_FRAMES")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_stride=$SHORT_TERM_MEMORY_STRIDE")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_inference_num_frames=$SHORT_TERM_MEMORY_INFERENCE_NUM_FRAMES")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_include_proprio=$SHORT_TERM_MEMORY_INCLUDE_PROPRIO")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_temporal_every_n_layers=$SHORT_TERM_MEMORY_TEMPORAL_EVERY_N_LAYERS")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_drop_history_last_n_layers=$SHORT_TERM_MEMORY_DROP_HISTORY_LAST_N_LAYERS")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_debug_interval=$SHORT_TERM_MEMORY_DEBUG_INTERVAL")
POLICY_EXTRA_ARGS+=("--policy.short_term_memory_current_token_keep_ratio=$SHORT_TERM_MEMORY_CURRENT_TOKEN_KEEP_RATIO")
POLICY_EXTRA_ARGS+=("--policy.enable_split_action_heads=$SPLIT_ACTION_HEADS_ENABLE")
POLICY_EXTRA_ARGS+=("--policy.enable_behavior_b1k_semantic_action_heads=$BEHAVIOR_B1K_SEMANTIC_ACTION_HEADS_ENABLE")
POLICY_EXTRA_ARGS+=("--policy.behavior_b1k_debug_metrics=$BEHAVIOR_B1K_DEBUG_METRICS")
POLICY_EXTRA_ARGS+=("--policy.enable_group_consistency_loss=$GROUP_CONSISTENCY_ENABLE")
POLICY_EXTRA_ARGS+=("--policy.group_consistency_project_dim=$GROUP_CONSISTENCY_PROJECT_DIM")
POLICY_EXTRA_ARGS+=("--policy.group_consistency_loss_weight=$GROUP_CONSISTENCY_LOSS_WEIGHT")
POLICY_EXTRA_ARGS+=("--policy.joint_action_loss_weight=$JOINT_ACTION_LOSS_WEIGHT")
POLICY_EXTRA_ARGS+=("--policy.gripper_action_loss_weight=$GRIPPER_ACTION_LOSS_WEIGHT")
POLICY_EXTRA_ARGS+=("--policy.pose_translation_loss_weight=$POSE_TRANSLATION_LOSS_WEIGHT")
POLICY_EXTRA_ARGS+=("--policy.pose_rotation_loss_weight=$POSE_ROTATION_LOSS_WEIGHT")
POLICY_EXTRA_ARGS+=("--policy.pose_loss_active_sides=$POSE_LOSS_ACTIVE_SIDES")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_oversample_weight=$GRIPPER_TRANSITION_OVERSAMPLE_WEIGHT")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_threshold=$GRIPPER_TRANSITION_THRESHOLD")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_window_before=$GRIPPER_TRANSITION_WINDOW_BEFORE")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_min_before=$GRIPPER_TRANSITION_MIN_BEFORE")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_window_after=$GRIPPER_TRANSITION_WINDOW_AFTER")
POLICY_EXTRA_ARGS+=("--policy.optimizer_backbone_lr_scale=$OPTIMIZER_BACKBONE_LR_SCALE")
POLICY_EXTRA_ARGS+=("--policy.optimizer_head_lr_scale=$OPTIMIZER_HEAD_LR_SCALE")
POLICY_EXTRA_ARGS+=("--policy.enable_language_in_prefix_token=$LANGUAGE_IN_PREFIX_TOKEN_ENABLE")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_extra_aug_repeats=$GRIPPER_TRANSITION_EXTRA_AUG_REPEATS")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_debug_max_samples=$GRIPPER_TRANSITION_DEBUG_MAX_SAMPLES")
POLICY_EXTRA_ARGS+=("--policy.gripper_transition_debug_future_steps=$GRIPPER_TRANSITION_DEBUG_FUTURE_STEPS")
if [[ -n "$GRIPPER_TRANSITION_DEBUG_DUMP_DIR" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.gripper_transition_debug_dump_dir=$GRIPPER_TRANSITION_DEBUG_DUMP_DIR")
fi
if [[ -n "$GRIPPER_ACTION_INDICES" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.gripper_action_indices=$GRIPPER_ACTION_INDICES")
fi
if [[ -n "$ACTION_LOSS_ACTIVE_INDICES" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.action_loss_active_indices=$ACTION_LOSS_ACTIVE_INDICES")
fi
if [[ -n "$PALIGEMMA_TOKENIZER_PATH" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.paligemma_tokenizer_path=$PALIGEMMA_TOKENIZER_PATH")
fi
if [[ -n "$NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.group_consistency_camera_keys=$NORMALIZED_GROUP_CONSISTENCY_CAMERA_KEYS")
fi
if [[ -n "$GROUP_CONSISTENCY_ANCHOR_CAMERA_KEY" ]]; then
  POLICY_EXTRA_ARGS+=("--policy.group_consistency_anchor_camera_key=$GROUP_CONSISTENCY_ANCHOR_CAMERA_KEY")
fi

TRAIN_CMD=(
  "$LEROBOT_PYTHON" -m torch.distributed.run
)

if [[ "$NNODES" -gt 1 ]]; then
  TRAIN_CMD+=(
    --nnodes="$NNODES"
    --nproc_per_node="$NPROC_PER_NODE"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
  )
else
  TRAIN_CMD+=(
    --standalone
    --nnodes=1
    --nproc_per_node="$NPROC_PER_NODE"
  )
fi

TRAIN_CMD+=(
  -m lerobot.scripts.lerobot_train
  --dataset.repo_id="$DATASET_REPO_ARG"
  --dataset.root="$DATASET_DIR"
  "${DATASET_EXTRA_ARGS[@]}"
  --policy.type=pi05
  --policy.pretrained_path="$PI05_BASE_DIR"
  --output_dir="$RUN_DIR"
  --job_name=WJVLA_finetune
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --batch_size="$TRAIN_BATCH_SIZE"
  --dataset.image_transforms.enable="$IMAGE_TF_ENABLE"
  --dataset.image_transforms.max_num_transforms="$IMAGE_TF_MAX_NUM_TRANSFORMS"
  --dataset.image_transforms.random_order="$IMAGE_TF_RANDOM_ORDER"
  --policy.dtype="$DTYPE"
  --policy.compile_model=true
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING_ENABLE"
  --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER"
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY"
  --policy.action_expert_variant="$ACTION_EXPERT_VARIANT"
  --policy.load_rynnbrain_from_pretrained="$LOAD_RYNNBRAIN_FROM_PI05_BASE_DIR"
  --policy.enable_clip_running_status_condition="$CURRENT_PROGRESS_AS_INPUT_ENABLE"
  --policy.push_to_hub=false
  --policy.compile_mode=reduce-overhead
  --policy.optimizer_lr="$OPTIMIZER_LR"
  --policy.optimizer_weight_decay="$OPTIMIZER_WEIGHT_DECAY"
  --policy.optimizer_grad_clip_norm="$OPTIMIZER_GRAD_CLIP_NORM"
  --policy.action_space="$ACTION_SPACE"
  --policy.action_target_mode="$ACTION_TARGET_MODE"
  --policy.flow_source_mode="$FLOW_SOURCE_MODE"
  --policy.flow_source_state_num_frames="$FLOW_SOURCE_STATE_NUM_FRAMES"
  --policy.flow_source_blend_alpha="$FLOW_SOURCE_BLEND_ALPHA"
  --policy.train_rollout_debug_interval="$TRAIN_ROLLOUT_DEBUG_INTERVAL"
  --policy.input_image_debug_dump_dir="$INPUT_IMAGE_DEBUG_DUMP_DIR"
  --policy.input_image_debug_dump_interval="$INPUT_IMAGE_DEBUG_DUMP_INTERVAL"
  --policy.input_image_debug_max_samples_per_batch="$INPUT_IMAGE_DEBUG_MAX_SAMPLES_PER_BATCH"
  --policy.input_image_debug_max_cameras="$INPUT_IMAGE_DEBUG_MAX_CAMERAS"
  --policy.include_state_in_language_prompt="$LANGUAGE_INCLUDE_STATE_IN_PROMPT"
  --policy.enable_state_in_action_time_emb="$STATE_IN_ACTION_TIME_EMB_ENABLE"
  --policy.state_in_action_time_emb_debug_interval="$STATE_IN_ACTION_TIME_EMB_DEBUG_INTERVAL"
  --policy.enable_state_in_prefix_tokens="$STATE_IN_PREFIX_TOKENS_ENABLE"
  --policy.state_prefix_use_history="$STATE_PREFIX_USE_HISTORY"
  --policy.enable_task_aux_loss="$TASK_AUX_ENABLE"
  --policy.task_aux_num_classes="$TASK_AUX_NUM_CLASSES"
  --policy.task_aux_loss_weight="$TASK_AUX_LOSS_WEIGHT"
  --policy.enable_box_aux_loss="$BOX_AUX_ENABLE"
  --policy.box_aux_loss_weight="$BOX_AUX_LOSS_WEIGHT"
  --policy.enable_cross_center_aux_loss="$CROSS_CENTER_AUX_ENABLE"
  --policy.cross_center_aux_loss_weight="$CROSS_CENTER_AUX_LOSS_WEIGHT"
  --policy.enable_future_action_aux_loss="$FUTURE_ACTION_AUX_ENABLE"
  --policy.future_action_aux_dim="$FUTURE_ACTION_AUX_DIM"
  --policy.future_action_aux_loss_weight="$FUTURE_ACTION_AUX_LOSS_WEIGHT"
  --policy.ee_position_loss_weight="$EE_POSITION_LOSS_WEIGHT"
  --policy.ee_orientation_loss_weight="$EE_ORIENTATION_LOSS_WEIGHT"
  --policy.box_cross_debug_dump_dir="$BOX_CROSS_DEBUG_DUMP_DIR"
  --policy.box_cross_debug_dump_interval="$BOX_CROSS_DEBUG_DUMP_INTERVAL"
  --policy.box_cross_debug_camera_key="$BOX_CROSS_DEBUG_CAMERA_KEY"
  --policy.box_cross_debug_max_samples_per_batch="$BOX_CROSS_DEBUG_MAX_SAMPLES_PER_BATCH"
  --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS"
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS"
  --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR"
  --tolerance_s="$TOLERANCE_S"
  --use_language_mismatch_regularization="$LANG_MISMATCH_ENABLE"
  --language_mismatch_ratio="$LANG_MISMATCH_RATIO"
  --language_mismatch_mode="$LANG_MISMATCH_MODE"
  --language_mismatch_margin="$LANG_MISMATCH_MARGIN"
  --language_mismatch_weight="$LANG_MISMATCH_WEIGHT"
  --num_workers="$NUM_WORKERS"
  --step_profile="$STEP_PROFILE_ENABLE"
  --step_profile_start="$STEP_PROFILE_START"
  --step_profile_steps="$STEP_PROFILE_STEPS"
  --step_profile_cuda_sync="$STEP_PROFILE_CUDA_SYNC"
  "${POLICY_EXTRA_ARGS[@]}"
  --policy.compile_model=false
  --wandb.enable="$WANDB_ENABLE"
  --wandb.mode="$WANDB_MODE"
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT"
  --policy.normalization_mapping="$NORMALIZATION_MAPPING"
  "${CLI_ARGS[@]}"
)

"${TRAIN_CMD[@]}"
