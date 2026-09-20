#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import PI05AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


def _default_gripper_action_indices(action_dim: int) -> list[int]:
    d = int(action_dim)
    if d == 14:
        return [6, 13]
    if d == 7:
        return [6]
    return []


def _resolve_action_head_indices(configured_gripper_indices: list[int] | None, action_dim: int) -> tuple[list[int], list[int]]:
    if configured_gripper_indices:
        gripper_indices = [int(idx) for idx in configured_gripper_indices]
    else:
        gripper_indices = _default_gripper_action_indices(action_dim)

    seen: set[int] = set()
    for idx in gripper_indices:
        if idx < 0 or idx >= int(action_dim):
            raise ValueError(f"gripper_action_indices contains out-of-range index {idx} for action_dim={action_dim}")
        if idx in seen:
            raise ValueError(f"gripper_action_indices contains duplicate index {idx}")
        seen.add(idx)

    joint_indices = [idx for idx in range(int(action_dim)) if idx not in seen]
    return joint_indices, gripper_indices


BEHAVIOR_B1K_SEMANTIC_ACTION_HEAD_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("base", (0, 1, 2)),
    ("torso", (3, 4, 5, 6)),
    ("arm", (7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21)),
    ("gripper", (14, 22)),
)


def _resolve_behavior_b1k_semantic_action_head_groups(action_dim: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if int(action_dim) != 23:
        raise ValueError(
            "enable_behavior_b1k_semantic_action_heads=True requires Behavior 23D action layout, "
            f"got action_dim={action_dim}"
        )
    return BEHAVIOR_B1K_SEMANTIC_ACTION_HEAD_GROUPS


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    # RynnBrain Backbone Settings
    use_rynnbrain: bool = False
    rynnbrain_path: str = "/mnt/dataset/wj-dataset/vla_pretrain_model/RynnBrain-2B"
    # If false, keep RynnBrain weights from `rynnbrain_path` and only restore non-RynnBrain
    # modules from `pretrained_path`.
    load_rynnbrain_from_pretrained: bool = True
    enable_rynnbrain_lora: bool = False
    rynnbrain_lora_r: int = 64
    rynnbrain_lora_alpha: int = 128
    rynnbrain_lora_dropout: float = 0.01
    rynnbrain_lora_target_modules: list[str] | str = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Optional local tokenizer path used by RynnBrain-bridge mode to decode legacy PI05 tokens.
    paligemma_tokenizer_path: str | None = None
    enable_vggt_omega_prefix_token: bool = False
    vggt_omega_checkpoint_path: str | None = None
    vggt_omega_freeze: bool = True
    vggt_omega_camera_keys: list[str] | str = field(default_factory=list)

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    # Suffix attention behavior for action tokens:
    # - "causal": autoregressive within suffix
    # - "block": bidirectional within suffix
    suffix_attention_mask: str = "causal"
    # Deprecated and forced off for checkpoint compatibility.
    enable_learnable_layer_combination: bool = False
    layer_combination_init: str = "identity"
    layer_combination_source_from_last: list[int] = field(default_factory=list)
    enable_short_term_memory: bool = False
    short_term_memory_num_frames: list[int] | str = field(default_factory=lambda: [6])
    short_term_memory_stride: list[int] | str = field(default_factory=lambda: [1])
    short_term_memory_inference_num_frames: int = 6
    short_term_memory_include_proprio: bool = True
    short_term_memory_temporal_every_n_layers: int = 4
    short_term_memory_drop_history_last_n_layers: int = 2
    short_term_memory_debug_interval: int = 0
    short_term_memory_current_token_keep_ratio: float = 1.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    include_state_in_language_prompt: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    pytorch_compile_gemma_mode: str = ""  # Torch compile mode for PaliGemmaWithExpertModel.forward; empty disables it
    materialize_loss_metrics_every_n_steps: int = 1  # 1 keeps backward-compatible per-step metric sync
    rynnbrain_attn_implementation: str = ""  # Optional RynnBrain attention backend, e.g. sdpa/flash_attention_2
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections
    clip_running_status_key: str = "clip_running_status"
    enable_clip_running_status_condition: bool = False
    clip_running_status_loss_weight: float = 1.0
    clip_running_status_curve_dump_dir: str | None = None
    action_space: str = "joint"
    action_target_mode: str = "absolute"
    flow_source_mode: str = "gaussian"
    flow_source_state_num_frames: int = 1
    flow_source_blend_alpha: float = 0.3
    train_rollout_debug_interval: int = 0
    behavior_b1k_debug_metrics: bool = False
    enable_state_in_action_time_emb: bool = False
    state_in_action_time_emb_debug_interval: int = 100
    enable_state_in_prefix_tokens: bool = False
    enable_language_in_prefix_token: bool = False
    state_prefix_use_history: bool = False
    enable_task_aux_loss: bool = False
    task_aux_num_classes: int = 100
    task_aux_loss_weight: float = 1.0
    task_id_key: str = "task_id"
    task_id_map: dict[str, int] | None = None
    enable_box_aux_loss: bool = False
    box_aux_key: str = "observation.boxes"
    box_aux_loss_weight: float = 1.0
    enable_cross_center_aux_loss: bool = False
    cross_center_aux_key: str = "observation.cross_center"
    cross_center_aux_loss_weight: float = 1.0
    enable_future_action_aux_loss: bool = False
    future_action_aux_dim: int = 128
    future_action_aux_loss_weight: float = 0.0
    enable_group_consistency_loss: bool = False
    group_consistency_camera_keys: list[str] | str = field(default_factory=list)
    group_consistency_anchor_camera_key: str | None = None
    group_consistency_project_dim: int = 256
    group_consistency_loss_weight: float = 0.0
    box_cross_debug_dump_dir: str | None = None
    box_cross_debug_dump_interval: int = 0
    box_cross_debug_camera_key: str | None = None
    box_cross_debug_max_samples_per_batch: int = 0
    input_image_debug_dump_dir: str | None = None
    input_image_debug_dump_interval: int = 0
    input_image_debug_max_samples_per_batch: int = 1
    input_image_debug_max_cameras: int = 3
    ee_position_loss_weight: float = 0.0
    ee_orientation_loss_weight: float = 0.0
    pose_translation_loss_weight: float = 0.0
    pose_rotation_loss_weight: float = 0.0
    pose_loss_active_sides: list[str] | str = field(default_factory=lambda: ["left"])
    ee_kinematics_type: str = "piper"
    ee_joint_groups: list[list[int]] = field(default_factory=list)
    ee_joint_unit_scale: float = 0.001
    enable_split_action_heads: bool = False
    enable_behavior_b1k_semantic_action_heads: bool = False
    gripper_action_indices: list[int] | str = field(default_factory=list)
    action_loss_active_indices: list[int] | str = field(default_factory=list)
    joint_action_loss_weight: float = 1.0
    gripper_action_loss_weight: float = 1.0
    gripper_transition_oversample_weight: float = 1.0
    gripper_transition_threshold: float = 1e-3
    gripper_transition_window_before: int = 2
    gripper_transition_min_before: int = 0
    gripper_transition_window_after: int = 2
    gripper_transition_extra_aug_repeats: int = 0
    gripper_transition_debug_dump_dir: str | None = None
    gripper_transition_debug_max_samples: int = 24
    gripper_transition_debug_future_steps: int = 50

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_backbone_lr_scale: float = 1.0
    optimizer_head_lr_scale: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    @staticmethod
    def _normalize_int_list(value: object, field_name: str) -> list[int]:
        if isinstance(value, list):
            values = value
        elif isinstance(value, tuple):
            values = list(value)
        elif isinstance(value, str):
            spec = value.strip()
            if spec.startswith("[") and spec.endswith("]"):
                spec = spec[1:-1]
            values = [part.strip() for part in spec.split(",") if part.strip()]
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            try:
                values = [int(part) for part in values]
            except ValueError as exc:
                raise ValueError(f"{field_name} must contain integers only") from exc
        elif isinstance(value, int):
            values = [value]
        else:
            raise ValueError(f"Unsupported type for {field_name}: {type(value)!r}")

        if len(values) == 0:
            raise ValueError(f"{field_name} must not be empty")
        return [int(v) for v in values]

    @staticmethod
    def _normalize_optional_int_list(value: object, field_name: str) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return []
        if isinstance(value, str):
            spec = value.strip()
            if spec in {"", "[]"}:
                return []
        return PI05Config._normalize_int_list(value, field_name)

    @staticmethod
    def _normalize_optional_str_list(value: object, field_name: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            spec = value.strip()
            if spec in {"", "[]"}:
                return []
            if spec.startswith("[") and spec.endswith("]"):
                spec = spec[1:-1]
            items = [part.strip().strip("\"'") for part in spec.split(",") if part.strip().strip("\"'")]
            if not items:
                raise ValueError(f"{field_name} must not be empty")
            return items
        if isinstance(value, (list, tuple)):
            items = [str(part).strip() for part in value if str(part).strip()]
            if not items:
                return []
            return items
        raise ValueError(f"Unsupported type for {field_name}: {type(value)!r}")

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_1b", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.suffix_attention_mask not in {"causal", "block"}:
            raise ValueError("suffix_attention_mask must be one of {'causal', 'block'}")
        self.rynnbrain_lora_target_modules = self._normalize_optional_str_list(
            self.rynnbrain_lora_target_modules, "rynnbrain_lora_target_modules"
        )
        self.enable_learnable_layer_combination = False
        self.layer_combination_init = "identity"
        self.layer_combination_source_from_last = []
        self.short_term_memory_num_frames = self._normalize_int_list(
            self.short_term_memory_num_frames, "short_term_memory_num_frames"
        )
        self.short_term_memory_stride = self._normalize_int_list(
            self.short_term_memory_stride, "short_term_memory_stride"
        )
        if len(self.short_term_memory_num_frames) != len(self.short_term_memory_stride):
            raise ValueError("short_term_memory_num_frames and short_term_memory_stride must have the same length")
        if any(num_frames < 1 for num_frames in self.short_term_memory_num_frames):
            raise ValueError("short_term_memory_num_frames must be >= 1")
        for num_frames, stride in zip(self.short_term_memory_num_frames, self.short_term_memory_stride, strict=True):
            if stride < 0:
                raise ValueError("short_term_memory_stride must be >= 0")
            if num_frames > 1 and stride < 1:
                raise ValueError("short_term_memory_stride must be >= 1 when short_term_memory_num_frames > 1")
        if self.short_term_memory_inference_num_frames < 1:
            raise ValueError("short_term_memory_inference_num_frames must be >= 1")
        if self.short_term_memory_temporal_every_n_layers < 1:
            raise ValueError("short_term_memory_temporal_every_n_layers must be >= 1")
        if self.short_term_memory_drop_history_last_n_layers < 0:
            raise ValueError("short_term_memory_drop_history_last_n_layers must be >= 0")
        if self.short_term_memory_debug_interval < 0:
            raise ValueError("short_term_memory_debug_interval must be >= 0")
        if not (0.0 < float(self.short_term_memory_current_token_keep_ratio) <= 1.0):
            raise ValueError("short_term_memory_current_token_keep_ratio must be in (0, 1]")
        if self.enable_rynnbrain_lora:
            if not self.use_rynnbrain:
                raise ValueError("enable_rynnbrain_lora=True requires use_rynnbrain=True")
            if self.train_expert_only:
                raise ValueError("enable_rynnbrain_lora=True is incompatible with train_expert_only=True")
            if self.rynnbrain_lora_r <= 0:
                raise ValueError("rynnbrain_lora_r must be > 0")
            if self.rynnbrain_lora_alpha <= 0:
                raise ValueError("rynnbrain_lora_alpha must be > 0")
            if not (0.0 <= float(self.rynnbrain_lora_dropout) <= 1.0):
                raise ValueError("rynnbrain_lora_dropout must be in [0, 1]")
            if len(self.rynnbrain_lora_target_modules) == 0:
                raise ValueError(
                    "enable_rynnbrain_lora=True requires at least one rynnbrain_lora_target_modules entry"
                )

        if self.clip_running_status_loss_weight < 0:
            raise ValueError("clip_running_status_loss_weight must be non-negative")
        if self.action_space not in {"joint", "kitt_se3_pose", "revo2_eef_pose"}:
            raise ValueError("action_space must be one of {'joint', 'kitt_se3_pose', 'revo2_eef_pose'}")
        if self.action_target_mode not in {"absolute", "delta_from_state", "relative_pose"}:
            raise ValueError(
                "action_target_mode must be one of {'absolute', 'delta_from_state', 'relative_pose'}"
            )
        if self.flow_source_mode not in {"gaussian", "state_history", "blend"}:
            raise ValueError("flow_source_mode must be one of {'gaussian', 'state_history', 'blend'}")
        if self.flow_source_state_num_frames < 1:
            raise ValueError("flow_source_state_num_frames must be >= 1")
        if not (0.0 <= self.flow_source_blend_alpha <= 1.0):
            raise ValueError("flow_source_blend_alpha must be in [0, 1]")
        if self.train_rollout_debug_interval < 0:
            raise ValueError("train_rollout_debug_interval must be non-negative")
        if self.state_in_action_time_emb_debug_interval < 0:
            raise ValueError("state_in_action_time_emb_debug_interval must be non-negative")
        if self.enable_task_aux_loss and self.task_aux_num_classes <= 1:
            raise ValueError("task_aux_num_classes must be greater than 1")
        if self.task_aux_loss_weight < 0:
            raise ValueError("task_aux_loss_weight must be non-negative")
        if self.box_aux_loss_weight < 0:
            raise ValueError("box_aux_loss_weight must be non-negative")
        if self.cross_center_aux_loss_weight < 0:
            raise ValueError("cross_center_aux_loss_weight must be non-negative")
        if self.future_action_aux_dim < 1:
            raise ValueError("future_action_aux_dim must be >= 1")
        if self.future_action_aux_loss_weight < 0:
            raise ValueError("future_action_aux_loss_weight must be non-negative")
        if self.box_cross_debug_dump_interval < 0:
            raise ValueError("box_cross_debug_dump_interval must be non-negative")
        if self.box_cross_debug_max_samples_per_batch < 0:
            raise ValueError("box_cross_debug_max_samples_per_batch must be non-negative")
        if self.ee_position_loss_weight < 0:
            raise ValueError("ee_position_loss_weight must be non-negative")
        if self.ee_orientation_loss_weight < 0:
            raise ValueError("ee_orientation_loss_weight must be non-negative")
        if self.pose_translation_loss_weight < 0:
            raise ValueError("pose_translation_loss_weight must be non-negative")
        if self.pose_rotation_loss_weight < 0:
            raise ValueError("pose_rotation_loss_weight must be non-negative")
        if self.action_space in {"kitt_se3_pose", "revo2_eef_pose"}:
            if self.action_target_mode not in {"absolute", "relative_pose"}:
                raise ValueError(
                    f"{self.action_space} action space requires action_target_mode='absolute' or 'relative_pose'"
                )
            if self.flow_source_mode != "gaussian":
                raise ValueError(f"{self.action_space} action space requires flow_source_mode='gaussian'")
            if self.action_space == "kitt_se3_pose" and (self.ee_position_loss_weight > 0 or self.ee_orientation_loss_weight > 0):
                raise ValueError(
                    "kitt_se3_pose is already supervised in pose space; disable the legacy Piper EE losses"
                )
        elif self.action_target_mode == "relative_pose":
            raise ValueError("action_target_mode='relative_pose' requires action_space='kitt_se3_pose'")
        if self.ee_kinematics_type not in {"piper"}:
            raise ValueError("ee_kinematics_type must be one of {'piper'}")
        for group in self.ee_joint_groups:
            if len(group) == 0:
                raise ValueError("ee_joint_groups must not contain empty groups")
            if any(int(idx) < 0 for idx in group):
                raise ValueError("ee_joint_groups must contain non-negative indices only")
        if self.ee_joint_unit_scale <= 0:
            raise ValueError("ee_joint_unit_scale must be positive")
        if self.task_id_map is not None:
            for task_name, task_id in self.task_id_map.items():
                if int(task_id) < 0:
                    raise ValueError(f"task_id_map contains negative id for key {task_name!r}: {task_id}")
        self.gripper_action_indices = self._normalize_optional_int_list(
            self.gripper_action_indices, "gripper_action_indices"
        )
        self.action_loss_active_indices = self._normalize_optional_int_list(
            self.action_loss_active_indices, "action_loss_active_indices"
        )
        self.pose_loss_active_sides = self._normalize_optional_str_list(
            self.pose_loss_active_sides, "pose_loss_active_sides"
        )
        invalid_pose_sides = sorted(set(self.pose_loss_active_sides) - {"left", "right"})
        if invalid_pose_sides:
            raise ValueError(f"pose_loss_active_sides contains invalid sides: {invalid_pose_sides}")
        if (
            self.pose_translation_loss_weight > 0 or self.pose_rotation_loss_weight > 0
        ) and not self.pose_loss_active_sides:
            raise ValueError("pose semantic loss requires at least one pose_loss_active_sides entry")
        self.group_consistency_camera_keys = self._normalize_optional_str_list(
            self.group_consistency_camera_keys, "group_consistency_camera_keys"
        )
        self.vggt_omega_camera_keys = self._normalize_optional_str_list(
            self.vggt_omega_camera_keys, "vggt_omega_camera_keys"
        )
        if self.group_consistency_anchor_camera_key is not None:
            self.group_consistency_anchor_camera_key = self.group_consistency_anchor_camera_key.strip()
            if not self.group_consistency_anchor_camera_key:
                self.group_consistency_anchor_camera_key = None
        if self.enable_vggt_omega_prefix_token:
            if self.vggt_omega_checkpoint_path is None or not str(self.vggt_omega_checkpoint_path).strip():
                raise ValueError("enable_vggt_omega_prefix_token=True requires vggt_omega_checkpoint_path")
            if len(self.vggt_omega_camera_keys) == 0:
                raise ValueError("enable_vggt_omega_prefix_token=True requires at least one vggt_omega_camera_key")
        if self.joint_action_loss_weight < 0:
            raise ValueError("joint_action_loss_weight must be non-negative")
        if self.gripper_action_loss_weight < 0:
            raise ValueError("gripper_action_loss_weight must be non-negative")
        if self.group_consistency_project_dim < 1:
            raise ValueError("group_consistency_project_dim must be >= 1")
        if self.group_consistency_loss_weight < 0:
            raise ValueError("group_consistency_loss_weight must be non-negative")
        if self.enable_group_consistency_loss and len(self.group_consistency_camera_keys) < 2:
            raise ValueError("enable_group_consistency_loss=True requires at least 2 group_consistency_camera_keys")
        if (
            self.group_consistency_anchor_camera_key is not None
            and self.group_consistency_anchor_camera_key not in self.group_consistency_camera_keys
        ):
            raise ValueError("group_consistency_anchor_camera_key must be included in group_consistency_camera_keys")
        if self.gripper_transition_oversample_weight < 1.0:
            raise ValueError("gripper_transition_oversample_weight must be >= 1.0")
        if self.gripper_transition_threshold < 0:
            raise ValueError("gripper_transition_threshold must be non-negative")
        if self.gripper_transition_window_before < 0:
            raise ValueError("gripper_transition_window_before must be non-negative")
        if self.gripper_transition_min_before < 0:
            raise ValueError("gripper_transition_min_before must be non-negative")
        if self.gripper_transition_min_before > self.gripper_transition_window_before:
            raise ValueError("gripper_transition_min_before must be <= gripper_transition_window_before")
        if self.gripper_transition_window_after < 0:
            raise ValueError("gripper_transition_window_after must be non-negative")
        if self.gripper_transition_extra_aug_repeats < 0:
            raise ValueError("gripper_transition_extra_aug_repeats must be non-negative")
        if self.gripper_transition_debug_max_samples < 0:
            raise ValueError("gripper_transition_debug_max_samples must be non-negative")
        if self.gripper_transition_debug_future_steps < 1:
            raise ValueError("gripper_transition_debug_future_steps must be >= 1")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature
        if self.enable_split_action_heads:
            action_dim = int(self.output_features["action"].shape[0])
            joint_indices, gripper_indices = _resolve_action_head_indices(self.gripper_action_indices, action_dim)
            if not gripper_indices:
                raise ValueError(
                    "enable_split_action_heads=True requires gripper_action_indices or a known default layout "
                    f"(got action_dim={action_dim})"
                )
            if not joint_indices:
                raise ValueError("enable_split_action_heads=True requires at least one non-gripper action dim")
        action_dim = int(self.output_features["action"].shape[0])
        if self.action_loss_active_indices:
            invalid_active_indices = [
                index
                for index in self.action_loss_active_indices
                if index < 0 or index >= action_dim
            ]
            if invalid_active_indices:
                raise ValueError(
                    "action_loss_active_indices contains out-of-range indices "
                    f"{invalid_active_indices} for action_dim={action_dim}"
                )
        if self.enable_behavior_b1k_semantic_action_heads:
            _resolve_behavior_b1k_semantic_action_head_groups(int(self.output_features["action"].shape[0]))

        if self.enable_box_aux_loss and self.box_aux_key not in self.input_features:
            raise ValueError(f"box_aux_key {self.box_aux_key!r} is required when enable_box_aux_loss=True")

        if self.enable_cross_center_aux_loss and self.cross_center_aux_key not in self.input_features:
            raise ValueError(
                f"cross_center_aux_key {self.cross_center_aux_key!r} is required when enable_cross_center_aux_loss=True"
            )
        if self.enable_group_consistency_loss:
            available_image_keys = list(self.image_features)
            image_key_lookup = {key: key for key in available_image_keys}
            for key in available_image_keys:
                image_key_lookup.setdefault(key.split(".")[-1], key)
            resolved_group_keys = [image_key_lookup.get(camera_key, camera_key) for camera_key in self.group_consistency_camera_keys]
            missing_consistency_cameras = [
                camera_key for camera_key in resolved_group_keys if camera_key not in self.image_features
            ]
            if missing_consistency_cameras:
                raise ValueError(
                    "group_consistency_camera_keys must be visual feature keys present in image_features. "
                    f"Missing: {missing_consistency_cameras}"
                )
            self.group_consistency_camera_keys = resolved_group_keys
            if self.group_consistency_anchor_camera_key is not None:
                resolved_anchor = image_key_lookup.get(
                    self.group_consistency_anchor_camera_key, self.group_consistency_anchor_camera_key
                )
                if resolved_anchor not in self.group_consistency_camera_keys:
                    raise ValueError(
                        "group_consistency_anchor_camera_key must be included in group_consistency_camera_keys"
                    )
                self.group_consistency_anchor_camera_key = resolved_anchor
        if self.enable_vggt_omega_prefix_token:
            available_image_keys = list(self.image_features)
            image_key_lookup = {key: key for key in available_image_keys}
            for key in available_image_keys:
                image_key_lookup.setdefault(key.split(".")[-1], key)
            resolved_vggt_keys = [image_key_lookup.get(camera_key, camera_key) for camera_key in self.vggt_omega_camera_keys]
            missing_vggt_cameras = [camera_key for camera_key in resolved_vggt_keys if camera_key not in self.image_features]
            if missing_vggt_cameras:
                raise ValueError(
                    "vggt_omega_camera_keys must be visual feature keys present in image_features. "
                    f"Missing: {missing_vggt_cameras}"
                )
            self.vggt_omega_camera_keys = resolved_vggt_keys

    def get_optimizer_preset(self) -> PI05AdamWConfig:
        return PI05AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            backbone_lr_scale=self.optimizer_backbone_lr_scale,
            head_lr_scale=self.optimizer_head_lr_scale,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        if not self.enable_short_term_memory:
            return None
        return max(self.observation_delta_indices_candidates, key=len)

    @property
    def observation_delta_indices_candidates(self) -> list[list[int]] | None:
        if not self.enable_short_term_memory:
            return None
        candidates = []
        for num_frames, stride in zip(self.short_term_memory_num_frames, self.short_term_memory_stride, strict=True):
            if num_frames == 1:
                candidates.append([0])
                continue
            candidates.append(list(range(-(num_frames - 1) * stride, 1, stride)))
        return candidates

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
