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

import builtins
import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[7]
_TRANSFORMERS_PATH = str(_PROJECT_ROOT / "src" / "open_source" / "transformers_4.57.1")
if os.path.exists(_TRANSFORMERS_PATH) and _TRANSFORMERS_PATH not in sys.path:
    sys.path.insert(0, _TRANSFORMERS_PATH)
_VGGT_OMEGA_PATH = str(_PROJECT_ROOT / "src" / "open_source" / "vggt-omega")
if os.path.exists(_VGGT_OMEGA_PATH) and _VGGT_OMEGA_PATH not in sys.path:
    sys.path.insert(0, _VGGT_OMEGA_PATH)

from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.pi05.configuration_pi05 import (
    BEHAVIOR_B1K_SEMANTIC_ACTION_HEAD_GROUPS,
    DEFAULT_IMAGE_SIZE,
    PI05Config,
    _resolve_action_head_indices,
    _resolve_behavior_b1k_semantic_action_head_groups,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)

CLIP_RUNNING_STATUS = "clip_running_status"
BOX_AUX_LOSS = "box_aux_loss"
CROSS_CENTER_AUX_LOSS = "cross_center_aux_loss"
FUTURE_ACTION_AUX_LOSS = "future_action_aux_loss"
GROUP_CONSISTENCY_AUX_LOSS = "group_consistency_aux_loss"
BOX_AUX_VALID_COUNT = "box_aux_valid_count"
BOX_AUX_VALID_RATIO = "box_aux_valid_ratio"
CROSS_CENTER_AUX_VALID_COUNT = "cross_center_aux_valid_count"
CROSS_CENTER_AUX_VALID_RATIO = "cross_center_aux_valid_ratio"


def _default_ee_joint_groups(action_dim: int) -> list[list[int]]:
    if action_dim == 7:
        return [[0, 1, 2, 3, 4, 5]]
    if action_dim == 14:
        return [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]]
    return []


def _to_tensor_stat(
    stats: dict[str, dict[str, Tensor | list[float] | tuple[float, ...]]] | None,
    key: str,
    stat_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    if not stats or key not in stats or stat_name not in stats[key]:
        return None
    value = stats[key][stat_name]
    return torch.as_tensor(value, device=device, dtype=dtype)


def _piper_link_transform_torch(alpha: Tensor, a: Tensor, theta: Tensor, d: Tensor) -> Tensor:
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)

    shape = theta.shape + (4, 4)
    transform = torch.zeros(shape, device=theta.device, dtype=theta.dtype)
    transform[..., 0, 0] = cos_theta
    transform[..., 0, 1] = -sin_theta
    transform[..., 0, 3] = a
    transform[..., 1, 0] = sin_theta * cos_alpha
    transform[..., 1, 1] = cos_theta * cos_alpha
    transform[..., 1, 2] = -sin_alpha
    transform[..., 1, 3] = -sin_alpha * d
    transform[..., 2, 0] = sin_theta * sin_alpha
    transform[..., 2, 1] = cos_theta * sin_alpha
    transform[..., 2, 2] = cos_alpha
    transform[..., 2, 3] = cos_alpha * d
    transform[..., 3, 3] = 1.0
    return transform


def _piper_forward_kinematics_torch(joint_values: Tensor, joint_unit_scale: float) -> tuple[Tensor, Tensor]:
    """Differentiable Piper FK using the DH parameters from `tools/piper_sdk`."""
    if joint_values.shape[-1] != 6:
        raise ValueError(f"Piper FK expects 6 joints, got shape {tuple(joint_values.shape)}")

    dtype = joint_values.dtype
    device = joint_values.device
    pi = torch.tensor(math.pi, device=device, dtype=dtype)
    deg_to_rad = pi / 180.0
    joint_rad = joint_values * float(joint_unit_scale) * deg_to_rad

    a = torch.tensor([0.0, 0.0, 285.03, -21.98, 0.0, 0.0], device=device, dtype=dtype)
    alpha = torch.tensor([0.0, -pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0], device=device, dtype=dtype)
    theta_offset = torch.tensor(
        [0.0, -172.22 * deg_to_rad, -102.78 * deg_to_rad, 0.0, 0.0, 0.0],
        device=device,
        dtype=dtype,
    )
    d = torch.tensor([123.0, 0.0, 0.0, 250.75, 0.0, 91.0], device=device, dtype=dtype)

    transform = torch.eye(4, device=device, dtype=dtype).expand(joint_values.shape[:-1] + (4, 4)).clone()
    for joint_idx in range(6):
        link_transform = _piper_link_transform_torch(
            alpha[joint_idx],
            a[joint_idx],
            joint_rad[..., joint_idx] + theta_offset[joint_idx],
            d[joint_idx],
        )
        transform = torch.matmul(transform, link_transform)
    return transform[..., :3, 3], transform[..., :3, :3]


def _rotation_geodesic_distance(pred_rot: Tensor, target_rot: Tensor) -> Tensor:
    rel_rot = torch.matmul(pred_rot, target_rot.transpose(-1, -2))
    trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta)


def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """Project two predicted columns to a differentiable right-handed SO(3)."""

    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"rotation_6d must end in 6 dimensions, got {tuple(rotation_6d.shape)}")
    eps = torch.finfo(rotation_6d.dtype).eps * 16.0
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback_first = torch.zeros_like(first)
    fallback_first[..., 0] = 1.0
    basis_x = torch.where(first_norm > eps, first / first_norm.clamp_min(eps), fallback_first)
    second_orthogonal = second - (basis_x * second).sum(dim=-1, keepdim=True) * basis_x
    second_norm = torch.linalg.vector_norm(second_orthogonal, dim=-1, keepdim=True)
    fallback_index = torch.argmin(torch.abs(basis_x), dim=-1)
    fallback_axis = F.one_hot(fallback_index, num_classes=3).to(
        device=rotation_6d.device, dtype=rotation_6d.dtype
    )
    fallback_orthogonal = fallback_axis - (
        fallback_axis * basis_x
    ).sum(dim=-1, keepdim=True) * basis_x
    fallback_orthogonal = fallback_orthogonal / torch.linalg.vector_norm(
        fallback_orthogonal, dim=-1, keepdim=True
    ).clamp_min(eps)
    basis_y = torch.where(
        second_norm > eps,
        second_orthogonal / second_norm.clamp_min(eps),
        fallback_orthogonal,
    )
    basis_z = torch.linalg.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-1)


def _rotation_geodesic_angle(pred_rot: Tensor, target_rot: Tensor) -> Tensor:
    """SO(3) geodesic angle in radians with an exact zero at identity."""

    relative = target_rot.transpose(-1, -2) @ pred_rot
    cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def create_zero_anchored_temporal_embedding(
    num_frames: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a fixed temporal embedding whose current frame contribution is zero."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    frame_positions = torch.arange(float(-(num_frames - 1)), 1.0, device=device, dtype=torch.float32)
    time_emb = create_sinusoidal_pos_embedding(
        frame_positions,
        dimension,
        min_period=1.0,
        max_period=max(float(num_frames), 2.0),
        device=device,
    )
    time_emb = time_emb - time_emb[-1:]
    return time_emb.to(dtype=dtype)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks, attention_mode: str = "causal"):  # see openpi `make_att_2d_masks` (adapted)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if attention_mode not in ("causal", "block"):
        raise ValueError(f"attention_mode must be 'causal' or 'block', got '{attention_mode}'")
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    if attention_mode == "block":
        # Full bidirectional attention among valid tokens.
        return pad_2d_masks

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


def _init_layer_mix_weights(
    *,
    num_expert_layers: int,
    num_base_layers: int,
    stride: int,
    init_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if init_mode not in ("identity", "stride_uniform", "global_uniform"):
        raise ValueError(
            f"Invalid layer_combination_init: {init_mode}. "
            "Use one of: 'identity', 'stride_uniform', 'global_uniform'."
        )

    k_init = torch.zeros((num_expert_layers, num_base_layers), dtype=torch.float32)
    v_init = torch.zeros((num_expert_layers, num_base_layers), dtype=torch.float32)
    if init_mode == "identity":
        for expert_idx in range(num_expert_layers):
            base_idx = min(expert_idx * stride, num_base_layers - 1)
            k_init[expert_idx, base_idx] = 1.0
            v_init[expert_idx, base_idx] = 1.0
    elif init_mode == "stride_uniform":
        for expert_idx in range(num_expert_layers):
            start = min(expert_idx * stride, num_base_layers - 1)
            end = min(start + stride, num_base_layers)
            weight = 1.0 / float(max(1, end - start))
            k_init[expert_idx, start:end] = weight
            v_init[expert_idx, start:end] = weight
    else:
        weight = 1.0 / float(num_base_layers)
        k_init.fill_(weight)
        v_init.fill_(weight)
    return k_init, v_init


def _normalize_layer_mix_weights(mix_weights: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-6, dtype=mix_weights.dtype, device=mix_weights.device)
    return mix_weights / (mix_weights.sum(dim=-1, keepdim=True) + eps)


def _mix_layer_stack(
    layer_stack: torch.Tensor,
    mix_weights: nn.Parameter,
    layer_idx: int,
    mix_bias: nn.Parameter | None = None,
) -> torch.Tensor:
    weights = _normalize_layer_mix_weights(mix_weights[layer_idx].float()).to(dtype=layer_stack.dtype)
    mixed = torch.einsum("l,lbhtd->bhtd", weights, layer_stack)
    if mix_bias is not None:
        mixed = mixed + mix_bias[layer_idx].to(dtype=mixed.dtype, device=mixed.device).view(1, 1, 1, 1)
    return mixed


def _apply_rotary_query_key(rotary_model, query_state: torch.Tensor, key_state: torch.Tensor, position_ids: torch.Tensor):
    dummy_tensor = torch.zeros(
        query_state.shape[0],
        query_state.shape[2],
        query_state.shape[-1],
        device=query_state.device,
        dtype=query_state.dtype,
    )
    cos, sin = rotary_model.rotary_emb(dummy_tensor, position_ids)
    return modeling_gemma.apply_rotary_pos_emb(query_state, key_state, cos, sin, unsqueeze_dim=1)


def _apply_rotary_key_only(rotary_model, key_state: torch.Tensor, position_ids: torch.Tensor):
    _, rotated_key = _apply_rotary_query_key(rotary_model, key_state, key_state, position_ids)
    return rotated_key


def _run_suffix_expert_layer(
    layer,
    rotary_model,
    suffix_hidden: torch.Tensor,
    suffix_attention_mask: torch.Tensor,
    suffix_position_ids: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    adarms_cond: torch.Tensor | None,
) -> torch.Tensor:
    hidden_states, gate = layer.input_layernorm(suffix_hidden, cond=adarms_cond)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_state, key_state = _apply_rotary_query_key(rotary_model, query_state, key_state, suffix_position_ids)

    full_key_state = torch.cat([prefix_key.to(dtype=key_state.dtype), key_state], dim=2)
    full_value_state = torch.cat([prefix_value.to(dtype=value_state.dtype), value_state], dim=2)
    att_output, _ = modeling_gemma.eager_attention_forward(
        layer.self_attn,
        query_state,
        full_key_state,
        full_value_state,
        suffix_attention_mask,
        layer.self_attn.scaling,
    )
    att_output = att_output.reshape(suffix_hidden.shape[0], suffix_hidden.shape[1], -1)
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
    out_emb = layer.self_attn.o_proj(att_output)
    out_emb = modeling_gemma._gated_residual(suffix_hidden, out_emb, gate)  # noqa: SLF001
    after_first_residual = out_emb.clone()
    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        out_emb = out_emb.to(dtype=torch.bfloat16)
    out_emb = layer.mlp(out_emb)
    return modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_1b":
        return GemmaConfig(
            width=1536,
            depth=18,
            mlp_dim=8192,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


DEFAULT_VGGT_OMEGA_IMAGE_RESOLUTION = 224


class VGGTOmegaPrefixAdapter(nn.Module):
    """Extract one VGGT-Omega spatial token per selected camera."""

    def __init__(
        self,
        checkpoint_path: str,
        output_dim: int,
        *,
        freeze_backbone: bool = True,
        image_resolution: int = DEFAULT_VGGT_OMEGA_IMAGE_RESOLUTION,
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint}")

        from vggt_omega.models import VGGTOmega

        self.freeze_backbone = bool(freeze_backbone)
        self.image_resolution = int(image_resolution)
        self.backbone = VGGTOmega(
            enable_camera=False,
            enable_depth=False,
            enable_alignment=True,
        )
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=False)
        self.projector = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.token_type_embedding = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.token_type_embedding, std=0.02)

        if self.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_camera_images(self, camera_images: Tensor) -> Tensor:
        if camera_images.ndim == 4:
            camera_images = camera_images[:, None, :, :, :]
        elif camera_images.ndim != 5:
            raise ValueError(f"Unsupported VGGT camera image ndim: {camera_images.ndim}")

        batch_size, num_frames, channels, _, _ = camera_images.shape
        if channels != 3:
            raise ValueError(f"VGGT-Omega expects 3-channel images, got shape {tuple(camera_images.shape)}")

        camera_images = camera_images.to(dtype=torch.float32)
        if torch.any(camera_images < 0):
            camera_images = (camera_images + 1.0) * 0.5
        camera_images = camera_images.clamp_(0.0, 1.0)

        if camera_images.shape[-2:] != (self.image_resolution, self.image_resolution):
            flat_images = camera_images.reshape(batch_size * num_frames, channels, *camera_images.shape[-2:])
            flat_images = resize_with_pad_torch(flat_images, self.image_resolution, self.image_resolution)
            camera_images = flat_images.reshape(batch_size, num_frames, channels, self.image_resolution, self.image_resolution)

        return camera_images.contiguous()

    def forward(self, camera_images_list: list[Tensor]) -> Tensor:
        if len(camera_images_list) == 0:
            raise ValueError("VGGT-Omega prefix adapter requires at least one camera tensor")

        prepared_images_list = [self._prepare_camera_images(camera_images) for camera_images in camera_images_list]
        batch_size = prepared_images_list[0].shape[0]
        for prepared_images in prepared_images_list[1:]:
            if prepared_images.shape[0] != batch_size:
                raise ValueError(
                    "All VGGT-Omega camera tensors must share the same batch size, got "
                    f"{[tuple(images.shape) for images in prepared_images_list]}"
                )

        with torch.set_grad_enabled(self.training and not self.freeze_backbone):
            merged_images = torch.cat(prepared_images_list, dim=0)
            predictions = self.backbone(merged_images)
            per_camera_tokens = predictions["text_alignment_token"].to(dtype=torch.float32)

        per_camera_tokens = per_camera_tokens.reshape(len(prepared_images_list), batch_size, -1).permute(1, 0, 2)
        per_camera_tokens = self.projector(per_camera_tokens)
        per_camera_tokens = per_camera_tokens + self.token_type_embedding[None, None, :]
        return per_camera_tokens


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        enable_learnable_layer_combination: bool = False,
        layer_combination_init: str = "identity",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.enable_learnable_layer_combination = False
        self.layer_combination_init = "identity"
        self.num_base_layers = vlm_config.depth
        self.num_expert_layers = action_expert_config.depth
        self.action_k_mix_weights = None
        self.action_v_mix_weights = None
        self.action_k_mix_bias = None
        self.action_v_mix_bias = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def _build_prefix_kv_stack(
        self,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_outputs = self.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True,
            adarms_cond=None,
        )
        prefix_hidden_stack = prefix_outputs.hidden_states[:-1]
        prefix_key_stack = []
        prefix_value_stack = []
        for base_idx, hidden_states in enumerate(prefix_hidden_stack):
            layer = self.paligemma.language_model.layers[base_idx]
            hidden_states, _ = layer.input_layernorm(hidden_states, cond=None)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_state = _apply_rotary_key_only(self.paligemma.language_model, key_state, prefix_position_ids)
            prefix_key_stack.append(key_state)
            prefix_value_stack.append(value_state)
        return (
            prefix_outputs.last_hidden_state,
            torch.stack(prefix_key_stack, dim=0),
            torch.stack(prefix_value_stack, dim=0),
        )

    def _forward_suffix_with_mixed_prefix(
        self,
        suffix_embs: torch.Tensor,
        suffix_attention_mask: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        prefix_key_stack: torch.Tensor,
        prefix_value_stack: torch.Tensor,
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        suffix_hidden = suffix_embs
        for expert_idx, layer in enumerate(self.gemma_expert.model.layers[: self.num_expert_layers]):
            mixed_prefix_key = _mix_layer_stack(
                prefix_key_stack, self.action_k_mix_weights, expert_idx, self.action_k_mix_bias
            )
            mixed_prefix_value = _mix_layer_stack(
                prefix_value_stack, self.action_v_mix_weights, expert_idx, self.action_v_mix_bias
            )
            suffix_hidden = _run_suffix_expert_layer(
                layer,
                self.gemma_expert.model,
                suffix_hidden,
                suffix_attention_mask,
                suffix_position_ids,
                mixed_prefix_key,
                mixed_prefix_value,
                adarms_cond,
            )
        suffix_hidden, _ = self.gemma_expert.model.norm(suffix_hidden, cond=adarms_cond)
        return suffix_hidden

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        prefix_layer_states: torch.Tensor | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            return [prefix_output, None], prefix_past_key_values
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            return [None, suffix_output], None
        models = [self.paligemma.language_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        # Check if gradient checkpointing is enabled for any of the models
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        # Process all layers with gradient checkpointing if enabled
        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )
            else:
                inputs_embeds = compute_layer_complete(
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )

        def compute_final_norms(inputs_embeds, adarms_cond):
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        # Apply gradient checkpointing to final norm if enabled
        if use_gradient_checkpointing:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                inputs_embeds,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

        prefix_output = outputs_embeds[0]
        suffix_output = outputs_embeds[1]
        prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class RynnBrainWithExpertModel(nn.Module):
    def __init__(
        self,
        action_expert_config,
        rynnbrain_path="/mnt/dataset/wj-dataset/vla_pretrain_model/RynnBrain-2B",
        precision="bfloat16",
        train_expert_only=False,
        enable_rynnbrain_lora: bool = False,
        rynnbrain_lora_r: int = 16,
        rynnbrain_lora_alpha: int = 32,
        rynnbrain_lora_dropout: float = 0.05,
        rynnbrain_lora_target_modules: list[str] | None = None,
        paligemma_tokenizer_path: str | None = None,
        enable_learnable_layer_combination: bool = False,
        layer_combination_init: str = "identity",
        layer_combination_source_from_last: list[int] | None = None,
        enable_short_term_memory: bool = False,
        short_term_memory_include_proprio: bool = True,
        short_term_memory_state_dim: int = 32,
        short_term_memory_temporal_every_n_layers: int = 4,
        short_term_memory_drop_history_last_n_layers: int = 2,
        short_term_memory_current_token_keep_ratio: float = 1.0,
    ):
        super().__init__()
        self.train_expert_only = train_expert_only
        self.enable_learnable_layer_combination = False
        self.layer_combination_init = "identity"
        self.layer_combination_source_from_last = []
        self.enable_short_term_memory = enable_short_term_memory
        self.short_term_memory_include_proprio = short_term_memory_include_proprio
        self._mixed_prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        self.rynnbrain = AutoModelForImageTextToText.from_pretrained(
            rynnbrain_path, torch_dtype=dtype, trust_remote_code=True
        )
        if enable_rynnbrain_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "RynnBrain LoRA requires the `peft` package. Install `peft` in the training environment "
                    "before setting enable_rynnbrain_lora=True."
                ) from exc

            lora_config = LoraConfig(
                r=int(rynnbrain_lora_r),
                lora_alpha=int(rynnbrain_lora_alpha),
                target_modules=list(rynnbrain_lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=float(rynnbrain_lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.rynnbrain = get_peft_model(self.rynnbrain, lora_config)
            if hasattr(self.rynnbrain, "enable_input_require_grads"):
                self.rynnbrain.enable_input_require_grads()
            if hasattr(self.rynnbrain, "print_trainable_parameters"):
                self.rynnbrain.print_trainable_parameters()
        self.rynnbrain_processor = AutoProcessor.from_pretrained(rynnbrain_path, trust_remote_code=True)
        setattr(self.rynnbrain_processor, "compress_video_tokens", self.enable_short_term_memory)
        setattr(self.rynnbrain.config, "mem_compress_video_tokens", self.enable_short_term_memory)
        if hasattr(self.rynnbrain.config, "vision_config"):
            setattr(self.rynnbrain.config.vision_config, "mem_enable_short_term_memory", self.enable_short_term_memory)
            setattr(
                self.rynnbrain.config.vision_config,
                "mem_temporal_every_n_layers",
                int(short_term_memory_temporal_every_n_layers),
            )
            setattr(
                self.rynnbrain.config.vision_config,
                "mem_drop_history_last_n_layers",
                int(short_term_memory_drop_history_last_n_layers),
            )
            setattr(
                self.rynnbrain.config.vision_config,
                "mem_debug_interval",
                int(getattr(self.rynnbrain.config.vision_config, "mem_debug_interval", 0)),
            )
            setattr(
                self.rynnbrain.config.vision_config,
                "mem_current_token_keep_ratio",
                float(short_term_memory_current_token_keep_ratio),
            )
        tokenizer_src = paligemma_tokenizer_path or os.environ.get("PALIGEMMA_TOKENIZER_PATH")
        if tokenizer_src is None:
            tokenizer_src = "google/paligemma-3b-pt-224"

        try:
            # Never require online access for inference/training in offline clusters.
            self.paligemma_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_src, local_files_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load PaliGemma tokenizer locally. "
                "Set `paligemma_tokenizer_path` in config or env `PALIGEMMA_TOKENIZER_PATH` "
                "to a local tokenizer directory."
            ) from exc
        
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
        
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype=dtype,
            use_adarms=True,
            adarms_cond_dim=action_expert_config.width,
        )
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert = self.gemma_expert.to(dtype=dtype)
        self.gemma_expert.model.embed_tokens = None
        
        hidden_size = getattr(self.rynnbrain.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.rynnbrain.config, "text_config"):
            hidden_size = getattr(self.rynnbrain.config.text_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = 2048  # Default for RynnBrain-2B
        base_num_layers = getattr(self.rynnbrain.config, "num_hidden_layers", None)
        if base_num_layers is None and hasattr(self.rynnbrain.config, "text_config"):
            base_num_layers = getattr(self.rynnbrain.config.text_config, "num_hidden_layers", None)
        if base_num_layers is None:
            base_num_layers = 28
            
        self.align_proj = nn.Linear(hidden_size, action_expert_config.width, dtype=dtype)
        self.total_num_base_layers = int(base_num_layers)
        self.num_base_layers = self.total_num_base_layers
        self.num_expert_layers = action_expert_config.depth
        self.action_k_mix_weights = None
        self.action_v_mix_weights = None
        self.action_k_mix_bias = None
        self.action_v_mix_bias = None
        
        if self.train_expert_only:
            self.rynnbrain.eval()
            for param in self.rynnbrain.parameters():
                param.requires_grad = False

    def get_prefix_embs(self, rynnbrain_inputs):
        # Full finetune path: allow gradients through RynnBrain unless train_expert_only is enabled.
        with torch.set_grad_enabled(not self.train_expert_only):
            outputs = self.rynnbrain(
                **rynnbrain_inputs,
                output_hidden_states=False,
                output_last_hidden_state_only=True,
                use_cache=False,
                logits_to_keep=None,
            )
            prefix_hidden = outputs.pre_norm_last_hidden_state
        return self.align_proj(prefix_hidden)

    def get_text_prefix_embs(self, raw_texts: list[str], device: torch.device | str):
        if len(raw_texts) == 0:
            raise ValueError("raw_texts must not be empty")
        tokenizer = getattr(self.rynnbrain_processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("RynnBrain processor does not expose a tokenizer")
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": text}]}]
            for text in raw_texts
        ]
        prompts = self.rynnbrain_processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenizer_kwargs = {
            "padding": True,
            "return_tensors": "pt",
            "return_token_type_ids": False,
        }
        single_prompt = prompts[0] if isinstance(prompts, list) and prompts else prompts
        bos_token = getattr(tokenizer, "bos_token", None)
        if bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(bos_token):
            tokenizer_kwargs["add_special_tokens"] = False
        text_inputs = tokenizer(prompts, **tokenizer_kwargs)
        model_inputs = {
            "input_ids": text_inputs["input_ids"].to(device=device),
            "attention_mask": text_inputs["attention_mask"].to(device=device),
        }
        with torch.set_grad_enabled(not self.train_expert_only):
            outputs = self.rynnbrain(
                **model_inputs,
                output_hidden_states=False,
                output_last_hidden_state_only=True,
                use_cache=False,
                logits_to_keep=None,
            )
            prefix_hidden = outputs.pre_norm_last_hidden_state
        prefix_hidden = self.align_proj(
            prefix_hidden.to(device=self.align_proj.weight.device, dtype=self.align_proj.weight.dtype)
        )
        return prefix_hidden, model_inputs["attention_mask"].to(device=prefix_hidden.device, dtype=torch.bool)

    def embed_language_tokens(self, tokens: torch.Tensor):
        with torch.set_grad_enabled(not self.train_expert_only):
            return self.rynnbrain.get_input_embeddings()(tokens)

    def _build_prefix_kv_stack(
        self,
        prefix_layer_states: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        expert_idx: int,
        expert_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.gemma_expert.model.layers[expert_idx]
        prefix_key_stack = []
        prefix_value_stack = []
        for hidden_states in prefix_layer_states:
            hidden_states = hidden_states.to(dtype=expert_dtype)
            hidden_states, _ = layer.input_layernorm(hidden_states, cond=None)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_state = _apply_rotary_key_only(self.gemma_expert.model, key_state, prefix_position_ids)
            prefix_key_stack.append(key_state)
            prefix_value_stack.append(value_state)
        return torch.stack(prefix_key_stack, dim=0), torch.stack(prefix_value_stack, dim=0)

    def clear_mixed_prefix_kv_cache(self) -> None:
        self._mixed_prefix_kv_cache = None

    def prepare_mixed_prefix_kv_cache(
        self,
        prefix_layer_states: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        expert_dtype: torch.dtype,
    ) -> None:
        self._mixed_prefix_kv_cache = None

    def _forward_suffix_with_mixed_prefix(
        self,
        prefix_layer_states: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        adarms_cond: torch.Tensor | None,
        expert_dtype: torch.dtype,
    ) -> torch.Tensor:
        suffix_hidden = suffix_embs.to(dtype=expert_dtype)
        mixed_prefix_kv_cache = self._mixed_prefix_kv_cache
        for expert_idx, layer in enumerate(self.gemma_expert.model.layers[: self.num_expert_layers]):
            if mixed_prefix_kv_cache is not None:
                mixed_prefix_key, mixed_prefix_value = mixed_prefix_kv_cache[expert_idx]
            else:
                prefix_key_stack, prefix_value_stack = self._build_prefix_kv_stack(
                    prefix_layer_states, prefix_position_ids, expert_idx, expert_dtype
                )
                mixed_prefix_key = _mix_layer_stack(
                    prefix_key_stack, self.action_k_mix_weights, expert_idx, self.action_k_mix_bias
                )
                mixed_prefix_value = _mix_layer_stack(
                    prefix_value_stack, self.action_v_mix_weights, expert_idx, self.action_v_mix_bias
                )
            suffix_hidden = _run_suffix_expert_layer(
                layer,
                self.gemma_expert.model,
                suffix_hidden,
                suffix_attention_mask,
                suffix_position_ids,
                mixed_prefix_key,
                mixed_prefix_value,
                adarms_cond,
            )
        suffix_hidden, _ = self.gemma_expert.model.norm(suffix_hidden, cond=adarms_cond)
        return suffix_hidden
        
    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
        prefix_layer_states: torch.Tensor | None = None,
    ):
        prefix_embs, suffix_embs = inputs_embeds[0], inputs_embeds[1]
        
        expert_dtype = self.gemma_expert.model.layers[0].input_layernorm.dense.weight.dtype
        if self.training:
            self.clear_mixed_prefix_kv_cache()
        if adarms_cond is not None:
            if adarms_cond[0] is not None:
                adarms_cond[0] = adarms_cond[0].to(expert_dtype)
            if adarms_cond[1] is not None:
                adarms_cond[1] = adarms_cond[1].to(expert_dtype)
                
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if suffix_embs is None:
                expert_out = self.gemma_expert.model(
                    inputs_embeds=prefix_embs.to(expert_dtype),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None
                )
                return [expert_out.last_hidden_state, None], expert_out.past_key_values
                
            elif prefix_embs is None:
                expert_out = self.gemma_expert.model(
                    inputs_embeds=suffix_embs.to(expert_dtype),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1] if adarms_cond is not None else None
                )
                return [None, expert_out.last_hidden_state], expert_out.past_key_values
                
            else:
                combined_embs = torch.cat([prefix_embs.to(expert_dtype), suffix_embs.to(expert_dtype)], dim=1)
                expert_out = self.gemma_expert.model(
                    inputs_embeds=combined_embs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1] if adarms_cond is not None else None
                )
                out_hidden = expert_out.last_hidden_state
                prefix_out = out_hidden[:, :prefix_embs.shape[1]]
                suffix_out = out_hidden[:, prefix_embs.shape[1]:]
                return [prefix_out, suffix_out], None

class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self.last_inference_timing_ms: dict[str, float] = {}
        self._state_injection_debug_counter = 0
        self._cached_prefix_layer_states: torch.Tensor | None = None
        self._short_term_memory_debug_counter = 0

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)
        self.vggt_omega_prefix_adapter = None

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        if getattr(config, "use_rynnbrain", False):
            self.paligemma_with_expert = RynnBrainWithExpertModel(
                action_expert_config,
                rynnbrain_path=config.rynnbrain_path,
                precision=config.dtype,
                train_expert_only=config.train_expert_only,
                enable_rynnbrain_lora=getattr(config, "enable_rynnbrain_lora", False),
                rynnbrain_lora_r=getattr(config, "rynnbrain_lora_r", 16),
                rynnbrain_lora_alpha=getattr(config, "rynnbrain_lora_alpha", 32),
                rynnbrain_lora_dropout=getattr(config, "rynnbrain_lora_dropout", 0.05),
                rynnbrain_lora_target_modules=list(getattr(config, "rynnbrain_lora_target_modules", [])),
                paligemma_tokenizer_path=getattr(config, "paligemma_tokenizer_path", None),
                enable_short_term_memory=getattr(config, "enable_short_term_memory", False),
                short_term_memory_include_proprio=getattr(config, "short_term_memory_include_proprio", True),
                short_term_memory_state_dim=config.max_state_dim,
                short_term_memory_temporal_every_n_layers=getattr(
                    config, "short_term_memory_temporal_every_n_layers", 4
                ),
                short_term_memory_drop_history_last_n_layers=getattr(
                    config, "short_term_memory_drop_history_last_n_layers", 2
                ),
                short_term_memory_current_token_keep_ratio=getattr(
                    config, "short_term_memory_current_token_keep_ratio", 1.0
                ),
            )
            if hasattr(self.paligemma_with_expert.rynnbrain.config, "vision_config"):
                setattr(
                    self.paligemma_with_expert.rynnbrain.config.vision_config,
                    "mem_debug_interval",
                    int(getattr(config, "short_term_memory_debug_interval", 0)),
                )
        else:
            self.paligemma_with_expert = PaliGemmaWithExpertModel(
                paligemma_config,
                action_expert_config,
                use_adarms=[False, True],
                precision=config.dtype,
                image_size=config.image_resolution[0],
                freeze_vision_encoder=config.freeze_vision_encoder,
                train_expert_only=config.train_expert_only,
            )
        if getattr(config, "enable_vggt_omega_prefix_token", False):
            self.vggt_omega_prefix_adapter = VGGTOmegaPrefixAdapter(
                checkpoint_path=str(config.vggt_omega_checkpoint_path),
                output_dim=action_expert_config.width,
                freeze_backbone=bool(getattr(config, "vggt_omega_freeze", True)),
            )
        
        pytorch_compile_gemma_mode = getattr(config, "pytorch_compile_gemma_mode", "")
        if pytorch_compile_gemma_mode:
            torch.set_float32_matmul_precision("high")
            backbone_name = type(self.paligemma_with_expert).__name__
            logging.info(
                "Compiling %s.forward with torch.compile mode=%s",
                backbone_name,
                pytorch_compile_gemma_mode,
            )
            self.paligemma_with_expert.forward = torch.compile(
                self.paligemma_with_expert.forward,
                mode=pytorch_compile_gemma_mode,
            )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        action_dim = int(config.output_features[ACTION].shape[0])
        joint_action_indices, gripper_action_indices = _resolve_action_head_indices(
            getattr(config, "gripper_action_indices", None), action_dim
        )
        self._joint_action_indices = tuple(joint_action_indices)
        self._gripper_action_indices = tuple(gripper_action_indices)
        self._behavior_b1k_action_head_groups = (
            _resolve_behavior_b1k_semantic_action_head_groups(action_dim)
            if getattr(config, "enable_behavior_b1k_semantic_action_heads", False)
            else tuple()
        )
        self.behavior_b1k_action_out_projs = (
            nn.ModuleDict(
                {
                    group_name: nn.Linear(action_expert_config.width, len(group_indices))
                    for group_name, group_indices in self._behavior_b1k_action_head_groups
                }
            )
            if self._behavior_b1k_action_head_groups
            else None
        )
        self.joint_action_out_proj = (
            nn.Linear(action_expert_config.width, len(self._joint_action_indices))
            if getattr(config, "enable_split_action_heads", False)
            else None
        )
        self.gripper_action_out_proj = (
            nn.Linear(action_expert_config.width, len(self._gripper_action_indices))
            if getattr(config, "enable_split_action_heads", False)
            else None
        )
        self.clip_running_status_out_proj = nn.Linear(action_expert_config.width, 1)
        self.task_id_out_proj = nn.Linear(action_expert_config.width, config.task_aux_num_classes)
        self.box_aux_shape = tuple(config.input_features[config.box_aux_key].shape) if config.enable_box_aux_loss else None
        self.cross_center_aux_shape = (
            tuple(config.input_features[config.cross_center_aux_key].shape) if config.enable_cross_center_aux_loss else None
        )
        self.box_out_proj = (
            nn.Linear(action_expert_config.width, math.prod(self.box_aux_shape))
            if self.box_aux_shape is not None
            else None
        )
        self.cross_center_out_proj = (
            nn.Linear(action_expert_config.width, math.prod(self.cross_center_aux_shape))
            if self.cross_center_aux_shape is not None
            else None
        )

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_in_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.state_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.language_prefix_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.language_prefix_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.language_prefix_token = nn.Parameter(torch.zeros(action_expert_config.width))
        self.state_prefix_token = nn.Parameter(torch.zeros(action_expert_config.width))
        self.clip_running_status_in_proj = nn.Linear(1, action_expert_config.width)
        self.clip_running_status_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.clip_running_status_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.clip_running_status_prefix_token = nn.Parameter(torch.zeros(action_expert_config.width))
        self.flow_source_in_proj = nn.Linear(
            action_expert_config.width * int(config.flow_source_state_num_frames), action_expert_config.width
        )
        self.flow_source_out_proj = nn.Linear(
            action_expert_config.width, config.chunk_size * config.max_action_dim
        )
        self.future_action_aux_head = (
            nn.Linear(action_expert_config.width, int(config.future_action_aux_dim))
            if getattr(config, "enable_future_action_aux_loss", False)
            else None
        )
        self.future_action_aux_target_proj = (
            nn.Linear(config.chunk_size * config.max_action_dim, int(config.future_action_aux_dim), bias=False)
            if getattr(config, "enable_future_action_aux_loss", False)
            else None
        )
        self.group_consistency_proj = (
            nn.Linear(action_expert_config.width, int(config.group_consistency_project_dim))
            if getattr(config, "enable_group_consistency_loss", False)
            else None
        )
        if self.future_action_aux_target_proj is not None:
            self.future_action_aux_target_proj.requires_grad_(False)
        self._freeze_conditionally_unused_parameters()

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _freeze_conditionally_unused_parameters(self):
        if not getattr(self.config, "enable_clip_running_status_condition", False):
            for module in (
                self.clip_running_status_in_proj,
                self.clip_running_status_mlp_in,
                self.clip_running_status_mlp_out,
                self.clip_running_status_out_proj,
            ):
                module.requires_grad_(False)
            self.clip_running_status_prefix_token.requires_grad_(False)

        if not getattr(self.config, "enable_language_in_prefix_token", False):
            for module in (
                self.language_prefix_mlp_in,
                self.language_prefix_mlp_out,
            ):
                module.requires_grad_(False)
            self.language_prefix_token.requires_grad_(False)

        if getattr(self.config, "enable_behavior_b1k_semantic_action_heads", False):
            self.action_out_proj.requires_grad_(False)
            if self.joint_action_out_proj is not None:
                self.joint_action_out_proj.requires_grad_(False)
            if self.gripper_action_out_proj is not None:
                self.gripper_action_out_proj.requires_grad_(False)

        paligemma_with_expert = getattr(self, "paligemma_with_expert", None)
        if paligemma_with_expert is not None:
            gemma_expert = getattr(paligemma_with_expert, "gemma_expert", None)
            if gemma_expert is not None and hasattr(gemma_expert, "lm_head"):
                gemma_expert.lm_head.requires_grad_(False)

            rynnbrain = getattr(paligemma_with_expert, "rynnbrain", None)
            if rynnbrain is not None:
                language_model = getattr(getattr(rynnbrain, "model", None), "language_model", None)
                if language_model is not None and hasattr(language_model, "norm"):
                    language_model.norm.requires_grad_(False)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        if getattr(self.config, "use_rynnbrain", False):
            # Avoid nested checkpoint interactions that can trigger DDP "marked ready twice".
            self.paligemma_with_expert.rynnbrain.gradient_checkpointing_disable()
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        else:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        if getattr(self.config, "use_rynnbrain", False):
            self.paligemma_with_expert.rynnbrain.gradient_checkpointing_disable()
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        else:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _project_action(self, suffix_out: Tensor) -> Tensor:
        if self.behavior_b1k_action_out_projs is not None:
            action_out = torch.zeros(
                *suffix_out.shape[:-1],
                self.config.max_action_dim,
                device=suffix_out.device,
                dtype=suffix_out.dtype,
            )
            output_dtype = suffix_out.dtype
            for group_name, group_indices in self._behavior_b1k_action_head_groups:
                group_out = self.behavior_b1k_action_out_projs[group_name](suffix_out)
                output_dtype = torch.promote_types(output_dtype, group_out.dtype)
                if action_out.dtype != output_dtype:
                    action_out = action_out.to(dtype=output_dtype)
                action_out[..., list(group_indices)] = group_out.to(dtype=output_dtype)
            return self._zero_unused_action_dims(action_out)

        if self.joint_action_out_proj is None or self.gripper_action_out_proj is None:
            return self.action_out_proj(suffix_out)

        joint_out: Tensor | None = None
        gripper_out: Tensor | None = None
        if self._joint_action_indices:
            joint_out = self.joint_action_out_proj(suffix_out)
        if self._gripper_action_indices:
            gripper_out = self.gripper_action_out_proj(suffix_out)

        output_dtype = suffix_out.dtype
        if joint_out is not None:
            output_dtype = joint_out.dtype
        if gripper_out is not None:
            output_dtype = torch.promote_types(output_dtype, gripper_out.dtype)

        action_out = torch.zeros(
            *suffix_out.shape[:-1],
            self.config.max_action_dim,
            device=suffix_out.device,
            dtype=output_dtype,
        )
        if self._joint_action_indices:
            action_out[..., list(self._joint_action_indices)] = joint_out
        if self._gripper_action_indices:
            action_out[..., list(self._gripper_action_indices)] = gripper_out
        return self._zero_unused_action_dims(action_out)

    def initialize_split_action_heads_from_legacy_tensors(self, legacy_weight: Tensor, legacy_bias: Tensor) -> None:
        if self.joint_action_out_proj is None or self.gripper_action_out_proj is None:
            return
        with torch.no_grad():
            if self._joint_action_indices:
                self.joint_action_out_proj.weight.copy_(legacy_weight[list(self._joint_action_indices)])
                self.joint_action_out_proj.bias.copy_(legacy_bias[list(self._joint_action_indices)])
            if self._gripper_action_indices:
                self.gripper_action_out_proj.weight.copy_(legacy_weight[list(self._gripper_action_indices)])
                self.gripper_action_out_proj.bias.copy_(legacy_bias[list(self._gripper_action_indices)])

    def initialize_split_action_heads_from_legacy(self) -> None:
        self.initialize_split_action_heads_from_legacy_tensors(
            self.action_out_proj.weight.data, self.action_out_proj.bias.data
        )

    def initialize_behavior_b1k_semantic_action_heads_from_legacy_tensors(
        self, legacy_weight: Tensor, legacy_bias: Tensor
    ) -> None:
        if self.behavior_b1k_action_out_projs is None:
            return
        with torch.no_grad():
            for group_name, group_indices in self._behavior_b1k_action_head_groups:
                self.behavior_b1k_action_out_projs[group_name].weight.copy_(legacy_weight[list(group_indices)])
                self.behavior_b1k_action_out_projs[group_name].bias.copy_(legacy_bias[list(group_indices)])

    def initialize_behavior_b1k_semantic_action_heads_from_legacy(self) -> None:
        self.initialize_behavior_b1k_semantic_action_heads_from_legacy_tensors(
            self.action_out_proj.weight.data, self.action_out_proj.bias.data
        )

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _build_full_att_2d_masks(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Build the full [prefix+suffix]x[prefix+suffix] attention mask with configurable suffix mode."""
        suffix_attention_mode = getattr(self.config, "suffix_attention_mask", "causal")
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks, attention_mode="causal")
        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks, attention_mode=suffix_attention_mode
        )
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_to_suffix_masks = torch.zeros(
            (batch_size, prefix_len, suffix_len), dtype=torch.bool, device=prefix_pad_masks.device
        )
        suffix_to_prefix_masks = (
            suffix_pad_masks[:, :, None].expand(batch_size, suffix_len, prefix_len)
            & prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        )
        top_masks = torch.cat([prefix_att_2d_masks, prefix_to_suffix_masks], dim=2)
        bottom_masks = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)
        return torch.cat([top_masks, bottom_masks], dim=1)

    def _sync_for_timing(self, device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

    def _timing_mark(self, device: torch.device) -> float:
        self._sync_for_timing(device)
        return time.perf_counter()

    def _effective_action_dim(self) -> int:
        return int(self.config.output_features[ACTION].shape[0])

    def _zero_unused_action_dims(self, tensor: Tensor | None) -> Tensor | None:
        if tensor is None or not getattr(self.config, "enable_split_action_heads", False):
            return tensor
        action_dim = self._effective_action_dim()
        if tensor.shape[-1] <= action_dim:
            return tensor
        preserved = tensor[..., :action_dim]
        padded = torch.zeros_like(tensor[..., action_dim:])
        return torch.cat([preserved, padded], dim=-1)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return self._zero_unused_action_dims(noise)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _project_state_features(self, state_tensor: Tensor) -> Tensor:
        def state_proj_func(input_tensor: Tensor) -> Tensor:
            x = self.state_in_proj(input_tensor)
            x = F.silu(x)
            x = self.state_mlp_in(x)
            x = F.silu(x)
            x = self.state_mlp_out(x)
            return F.silu(x)

        return self._apply_checkpoint(state_proj_func, state_tensor)

    def _encode_state_history_tokens(self, state_history: Tensor | None) -> Tensor | None:
        if state_history is None:
            return None
        if state_history.ndim == 1:
            state_history = state_history[None, None, :]
        elif state_history.ndim == 2:
            state_history = state_history[:, None, :]
        elif state_history.ndim != 3:
            raise ValueError(f"Unsupported state_history ndim: {state_history.ndim}")

        state_history = state_history.to(device=self.state_in_proj.weight.device, dtype=torch.float32)
        state_history = pad_vector(state_history, self.config.max_state_dim)
        batch_size, history_len, _ = state_history.shape
        flat_state = state_history.reshape(batch_size * history_len, self.config.max_state_dim)
        state_tokens = self._project_state_features(flat_state)
        return state_tokens.reshape(batch_size, history_len, -1)

    def _build_state_history_flow_source(
        self,
        device: torch.device,
        dtype: torch.dtype,
        state_history: Tensor | None = None,
    ) -> Tensor:
        state_tokens = self._encode_state_history_tokens(state_history)
        if state_tokens is None:
            raise ValueError("state_history is required when flow_source uses state history")

        target_num_frames = int(getattr(self.config, "flow_source_state_num_frames", 1))
        if state_tokens.shape[1] > target_num_frames:
            state_tokens = state_tokens[:, -target_num_frames:, :]
        elif state_tokens.shape[1] < target_num_frames:
            pad_tokens = state_tokens[:, :1, :].expand(-1, target_num_frames - state_tokens.shape[1], -1)
            state_tokens = torch.cat([pad_tokens, state_tokens], dim=1)

        flat_tokens = state_tokens.reshape(state_tokens.shape[0], -1)

        def flow_source_proj_func(input_tensor: Tensor) -> Tensor:
            x = self.flow_source_in_proj(input_tensor)
            x = F.silu(x)
            return self.flow_source_out_proj(x)

        flow_source = self._apply_checkpoint(flow_source_proj_func, flat_tokens)
        flow_source = flow_source.reshape(state_tokens.shape[0], self.config.chunk_size, self.config.max_action_dim)
        return self._zero_unused_action_dims(flow_source).to(device=device, dtype=dtype)

    def _build_flow_source(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        state_history: Tensor | None = None,
    ) -> Tensor:
        flow_source_mode = getattr(self.config, "flow_source_mode", "gaussian")
        if flow_source_mode == "gaussian":
            return self.sample_noise(shape, device).to(dtype=dtype)
        if flow_source_mode == "state_history":
            return self._build_state_history_flow_source(
                device,
                dtype,
                state_history=state_history,
            )
        if flow_source_mode != "blend":
            raise ValueError(f"Unsupported flow_source_mode: {flow_source_mode}")

        gaussian_source = self.sample_noise(shape, device).to(dtype=dtype)
        state_source = self._build_state_history_flow_source(
            device,
            dtype,
            state_history=state_history,
        )
        blend_alpha = float(getattr(self.config, "flow_source_blend_alpha", 0.3))
        flow_source = (1.0 - blend_alpha) * gaussian_source + blend_alpha * state_source
        return self._zero_unused_action_dims(flow_source).to(dtype=dtype)

    def _compute_future_action_aux_loss(
        self,
        pooled_feature: Tensor,
        actions: Tensor,
        action_valid_mask: Tensor | None = None,
        action_dim_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.future_action_aux_head is None or self.future_action_aux_target_proj is None:
            return None, None

        pred_latent = self.future_action_aux_head(pooled_feature.to(dtype=torch.float32))
        target_actions = self._zero_unused_action_dims(actions.to(dtype=torch.float32))
        if action_valid_mask is not None:
            target_actions = target_actions * action_valid_mask.to(
                device=target_actions.device, dtype=target_actions.dtype
            ).unsqueeze(-1)
        if action_dim_valid_mask is not None:
            target_actions = target_actions * action_dim_valid_mask.to(
                device=target_actions.device, dtype=target_actions.dtype
            ).view(1, 1, -1)
        flat_actions = target_actions.reshape(actions.shape[0], -1)
        with torch.no_grad():
            target_latent = self.future_action_aux_target_proj(flat_actions)
        pred_latent = F.normalize(pred_latent, dim=-1)
        target_latent = F.normalize(target_latent, dim=-1)
        per_sample_loss = F.mse_loss(pred_latent, target_latent, reduction="none").mean(dim=-1)
        return per_sample_loss.mean(), per_sample_loss

    def project_group_consistency_feature(self, pooled_feature: Tensor) -> Tensor | None:
        if self.group_consistency_proj is None:
            return None
        return self.group_consistency_proj(pooled_feature.to(dtype=torch.float32))

    def _maybe_log_short_term_memory_debug(
        self,
        *,
        num_frames: int,
        num_cameras: int,
        debug_info: dict[str, int] | None,
        rynnbrain_inputs: dict,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if not (getattr(self.config, "use_rynnbrain", False) and getattr(self.config, "enable_short_term_memory", False)):
            return
        debug_interval = int(getattr(self.config, "short_term_memory_debug_interval", 0))
        if debug_interval <= 0:
            return
        self._short_term_memory_debug_counter += 1
        if self._short_term_memory_debug_counter % debug_interval != 0:
            return

        video_grid_thw = rynnbrain_inputs.get("video_grid_thw")
        pixel_values_videos = rynnbrain_inputs.get("pixel_values_videos")
        input_ids = rynnbrain_inputs.get("input_ids")
        video_token_count = -1
        if input_ids is not None:
            video_token_count = int((input_ids == self.paligemma_with_expert.rynnbrain.config.video_token_id).sum().item())
        batch_size = int(prefix_embs.shape[0])
        stm_num_frames = num_frames if debug_info is None else int(debug_info.get("short_term_memory_num_frames", num_frames))
        stm_stride = -1 if debug_info is None else int(debug_info.get("short_term_memory_stride", -1))
        strategy_idx = -1 if debug_info is None else int(debug_info.get("short_term_memory_strategy_idx", -1))
        images_per_sample = 0 if num_cameras <= 0 else stm_num_frames * num_cameras
        total_image_slots = batch_size * max(num_cameras, 0) * stm_num_frames
        temporal_bins = None
        if video_grid_thw is not None and len(video_grid_thw) > 0:
            temporal_bins = sorted({int(v[0]) for v in video_grid_thw.detach().cpu().tolist()})

        logging.info(
            "[PI05][mem] step=%d batch_size=%d cameras=%d strategy_idx=%d stm_num_frames=%d "
            "stm_stride=%d images_per_sample=%d total_image_slots=%d qwen_temporal_bins=%s "
            "video_token_count=%d attention_tokens=%d prefix_embs=%s",
            self._short_term_memory_debug_counter,
            batch_size,
            num_cameras,
            strategy_idx,
            stm_num_frames,
            stm_stride,
            images_per_sample,
            total_image_slots,
            temporal_bins,
            video_token_count,
            int(attention_mask.shape[1]),
            tuple(prefix_embs.shape),
        )

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        img_pad_masks=None,
        vggt_images: list[Tensor] | None = None,
        raw_texts: list[str] | None = None,
        clip_running_status: Tensor | None = None,
        state_history: Tensor | None = None,
        short_term_memory_debug_info: dict[str, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        self._cached_prefix_layer_states = None
        if getattr(self.config, "use_rynnbrain", False):
            use_short_term_memory = bool(getattr(self.config, "enable_short_term_memory", False)) and bool(images)
            num_frames = images[0].shape[1] if use_short_term_memory and images[0].ndim == 5 else 1
            import torch._dynamo as torch_dynamo
            torch_dynamo.config.suppress_errors = True
            torch_dynamo.config.verbose = False
            
            # 1. 禁用 Processor 和 Tokenizer 内的 Dynamo 追踪
            @torch_dynamo.disable
            def process_inputs(tokens_inner, images_inner, img_pad_masks_inner, raw_texts_inner):
                bsize = tokens_inner.shape[0]

                if raw_texts_inner is None:
                    raw_texts_local = self.paligemma_with_expert.paligemma_tokenizer.batch_decode(
                        tokens_inner, skip_special_tokens=True
                    )
                else:
                    raw_texts_local = list(raw_texts_inner)
                    if len(raw_texts_local) == 1 and bsize > 1:
                        raw_texts_local = raw_texts_local * bsize
                    if len(raw_texts_local) != bsize:
                        raise ValueError(
                            f"Expected {bsize} raw texts for RynnBrain inputs, got {len(raw_texts_local)}."
                        )

                conversations = []
                batch_images = []
                batch_videos = []

                def _is_padding_frame(img_tensor: torch.Tensor) -> bool:
                    return bool(torch.all(img_tensor == -1).item())

                def _is_explicitly_padded(
                    pad_tensor: torch.Tensor | None,
                    sample_idx: int,
                    frame_idx: int | None = None,
                ) -> bool | None:
                    if pad_tensor is None:
                        return None
                    if pad_tensor.ndim == 1:
                        return bool(pad_tensor[sample_idx].item())
                    if frame_idx is not None and pad_tensor.ndim >= 2:
                        return bool(pad_tensor[sample_idx, frame_idx].item())
                    return None

                def _should_skip_frame(
                    img_tensor: torch.Tensor,
                    pad_tensor: torch.Tensor | None,
                    sample_idx: int,
                    frame_idx: int | None = None,
                ) -> bool:
                    explicit_pad = _is_explicitly_padded(pad_tensor, sample_idx, frame_idx)
                    if explicit_pad is not None:
                        return explicit_pad
                    return _is_padding_frame(img_tensor)

                def _to_preprocessed_video_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
                    # Training images are already normalized to [-1, 1]. Keep them as contiguous
                    # tensors so the video processor can run directly on torch tensors.
                    return img_tensor.detach().contiguous()

                def _to_uint8_cpu_image_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
                    # Static images on the RynnBrain path are already kept in [0, 1] after resize/pad.
                    img_tensor = img_tensor.detach().mul(255.0).clamp(0, 255)
                    return img_tensor.to(device="cpu", dtype=torch.uint8).contiguous()

                if use_short_term_memory and num_frames > 1:
                    for b_idx in range(bsize):
                        content = []
                        sample_images = []
                        sample_videos = []
                        for cam_imgs, cam_pad_mask in zip(images_inner, img_pad_masks_inner, strict=True):
                            if cam_imgs.ndim != 5:
                                continue
                            prepared_frames: list[torch.Tensor | None] = []
                            for frame_idx in range(num_frames):
                                img_tensor = cam_imgs[b_idx, frame_idx]
                                if _should_skip_frame(img_tensor, cam_pad_mask, b_idx, frame_idx):
                                    prepared_frames.append(None)
                                    continue
                                prepared_frames.append(_to_preprocessed_video_tensor(img_tensor))
                            first_valid_frame = next((frame for frame in prepared_frames if frame is not None), None)
                            if first_valid_frame is not None:
                                frames = []
                                last_valid_frame = first_valid_frame
                                for frame in prepared_frames:
                                    if frame is None:
                                        frames.append(last_valid_frame.clone())
                                    else:
                                        last_valid_frame = frame
                                        frames.append(frame)
                                video_tensor = torch.stack(frames, dim=0)
                                content.append({"type": "video", "video": video_tensor})
                                sample_videos.append(video_tensor)
                        content.append({"type": "text", "text": raw_texts_local[b_idx]})
                        conversations.append([{"role": "user", "content": content}])
                        batch_images.append(sample_images)
                        batch_videos.append(sample_videos)
                else:
                    for b_idx in range(bsize):
                        for frame_idx in range(num_frames):
                            frame_images = []
                            for cam_imgs, cam_pad_mask in zip(images_inner, img_pad_masks_inner, strict=True):
                                img_tensor = cam_imgs[b_idx, frame_idx] if cam_imgs.ndim == 5 else cam_imgs[b_idx]
                                if _should_skip_frame(
                                    img_tensor,
                                    cam_pad_mask,
                                    b_idx,
                                    frame_idx if cam_imgs.ndim == 5 else None,
                                ):
                                    continue
                                frame_images.append(_to_uint8_cpu_image_tensor(img_tensor))

                            content = []
                            for img in frame_images:
                                content.append({"type": "image", "image": img})
                            content.append({"type": "text", "text": raw_texts_local[b_idx]})
                            conversations.append([{"role": "user", "content": content}])
                            batch_images.append(frame_images)
                            batch_videos.append([])

                processor = self.paligemma_with_expert.rynnbrain_processor
                prompts = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                has_images = any(sample_images for sample_images in batch_images)
                has_videos = any(sample_videos for sample_videos in batch_videos)
                if has_videos and not has_images and use_short_term_memory and num_frames > 1:
                    video_inputs = processor.video_processor(
                        videos=batch_videos,
                        do_sample_frames=False,
                        do_convert_rgb=False,
                        do_rescale=False,
                        do_normalize=False,
                        return_tensors="pt",
                    )

                    processed_prompts = prompts.copy() if isinstance(prompts, list) else [prompts]
                    merge_length = processor.video_processor.merge_size**2
                    merge_size = processor.video_processor.merge_size
                    keep_ratio = (
                        float(getattr(self.config, "short_term_memory_current_token_keep_ratio", 1.0))
                        if int(getattr(self.config, "short_term_memory_drop_history_last_n_layers", 0)) > 0
                        else 1.0
                    )
                    keep_scale = math.sqrt(max(0.0, min(1.0, keep_ratio)))
                    video_grid_thw = video_inputs["video_grid_thw"]
                    index = 0
                    for i in range(len(processed_prompts)):
                        while processor.video_token in processed_prompts[i]:
                            grid_height = int(video_grid_thw[index][1].item())
                            grid_width = int(video_grid_thw[index][2].item())
                            if keep_ratio < 1.0:
                                reduced_height = max(
                                    merge_size,
                                    int(round(grid_height * keep_scale / merge_size)) * merge_size,
                                )
                                reduced_width = max(
                                    merge_size,
                                    int(round(grid_width * keep_scale / merge_size)) * merge_size,
                                )
                                reduced_height = min(grid_height, reduced_height)
                                reduced_width = min(grid_width, reduced_width)
                                if reduced_height == grid_height and reduced_width == grid_width:
                                    if grid_width - merge_size >= merge_size:
                                        reduced_width = grid_width - merge_size
                                    elif grid_height - merge_size >= merge_size:
                                        reduced_height = grid_height - merge_size
                                grid_height = max(merge_size, reduced_height)
                                grid_width = max(merge_size, reduced_width)
                            frame_seqlen = (grid_height * grid_width) // merge_length
                            video_placeholder = (
                                processor.vision_start_token
                                + "<|placeholder|>" * frame_seqlen
                                + processor.vision_end_token
                            )
                            wrapped_video_token = (
                                f"{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}"
                            )
                            if wrapped_video_token in processed_prompts[i]:
                                processed_prompts[i] = processed_prompts[i].replace(
                                    wrapped_video_token,
                                    video_placeholder,
                                    1,
                                )
                            else:
                                processed_prompts[i] = processed_prompts[i].replace(processor.video_token, video_placeholder, 1)
                            index += 1
                        processed_prompts[i] = processed_prompts[i].replace("<|placeholder|>", processor.video_token)

                    tokenizer_kwargs = {
                        "padding": True,
                        "return_tensors": "pt",
                        "return_token_type_ids": False,
                    }
                    single_prompt = processed_prompts[0] if processed_prompts else None
                    bos_token = getattr(processor.tokenizer, "bos_token", None)
                    if bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(bos_token):
                        tokenizer_kwargs["add_special_tokens"] = False
                    text_inputs = processor.tokenizer(processed_prompts, **tokenizer_kwargs)
                    processor._check_special_mm_tokens(processed_prompts, text_inputs, modalities=["video"])
                    rynnbrain_inputs = {**text_inputs, **video_inputs}
                else:
                    processor_kwargs = {
                        "text": prompts,
                        "images": batch_images if has_images else None,
                        "videos": batch_videos if has_videos else None,
                        "padding": True,
                        "return_dict": True,
                        "return_tensors": "pt",
                        "compress_video_tokens": use_short_term_memory and num_frames > 1,
                        "current_token_keep_ratio": (
                            float(getattr(self.config, "short_term_memory_current_token_keep_ratio", 1.0))
                            if use_short_term_memory
                            and num_frames > 1
                            and int(getattr(self.config, "short_term_memory_drop_history_last_n_layers", 0)) > 0
                            else 1.0
                        ),
                        "do_sample_frames": False,
                    }
                    single_prompt = prompts[0] if isinstance(prompts, list) and prompts else prompts
                    bos_token = getattr(processor.tokenizer, "bos_token", None)
                    if bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(bos_token):
                        processor_kwargs["add_special_tokens"] = False
                    rynnbrain_inputs = processor(**processor_kwargs)
                # Ensure tensors are contiguous and on correct device to avoid dynamo guards
                return {k: v.contiguous().to(tokens_inner.device) if isinstance(v, torch.Tensor) else v for k, v in rynnbrain_inputs.items()}

            normalized_img_pad_masks = (
                list(img_pad_masks)
                if img_pad_masks is not None
                else [None] * len(images)
            )
            rynnbrain_inputs = process_inputs(tokens, images, normalized_img_pad_masks, raw_texts)

            # 2. 调用模型提取特征 (这一步是 nn.Module，支持 Dynamo)
            prefix_embs = self.paligemma_with_expert.get_prefix_embs(rynnbrain_inputs)
            attention_mask = rynnbrain_inputs["attention_mask"].bool()
            prefix_pad_masks = attention_mask
            prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.bool)
            language_embs, language_pad_masks = self._get_pure_language_embeddings(tokens, masks, raw_texts)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_language_token_to_prefix(
                prefix_embs, prefix_pad_masks, prefix_att_masks, language_embs, language_pad_masks
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_vggt_omega_tokens_to_prefix(
                prefix_embs, prefix_pad_masks, prefix_att_masks, vggt_images
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_state_tokens_to_prefix(
                prefix_embs, prefix_pad_masks, prefix_att_masks, state_history
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_clip_running_status_to_prefix(
                prefix_embs, prefix_pad_masks, prefix_att_masks, clip_running_status
            )
            self._maybe_log_short_term_memory_debug(
                num_frames=num_frames,
                num_cameras=len(images),
                debug_info=short_term_memory_debug_info,
                rynnbrain_inputs=rynnbrain_inputs,
                prefix_embs=prefix_embs,
                attention_mask=attention_mask,
            )

            return prefix_embs, prefix_pad_masks, prefix_att_masks

        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        embs, pad_masks, att_masks = self._append_language_token_to_prefix(
            embs, pad_masks, att_masks, lang_emb, masks.to(device=lang_emb.device, dtype=torch.bool)
        )
        embs, pad_masks, att_masks = self._append_vggt_omega_tokens_to_prefix(
            embs, pad_masks, att_masks, vggt_images
        )
        embs, pad_masks, att_masks = self._append_state_tokens_to_prefix(
            embs, pad_masks, att_masks, state_history
        )
        embs, pad_masks, att_masks = self._append_clip_running_status_to_prefix(
            embs, pad_masks, att_masks, clip_running_status
        )
        return embs, pad_masks, att_masks

    def _encode_vggt_omega_prefix_token(self, vggt_images: list[Tensor] | None) -> Tensor | None:
        if self.vggt_omega_prefix_adapter is None or vggt_images is None:
            return None
        if len(vggt_images) == 0:
            return None
        return self.vggt_omega_prefix_adapter(vggt_images)

    def _append_vggt_omega_tokens_to_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        vggt_images: list[Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vggt_prefix_token = self._encode_vggt_omega_prefix_token(vggt_images)
        if vggt_prefix_token is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks
        vggt_prefix_token = vggt_prefix_token.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        vggt_pad_masks = torch.ones(
            vggt_prefix_token.shape[:2], dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device
        )
        vggt_att_masks = torch.zeros(
            vggt_prefix_token.shape[:2], dtype=prefix_att_masks.dtype, device=prefix_att_masks.device
        )
        if self._cached_prefix_layer_states is not None:
            self._cached_prefix_layer_states = None
        prefix_embs = torch.cat([prefix_embs, vggt_prefix_token], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, vggt_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, vggt_att_masks], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _get_pure_language_embeddings(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        raw_texts: list[str] | None = None,
    ) -> tuple[Tensor | None, torch.Tensor | None]:
        if not getattr(self.config, "enable_language_in_prefix_token", False):
            return None, None
        if getattr(self.config, "use_rynnbrain", False):
            if raw_texts is None or len(raw_texts) == 0:
                return None, None
            if not hasattr(self.paligemma_with_expert, "get_text_prefix_embs"):
                return None, None
            language_embs, attention_mask = self.paligemma_with_expert.get_text_prefix_embs(
                list(raw_texts),
                device=tokens.device,
            )
            return language_embs, attention_mask

        if tokens.ndim != 2 or masks.ndim != 2:
            return None, None

        def language_embed_func(input_tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(input_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            lang_emb = lang_emb * math.sqrt(lang_emb_dim)
            align_proj = getattr(self.paligemma_with_expert, "align_proj", None)
            target_dim = int(self.language_prefix_mlp_in.in_features)
            if align_proj is not None and lang_emb.shape[-1] != target_dim:
                lang_emb = align_proj(lang_emb.to(device=align_proj.weight.device, dtype=align_proj.weight.dtype))
            return lang_emb

        language_embs = self._apply_checkpoint(language_embed_func, tokens)
        language_pad_masks = masks.to(device=language_embs.device, dtype=torch.bool)
        return language_embs, language_pad_masks

    def _encode_language_as_prefix_token(
        self,
        language_embs: torch.Tensor | None,
        language_pad_masks: torch.Tensor | None,
    ) -> Tensor | None:
        if not getattr(self.config, "enable_language_in_prefix_token", False):
            return None
        if language_embs is None or language_pad_masks is None:
            return None
        if language_embs.ndim != 3 or language_pad_masks.ndim != 2:
            return None

        valid_mask = language_pad_masks.to(dtype=torch.bool)
        if valid_mask.shape[0] == 0 or valid_mask.shape[1] == 0:
            return None

        valid_mask_f = valid_mask[:, :, None].to(device=language_embs.device, dtype=language_embs.dtype)
        pooled_token = (language_embs * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1).clamp(min=1.0)

        def language_proj_func(input_tensor: Tensor) -> Tensor:
            input_tensor = input_tensor.to(
                device=self.language_prefix_mlp_in.weight.device,
                dtype=self.language_prefix_mlp_in.weight.dtype,
            )
            x = self.language_prefix_mlp_in(input_tensor)
            x = F.silu(x)
            x = self.language_prefix_mlp_out(x)
            return F.silu(x)

        language_prefix_token = self._apply_checkpoint(language_proj_func, pooled_token)
        language_prefix_token = language_prefix_token[:, None, :]
        return language_prefix_token + self.language_prefix_token[None, None, :].to(
            device=language_prefix_token.device,
            dtype=language_prefix_token.dtype,
        )

    def _append_language_token_to_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        language_embs: torch.Tensor | None,
        language_pad_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_prefix_token = self._encode_language_as_prefix_token(language_embs, language_pad_masks)
        if language_prefix_token is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        language_prefix_token = language_prefix_token.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        prefix_valid_mask = prefix_pad_masks.to(dtype=torch.bool)
        prefix_valid_mask_f = prefix_valid_mask[:, :, None].to(device=prefix_embs.device, dtype=torch.float32)
        prefix_token_norm = prefix_embs.detach().float().norm(dim=-1, keepdim=True)
        target_norm = (prefix_token_norm * prefix_valid_mask_f).sum(dim=1, keepdim=True) / prefix_valid_mask_f.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        source_norm = language_prefix_token.detach().float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        norm_scale = (target_norm / source_norm).clamp(min=0.25, max=16.0).to(
            device=language_prefix_token.device,
            dtype=language_prefix_token.dtype,
        )
        language_prefix_token = language_prefix_token * norm_scale
        language_pad_masks = torch.ones(
            language_prefix_token.shape[:2], dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device
        )
        language_att_masks = torch.zeros(
            language_prefix_token.shape[:2], dtype=prefix_att_masks.dtype, device=prefix_att_masks.device
        )
        if self._cached_prefix_layer_states is not None:
            self._cached_prefix_layer_states = None
        prefix_embs = torch.cat([prefix_embs, language_prefix_token], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, language_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, language_att_masks], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _encode_state_for_action_time(self, state: Tensor | None) -> Tensor | None:
        if (not self.config.enable_state_in_action_time_emb) or state is None:
            return None
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim == 3:
            state = state[:, -1, :]
        state = state.to(device=self.state_in_proj.weight.device, dtype=torch.float32)
        state = pad_vector(state, self.config.max_state_dim)
        return self._project_state_features(state)

    def _encode_state_history_as_prefix_tokens(self, state_history: Tensor | None) -> Tensor | None:
        if (not getattr(self.config, "enable_state_in_prefix_tokens", False)) or state_history is None:
            return None
        state_tokens = self._encode_state_history_tokens(state_history)
        if state_tokens is None:
            return None
        return state_tokens + self.state_prefix_token[None, None, :].to(
            device=state_tokens.device, dtype=state_tokens.dtype
        )

    def _append_state_tokens_to_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        state_history: Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_prefix_tokens = self._encode_state_history_as_prefix_tokens(state_history)
        if state_prefix_tokens is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        state_prefix_tokens = state_prefix_tokens.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        state_pad_masks = torch.ones(
            state_prefix_tokens.shape[:2], dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device
        )
        state_att_masks = torch.zeros(
            state_prefix_tokens.shape[:2], dtype=prefix_att_masks.dtype, device=prefix_att_masks.device
        )
        if self._cached_prefix_layer_states is not None:
            # Cached layer states only cover the original visual/language prefix; disable mixed-cache reuse.
            self._cached_prefix_layer_states = None
        prefix_embs = torch.cat([prefix_embs, state_prefix_tokens], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, state_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, state_att_masks], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _encode_clip_running_status_as_prefix_token(self, clip_running_status: Tensor | None) -> Tensor | None:
        if (not getattr(self.config, "enable_clip_running_status_condition", False)) or clip_running_status is None:
            return None
        if clip_running_status.ndim == 0:
            current_progress = clip_running_status[None, None]
        elif clip_running_status.ndim == 1:
            current_progress = clip_running_status[:, None]
        else:
            current_progress = clip_running_status[:, :1]
        current_progress = current_progress.to(
            device=self.clip_running_status_in_proj.weight.device,
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)

        def progress_proj_func(input_tensor: Tensor) -> Tensor:
            x = self.clip_running_status_in_proj(input_tensor)
            x = F.silu(x)
            x = self.clip_running_status_mlp_in(x)
            x = F.silu(x)
            x = self.clip_running_status_mlp_out(x)
            return F.silu(x)

        progress_token = self._apply_checkpoint(progress_proj_func, current_progress)
        progress_token = progress_token[:, None, :]
        return progress_token + self.clip_running_status_prefix_token[None, None, :].to(
            device=progress_token.device, dtype=progress_token.dtype
        )

    def _append_clip_running_status_to_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        clip_running_status: Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        progress_prefix_token = self._encode_clip_running_status_as_prefix_token(clip_running_status)
        if progress_prefix_token is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        progress_prefix_token = progress_prefix_token.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        progress_pad_masks = torch.ones(
            progress_prefix_token.shape[:2], dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device
        )
        progress_att_masks = torch.zeros(
            progress_prefix_token.shape[:2], dtype=prefix_att_masks.dtype, device=prefix_att_masks.device
        )
        if self._cached_prefix_layer_states is not None:
            self._cached_prefix_layer_states = None
        prefix_embs = torch.cat([prefix_embs, progress_prefix_token], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, progress_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, progress_att_masks], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def embed_suffix(self, noisy_actions, timestep, state_cond: Tensor | None = None):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        if state_cond is not None:
            action_time_emb = action_time_emb + state_cond[:, None, :].to(
                device=action_time_emb.device, dtype=action_time_emb.dtype
            )
            self._state_injection_debug_counter += 1
            debug_interval = int(getattr(self.config, "state_in_action_time_emb_debug_interval", 100))
            if debug_interval > 0 and self._state_injection_debug_counter % debug_interval == 0:
                action_norm = action_emb.detach().float().norm(dim=-1).mean().item()
                state_norm = state_cond.detach().float().norm(dim=-1).mean().item()
                ratio = state_norm / max(action_norm, 1e-6)
                logging.info(
                    "[PI05][state-inject] step=%d ||state_emb||/||action_emb||=%.6f "
                    "(state_norm=%.6f action_norm=%.6f)",
                    self._state_injection_debug_counter,
                    ratio,
                    state_norm,
                    action_norm,
                )
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        img_pad_masks,
        tokens,
        masks,
        actions,
        action_valid_mask: Tensor | None = None,
        action_dim_valid_mask: Tensor | None = None,
        vggt_images: list[Tensor] | None = None,
        raw_texts: list[str] | None = None,
        clip_running_status: Tensor | None = None,
        state: Tensor | None = None,
        state_history: Tensor | None = None,
        flow_source_state_history: Tensor | None = None,
        short_term_memory_debug_info: dict[str, int] | None = None,
        noise=None,
        time=None,
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor,
    ]:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            flow_source = self._build_flow_source(
                actions.shape,
                actions.device,
                actions.dtype,
                state_history=flow_source_state_history,
            )
        else:
            flow_source = self._zero_unused_action_dims(noise.to(device=actions.device, dtype=actions.dtype))

        if action_dim_valid_mask is not None:
            action_dim_valid_mask = action_dim_valid_mask.to(device=actions.device, dtype=torch.bool)
            if action_dim_valid_mask.ndim != 1 or action_dim_valid_mask.shape[0] != actions.shape[-1]:
                raise ValueError(
                    "action_dim_valid_mask must have shape [action_dim]: "
                    f"{tuple(action_dim_valid_mask.shape)} != ({actions.shape[-1]},)"
                )
            dim_weights = action_dim_valid_mask.to(dtype=actions.dtype).view(1, 1, -1)
            # Inactive arms must not leak their target values or random flow
            # noise into the shared action token, even though their output loss
            # dimensions are excluded later as well.
            actions = actions * dim_weights
            flow_source = flow_source * dim_weights

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * flow_source + (1 - time_expanded) * actions
        x_t = self._zero_unused_action_dims(x_t)
        u_t = flow_source - actions
        u_t = self._zero_unused_action_dims(u_t)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            img_pad_masks=img_pad_masks,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status,
            state_history=state_history,
            short_term_memory_debug_info=short_term_memory_debug_info,
        )
        
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        state_cond = self._encode_state_for_action_time(state)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time, state_cond=state_cond
        )
        if action_valid_mask is not None:
            action_valid_mask = action_valid_mask.to(device=suffix_pad_masks.device, dtype=torch.bool)
            if action_valid_mask.shape != suffix_pad_masks.shape:
                raise ValueError(
                    "action_valid_mask shape must match action suffix mask: "
                    f"{tuple(action_valid_mask.shape)} != {tuple(suffix_pad_masks.shape)}"
                )
            suffix_pad_masks = suffix_pad_masks & action_valid_mask
        if getattr(self.config, "use_rynnbrain", False):
            if self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        else:
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_2d_masks = self._build_full_att_2d_masks(
            prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                prefix_layer_states=self._cached_prefix_layer_states,
            )
            return suffix_out

        if getattr(self.config, "use_rynnbrain", False):
            # In RynnBrain mode, avoid an extra outer checkpoint wrapper over the full VLM+expert call.
            suffix_out = forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
        else:
            suffix_out = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self._project_action(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        if action_valid_mask is None:
            pooled_feature = suffix_out.mean(dim=1)
        else:
            pool_weights = action_valid_mask.to(device=suffix_out.device, dtype=suffix_out.dtype).unsqueeze(-1)
            pooled_feature = (suffix_out * pool_weights).sum(dim=1) / pool_weights.sum(dim=1).clamp_min(1.0)
        task_id_logits = self.task_id_out_proj(pooled_feature)
        predicted_boxes = None
        if self.box_out_proj is not None:
            predicted_boxes = self.box_out_proj(pooled_feature).reshape(pooled_feature.shape[0], *self.box_aux_shape)
        predicted_cross_center = None
        if self.cross_center_out_proj is not None:
            predicted_cross_center = self.cross_center_out_proj(pooled_feature).reshape(
                pooled_feature.shape[0], *self.cross_center_aux_shape
            )
        predicted_clip_running_status = None
        if clip_running_status is not None:
            predicted_clip_running_status = torch.sigmoid(self.clip_running_status_out_proj(suffix_out)).squeeze(-1)
        predicted_action = flow_source - v_t
        future_action_aux_loss, future_action_aux_per_sample_loss = self._compute_future_action_aux_loss(
            pooled_feature,
            actions,
            action_valid_mask=action_valid_mask,
            action_dim_valid_mask=action_dim_valid_mask,
        )
        return (
            F.mse_loss(u_t, v_t, reduction="none"),
            predicted_clip_running_status,
            predicted_action,
            task_id_logits,
            predicted_boxes,
            predicted_cross_center,
            future_action_aux_loss,
            future_action_aux_per_sample_loss,
            flow_source,
            pooled_feature,
        )

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        img_pad_masks,
        tokens,
        masks,
        vggt_images: list[Tensor] | None = None,
        raw_texts: list[str] | None = None,
        clip_running_status: Tensor | None = None,
        state: Tensor | None = None,
        state_history: Tensor | None = None,
        flow_source_state_history: Tensor | None = None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            x_t = self._build_flow_source(
                actions_shape,
                device,
                torch.float32,
                state_history=flow_source_state_history,
            )
        else:
            x_t = self._zero_unused_action_dims(noise.to(device=device, dtype=torch.float32))
        state_cond = self._encode_state_for_action_time(state)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            img_pad_masks=img_pad_masks,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status,
            state_history=state_history,
        )
        use_prefix_cache = not getattr(self.config, "use_rynnbrain", False)
        past_key_values = None
        if use_prefix_cache:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        dt = -1.0 / num_steps
        try:
            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

                def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                    return self.denoise_step(
                        prefix_embs=prefix_embs if not use_prefix_cache else None,
                        prefix_pad_masks=prefix_pad_masks,
                        prefix_att_masks=prefix_att_masks if not use_prefix_cache else None,
                        past_key_values=past_key_values,
                        x_t=input_x_t,
                        timestep=current_timestep,
                        state_cond=state_cond,
                    )

                if self._rtc_enabled():
                    inference_delay = kwargs.get("inference_delay")
                    prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                    execution_horizon = kwargs.get("execution_horizon")

                    v_t = self.rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=time,
                        original_denoise_step_partial=denoise_step_partial_call,
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = denoise_step_partial_call(x_t)

                x_t = self._zero_unused_action_dims(x_t + dt * v_t)

                if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                    self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        finally:
            pass

        return self._zero_unused_action_dims(x_t)

    @torch.no_grad()
    def sample_actions_and_status(
        self,
        images,
        img_masks,
        img_pad_masks,
        tokens,
        masks,
        vggt_images: list[Tensor] | None = None,
        raw_texts: list[str] | None = None,
        clip_running_status: Tensor | None = None,
        state: Tensor | None = None,
        state_history: Tensor | None = None,
        flow_source_state_history: Tensor | None = None,
        noise=None,
        num_steps=None,
        return_task_logits: bool = False,
        return_aux_predictions: bool = False,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None, Tensor | None] | tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device
        timing_device = tokens.device
        t0 = self._timing_mark(timing_device)

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            x_t = self._build_flow_source(
                actions_shape,
                device,
                torch.float32,
                state_history=flow_source_state_history,
            )
        else:
            x_t = self._zero_unused_action_dims(noise.to(device=device, dtype=torch.float32))
        state_cond = self._encode_state_for_action_time(state)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            img_pad_masks=img_pad_masks,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status,
            state_history=state_history,
        )
        t1 = self._timing_mark(timing_device)
        use_prefix_cache = not getattr(self.config, "use_rynnbrain", False)
        past_key_values = None
        if use_prefix_cache:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        t2 = self._timing_mark(timing_device)

        dt = -1.0 / num_steps
        status_suffix_out: Tensor | None = None
        final_time = max(0.0, 1.0 + (num_steps - 1) * dt)

        try:
            for step in range(num_steps):
                step_time = 1.0 + step * dt
                time_tensor = torch.tensor(step_time, dtype=torch.float32, device=device).expand(bsize)

                def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                    return self.denoise_step(
                        prefix_embs=prefix_embs if not use_prefix_cache else None,
                        prefix_pad_masks=prefix_pad_masks,
                        prefix_att_masks=prefix_att_masks if not use_prefix_cache else None,
                        past_key_values=past_key_values,
                        x_t=input_x_t,
                        timestep=current_timestep,
                        state_cond=state_cond,
                    )

                if self._rtc_enabled():
                    inference_delay = kwargs.get("inference_delay")
                    prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                    execution_horizon = kwargs.get("execution_horizon")

                    v_t = self.rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=step_time,
                        original_denoise_step_partial=denoise_step_partial_call,
                        execution_horizon=execution_horizon,
                    )
                else:
                    if step == num_steps - 1:
                        v_t, status_suffix_out = self.denoise_step(
                            prefix_embs=prefix_embs if not use_prefix_cache else None,
                            prefix_pad_masks=prefix_pad_masks,
                            prefix_att_masks=prefix_att_masks if not use_prefix_cache else None,
                            past_key_values=past_key_values,
                            x_t=x_t,
                            timestep=time_tensor,
                            state_cond=state_cond,
                            return_suffix_out=True,
                        )
                    else:
                        v_t = denoise_step_partial_call(x_t)

                x_t = self._zero_unused_action_dims(x_t + dt * v_t)

                if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                    self.rtc_processor.track(time=step_time, x_t=x_t, v_t=v_t)

            actions = self._zero_unused_action_dims(x_t)
            if status_suffix_out is None:
                fallback_time = torch.tensor(final_time, dtype=torch.float32, device=device).expand(bsize)
                _, status_suffix_out = self.denoise_step(
                    prefix_embs=prefix_embs if not use_prefix_cache else None,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks if not use_prefix_cache else None,
                    past_key_values=past_key_values,
                    x_t=actions,
                    timestep=fallback_time,
                    state_cond=state_cond,
                    return_suffix_out=True,
                )
        finally:
            pass
        t3 = self._timing_mark(timing_device)
        predicted_clip_running_status_chunk = torch.sigmoid(self.clip_running_status_out_proj(status_suffix_out)).squeeze(-1)
        pooled_suffix_feature = status_suffix_out.mean(dim=1)
        task_id_logits = self.task_id_out_proj(pooled_suffix_feature)
        predicted_boxes = None
        if self.box_out_proj is not None:
            predicted_boxes = self.box_out_proj(pooled_suffix_feature).reshape(
                pooled_suffix_feature.shape[0], *self.box_aux_shape
            )
        predicted_cross_center = None
        if self.cross_center_out_proj is not None:
            predicted_cross_center = self.cross_center_out_proj(pooled_suffix_feature).reshape(
                pooled_suffix_feature.shape[0], *self.cross_center_aux_shape
            )
        t4 = self._timing_mark(timing_device)
        self.last_inference_timing_ms = {
            "vlm_embed_prefix": (t1 - t0) * 1000.0,
            "vlm_prefill": (t2 - t1) * 1000.0,
            "vlm_total": (t2 - t0) * 1000.0,
            "action_expert_denoise": (t3 - t2) * 1000.0,
            "status_head": (t4 - t3) * 1000.0,
            "model_total": (t4 - t0) * 1000.0,
            "denoise_steps": float(num_steps),
            "denoise_step_avg": ((t3 - t2) * 1000.0) / max(float(num_steps), 1.0),
        }
        if return_task_logits and return_aux_predictions:
            return actions, predicted_clip_running_status_chunk, task_id_logits, predicted_boxes, predicted_cross_center
        if return_task_logits:
            return actions, predicted_clip_running_status_chunk, task_id_logits
        if return_aux_predictions:
            return actions, predicted_clip_running_status_chunk, predicted_boxes, predicted_cross_center
        return actions, predicted_clip_running_status_chunk

    def denoise_step(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        past_key_values,
        x_t,
        timestep,
        state_cond: Tensor | None = None,
        return_suffix_out: bool = False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        x_t = self._zero_unused_action_dims(x_t)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep, state_cond=state_cond
        )

        if past_key_values is None:
            if prefix_embs is None or prefix_att_masks is None:
                raise ValueError("prefix_embs and prefix_att_masks are required when prefix cache is disabled")
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_2d_masks = self._build_full_att_2d_masks(
                prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
            )
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                prefix_layer_states=self._cached_prefix_layer_states,
            )
        else:
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_attention_mode = getattr(self.config, "suffix_attention_mask", "causal")
            suffix_att_2d_masks = make_att_2d_masks(
                suffix_pad_masks, suffix_att_masks, attention_mode=suffix_attention_mode
            )
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        action_out = self._project_action(suffix_out)
        if return_suffix_out:
            return action_out, suffix_out
        return action_out


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self._clip_status_curve_dump_idx = 0
        self._clip_status_curve_batch_idx = 0
        self._box_cross_debug_dump_idx = 0
        self._box_cross_debug_batch_idx = 0
        self._input_image_debug_dump_idx = 0
        self._input_image_debug_batch_idx = 0
        self._train_rollout_debug_counter = 0
        self._pending_train_rollout_debug_payload = None
        self._dataset_stats = kwargs.get("dataset_stats")
        self._warned_missing_ee_stats = False
        self._warned_missing_pose_stats = False

        self.reset()

    def _ee_loss_enabled(self) -> bool:
        return self.config.ee_position_loss_weight > 0 or self.config.ee_orientation_loss_weight > 0

    def _pose_loss_enabled(self) -> bool:
        return (
            getattr(self.config, "pose_translation_loss_weight", 0.0) > 0
            or getattr(self.config, "pose_rotation_loss_weight", 0.0) > 0
        )

    def _resolve_ee_joint_groups(self, action_dim: int) -> list[list[int]]:
        configured_groups = getattr(self.config, "ee_joint_groups", None) or []
        groups = [list(group) for group in configured_groups] if configured_groups else _default_ee_joint_groups(action_dim)
        valid_groups: list[list[int]] = []
        for group in groups:
            if len(group) != 6:
                logging.warning("[PI05][ee-loss] skip joint group %s because Piper FK expects 6 joints", group)
                continue
            if any(idx >= action_dim for idx in group):
                logging.warning("[PI05][ee-loss] skip joint group %s because action_dim=%d", group, action_dim)
                continue
            valid_groups.append(group)
        return valid_groups

    def _split_action_loss_tensors(self, losses: Tensor) -> tuple[Tensor, Tensor | None]:
        active_indices = self._active_action_indices(int(losses.shape[-1]))
        if not getattr(self.config, "enable_split_action_heads", False):
            return losses[..., active_indices], None

        joint_indices, gripper_indices = _resolve_action_head_indices(
            getattr(self.config, "gripper_action_indices", None),
            int(losses.shape[-1]),
        )
        active_set = set(active_indices)
        joint_indices = [index for index in joint_indices if index in active_set]
        gripper_indices = [index for index in gripper_indices if index in active_set]
        if not joint_indices:
            raise ValueError("action_loss_active_indices excludes every non-gripper action dimension")
        joint_losses = losses[..., joint_indices]
        gripper_losses = losses[..., gripper_indices] if gripper_indices else None
        return joint_losses, gripper_losses

    def _active_action_indices(self, action_dim: int) -> list[int]:
        configured = list(getattr(self.config, "action_loss_active_indices", []))
        if not configured:
            return list(range(int(action_dim)))
        invalid = [index for index in configured if index < 0 or index >= int(action_dim)]
        if invalid:
            raise ValueError(
                f"action_loss_active_indices {invalid} are outside action_dim={action_dim}"
            )
        return configured

    def _prepare_action_dim_valid_mask(self, actions: Tensor) -> Tensor | None:
        configured = list(getattr(self.config, "action_loss_active_indices", []))
        if not configured:
            return None
        mask = torch.zeros(actions.shape[-1], dtype=torch.bool, device=actions.device)
        mask[self._active_action_indices(int(actions.shape[-1]))] = True
        return mask

    @staticmethod
    def _masked_action_mean(
        losses: Tensor,
        action_valid_mask: Tensor | None,
        *,
        per_sample: bool,
    ) -> Tensor:
        if action_valid_mask is None:
            return losses.mean(dim=(1, 2)) if per_sample else losses.mean()

        if action_valid_mask.shape != losses.shape[:2]:
            raise ValueError(
                "action_valid_mask must have shape [batch, time]: "
                f"{tuple(action_valid_mask.shape)} != {tuple(losses.shape[:2])}"
            )
        weights = action_valid_mask.to(device=losses.device, dtype=losses.dtype).unsqueeze(-1)
        if per_sample:
            numerator = (losses * weights).sum(dim=(1, 2))
            denominator = weights.sum(dim=(1, 2)) * losses.shape[-1]
        else:
            numerator = (losses * weights).sum()
            denominator = weights.sum() * losses.shape[-1]
        return numerator / denominator.clamp_min(1.0)

    def _compute_action_loss(
        self, losses: Tensor, action_valid_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        joint_losses, gripper_losses = self._split_action_loss_tensors(losses)
        joint_action_loss = self._masked_action_mean(joint_losses, action_valid_mask, per_sample=False)
        if gripper_losses is None:
            return joint_action_loss, None, None

        gripper_action_loss = self._masked_action_mean(gripper_losses, action_valid_mask, per_sample=False)
        total_action_loss = (
            self.config.joint_action_loss_weight * joint_action_loss
            + self.config.gripper_action_loss_weight * gripper_action_loss
        )
        return total_action_loss, joint_action_loss, gripper_action_loss

    def _non_gripper_action_loss_name(self) -> str:
        if getattr(self.config, "action_space", "joint") == "kitt_se3_pose":
            return "pose_action_loss"
        return "joint_action_loss"

    def _compute_action_per_sample_loss(
        self, losses: Tensor, action_valid_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        joint_losses, gripper_losses = self._split_action_loss_tensors(losses)
        joint_action_per_sample_loss = self._masked_action_mean(
            joint_losses, action_valid_mask, per_sample=True
        )
        if gripper_losses is None:
            return joint_action_per_sample_loss, None, None

        gripper_action_per_sample_loss = self._masked_action_mean(
            gripper_losses, action_valid_mask, per_sample=True
        )
        action_per_sample_loss = (
            self.config.joint_action_loss_weight * joint_action_per_sample_loss
            + self.config.gripper_action_loss_weight * gripper_action_per_sample_loss
        )
        return action_per_sample_loss, joint_action_per_sample_loss, gripper_action_per_sample_loss

    def _compute_action_error_metrics(self, diff: Tensor) -> dict[str, float]:
        mse = diff.pow(2)
        mae = diff.abs()
        joint_mse, gripper_mse = self._split_action_loss_tensors(mse)
        joint_mae, gripper_mae = self._split_action_loss_tensors(mae)
        metrics = {
            "mse": float(mse.mean().item()),
            "mae": float(mae.mean().item()),
            "traj_mse": float(joint_mse.mean().item()),
            "traj_mae": float(joint_mae.mean().item()),
        }
        if gripper_mse is not None and gripper_mae is not None:
            metrics["gripper_mse"] = float(gripper_mse.mean().item())
            metrics["gripper_mae"] = float(gripper_mae.mean().item())
        if bool(getattr(self.config, "behavior_b1k_debug_metrics", False)) and int(diff.shape[-1]) >= 23:
            behavior_groups = {
                "base_xy": [0, 1],
                "base_yaw": [2],
                "torso": [3, 4, 5, 6],
                "left_arm": [7, 8, 9, 10, 11, 12, 13],
                "left_gripper": [14],
                "right_arm": [15, 16, 17, 18, 19, 20, 21],
                "right_gripper": [22],
            }
            for group_name, indices in behavior_groups.items():
                metrics[f"{group_name}_mse"] = float(mse[..., indices].mean().item())
                metrics[f"{group_name}_mae"] = float(mae[..., indices].mean().item())
        return metrics

    def _format_rollout_debug_metrics(self, prefix: str, metrics: dict[str, float] | None) -> str:
        metric_parts: list[str] = []
        base_keys = (
            "mse",
            "mae",
            "traj_mse",
            "traj_mae",
            "gripper_mse",
            "gripper_mae",
        )
        for key in base_keys:
            value = float("nan") if metrics is None else metrics.get(key, float("nan"))
            metric_parts.append(f"{prefix}_{key}={value:.6f}")

        if bool(getattr(self.config, "behavior_b1k_debug_metrics", False)):
            behavior_keys = (
                "base_xy_mse",
                "base_xy_mae",
                "base_yaw_mse",
                "base_yaw_mae",
                "torso_mse",
                "torso_mae",
                "left_arm_mse",
                "left_arm_mae",
                "left_gripper_mse",
                "left_gripper_mae",
                "right_arm_mse",
                "right_arm_mae",
                "right_gripper_mse",
                "right_gripper_mae",
            )
            for key in behavior_keys:
                value = float("nan") if metrics is None else metrics.get(key, float("nan"))
                metric_parts.append(f"{prefix}_{key}={value:.6f}")
        return " ".join(metric_parts)

    @staticmethod
    def _detach_train_rollout_debug_value(value):
        if isinstance(value, Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {
                key: PI05Policy._detach_train_rollout_debug_value(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, list):
            return [PI05Policy._detach_train_rollout_debug_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PI05Policy._detach_train_rollout_debug_value(item) for item in value)
        return value

    def _schedule_train_rollout_debug(
        self,
        *,
        images,
        img_masks,
        vggt_images: list[Tensor] | None,
        tokens,
        masks,
        batch: dict[str, Tensor],
        state_cond: Tensor | None,
        state_history: Tensor | None,
        flow_source_state_history: Tensor | None,
        flow_source: Tensor,
        predicted_action: Tensor,
        target_action: Tensor,
        original_action_dim: int,
    ) -> None:
        debug_interval = int(getattr(self.config, "train_rollout_debug_interval", 0))
        if debug_interval <= 0:
            self._pending_train_rollout_debug_payload = None
            return

        self._train_rollout_debug_counter += 1
        if self._train_rollout_debug_counter % debug_interval != 0:
            self._pending_train_rollout_debug_payload = None
            return

        self._pending_train_rollout_debug_payload = {
            "images": self._detach_train_rollout_debug_value(images),
            "img_masks": self._detach_train_rollout_debug_value(img_masks),
            "vggt_images": self._detach_train_rollout_debug_value(vggt_images),
            "tokens": self._detach_train_rollout_debug_value(tokens),
            "masks": self._detach_train_rollout_debug_value(masks),
            "batch": self._detach_train_rollout_debug_value(batch),
            "state_cond": self._detach_train_rollout_debug_value(state_cond),
            "state_history": self._detach_train_rollout_debug_value(state_history),
            "flow_source_state_history": self._detach_train_rollout_debug_value(flow_source_state_history),
            "flow_source": self._detach_train_rollout_debug_value(flow_source),
            "predicted_action": self._detach_train_rollout_debug_value(predicted_action),
            "target_action": self._detach_train_rollout_debug_value(target_action),
            "original_action_dim": int(original_action_dim),
        }

    @torch.no_grad()
    def run_pending_train_rollout_debug(self) -> None:
        payload = self._pending_train_rollout_debug_payload
        if payload is None:
            return
        self._pending_train_rollout_debug_payload = None
        self._maybe_log_train_rollout_debug(**payload)

    def _maybe_log_train_rollout_debug(
        self,
        *,
        images,
        img_masks,
        vggt_images: list[Tensor] | None,
        tokens,
        masks,
        batch: dict[str, Tensor],
        state_cond: Tensor | None,
        state_history: Tensor | None,
        flow_source_state_history: Tensor | None,
        flow_source: Tensor,
        predicted_action: Tensor,
        target_action: Tensor,
        original_action_dim: int,
    ) -> None:

        rollout_steps_list = (1, 5, 10)
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            target_action_raw = self._recover_action_for_ee(target_action, batch, original_action_dim)
            if target_action_raw is None:
                logging.info(
                    "[PI05][rollout-debug] step=%d skip rollout metrics because raw action recovery is unavailable",
                    self._train_rollout_debug_counter,
                )
                return

            metric_parts: list[str] = []
            rollout_clip_running_status = self._prepare_clip_running_status(batch)
            # Rebuild rollout conditions from the batch so debug sampling matches the real inference path.
            rollout_state_cond = self._prepare_state_for_denoise_condition(batch, tokens.device, torch.float32)
            if rollout_state_cond is None:
                rollout_state_cond = state_cond
            rollout_state_history = self._prepare_state_history_for_prefix(batch, tokens.device, torch.float32)
            if rollout_state_history is None:
                rollout_state_history = state_history
            rollout_flow_source_state_history = self._prepare_state_history_for_flow_source(
                batch, tokens.device, torch.float32
            )
            if rollout_flow_source_state_history is None:
                rollout_flow_source_state_history = flow_source_state_history
            train_step1_action = predicted_action[:, :, :original_action_dim]
            train_step1_action_raw = self._recover_action_for_ee(train_step1_action, batch, original_action_dim)
            if train_step1_action_raw is None:
                metric_parts.append(self._format_rollout_debug_metrics("train_direct_step1", None))
            else:
                train_step1_metrics = self._compute_action_error_metrics(train_step1_action_raw - target_action_raw)
                metric_parts.append(self._format_rollout_debug_metrics("train_direct_step1", train_step1_metrics))
            for rollout_steps in rollout_steps_list:
                raw_texts = self._extract_raw_texts_from_batch(batch)
                rollout_action = self.model.sample_actions(
                    images,
                    img_masks,
                    self._prepare_image_pad_masks(batch),
                    tokens,
                    masks,
                    vggt_images=vggt_images,
                    raw_texts=raw_texts,
                    clip_running_status=rollout_clip_running_status,
                    state=rollout_state_cond,
                    state_history=rollout_state_history,
                    flow_source_state_history=rollout_flow_source_state_history,
                    noise=flow_source,
                    num_steps=rollout_steps,
                )
                rollout_action = rollout_action[:, :, :original_action_dim]
                rollout_action_raw = self._recover_action_for_ee(rollout_action, batch, original_action_dim)
                if rollout_action_raw is None:
                    metric_parts.append(self._format_rollout_debug_metrics(f"rollout_{rollout_steps}step", None))
                    continue
                rollout_metrics = self._compute_action_error_metrics(rollout_action_raw - target_action_raw)
                rollout_metric_str = self._format_rollout_debug_metrics(f"rollout_{rollout_steps}step", rollout_metrics)
                if rollout_steps == 1 and train_step1_action_raw is not None:
                    train_vs_rollout_metrics = self._compute_action_error_metrics(train_step1_action_raw - rollout_action_raw)
                    rollout_metric_str = (
                        f"{rollout_metric_str} "
                        f"{self._format_rollout_debug_metrics('train_vs_rollout1', train_vs_rollout_metrics)}"
                    )
                metric_parts.append(rollout_metric_str)
            logging.info(
                "[PI05][rollout-debug] step=%d %s",
                self._train_rollout_debug_counter,
                " ".join(metric_parts),
            )
        finally:
            if was_training:
                self.model.train()

    @staticmethod
    def _repeat_tensor_for_group_consistency(value: Tensor | None, group_size: int) -> Tensor | None:
        if value is None:
            return None
        return value.repeat_interleave(group_size, dim=0)

    def _repeat_batch_for_group_consistency(self, batch: dict[str, Tensor], group_size: int) -> dict[str, Tensor]:
        repeated_batch: dict[str, Tensor] = {}
        batch_size = None
        for value in batch.values():
            if isinstance(value, Tensor) and value.ndim > 0:
                batch_size = int(value.shape[0])
                break
        if batch_size is None:
            return dict(batch)

        for key, value in batch.items():
            if isinstance(value, Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
                repeated_batch[key] = value.repeat_interleave(group_size, dim=0)
            elif isinstance(value, list) and len(value) == batch_size and all(isinstance(item, str) for item in value):
                repeated_batch[key] = [item for item in value for _ in range(group_size)]
            else:
                repeated_batch[key] = value
        return repeated_batch

    def _maybe_expand_group_consistency_inputs(
        self,
        *,
        batch: dict[str, Tensor],
        images: list[Tensor],
        img_masks: list[Tensor],
        img_pad_masks: list[Tensor | None],
    ) -> tuple[dict[str, Tensor], list[Tensor], list[Tensor], list[Tensor | None], dict[str, int] | None]:
        if not getattr(self.config, "enable_group_consistency_loss", False):
            return batch, images, img_masks, img_pad_masks, None

        group_camera_keys = list(getattr(self.config, "group_consistency_camera_keys", []))
        if len(group_camera_keys) < 2:
            return batch, images, img_masks, img_pad_masks, None

        group_size = len(group_camera_keys)
        batch_size = int(images[0].shape[0]) if images else 0
        if batch_size == 0:
            return batch, images, img_masks, img_pad_masks, None

        anchor_key = getattr(self.config, "group_consistency_anchor_camera_key", None) or group_camera_keys[0]
        try:
            anchor_index = group_camera_keys.index(anchor_key)
        except ValueError as exc:
            raise ValueError(
                f"group_consistency_anchor_camera_key={anchor_key!r} is not in "
                f"group_consistency_camera_keys={group_camera_keys}"
            ) from exc

        expanded_images = [torch.stack(images, dim=1).reshape(batch_size * group_size, *images[0].shape[1:])]
        expanded_img_masks = [
            torch.stack(img_masks, dim=1).reshape(batch_size * group_size, *img_masks[0].shape[1:])
        ]
        if img_pad_masks and all(isinstance(pad_mask, torch.Tensor) for pad_mask in img_pad_masks):
            stacked_pad_masks = torch.stack(img_pad_masks, dim=1)
            expanded_img_pad_masks = [stacked_pad_masks.reshape(batch_size * group_size, *stacked_pad_masks.shape[2:])]
        else:
            expanded_img_pad_masks = [None]
        expanded_batch = self._repeat_batch_for_group_consistency(batch, group_size)
        consistency_meta = {
            "base_batch_size": batch_size,
            "group_size": group_size,
            "anchor_index": anchor_index,
        }
        return expanded_batch, expanded_images, expanded_img_masks, expanded_img_pad_masks, consistency_meta

    def _compute_group_consistency_loss(
        self,
        projected_latent: Tensor | None,
        consistency_meta: dict[str, int] | None,
    ) -> tuple[Tensor | None, Tensor | None, float | None]:
        if projected_latent is None or consistency_meta is None:
            return None, None, None

        group_size = int(consistency_meta["group_size"])
        anchor_index = int(consistency_meta["anchor_index"])
        if group_size < 2:
            return None, None, None
        if projected_latent.shape[0] % group_size != 0:
            raise ValueError(
                f"projected_latent batch size {projected_latent.shape[0]} is not divisible by group_size={group_size}"
            )

        latent = projected_latent.reshape(-1, group_size, projected_latent.shape[-1])
        anchor = latent[:, anchor_index : anchor_index + 1, :]
        other_indices = [idx for idx in range(group_size) if idx != anchor_index]
        if not other_indices:
            return None, None, None
        others = latent[:, other_indices, :]
        anchor = F.normalize(anchor, dim=-1)
        others = F.normalize(others, dim=-1)
        cosine = (others * anchor.expand_as(others)).sum(dim=-1)
        per_base_sample_loss = 1.0 - cosine.mean(dim=-1)
        per_sample_loss = per_base_sample_loss.repeat_interleave(group_size, dim=0)
        return per_base_sample_loss.mean(), per_sample_loss, float(cosine.mean().item())

    def _unnormalize_action_for_ee(self, action: Tensor) -> Tensor | None:
        if self._dataset_stats is None or ACTION not in self._dataset_stats:
            if self._ee_loss_enabled() and not self._warned_missing_ee_stats:
                logging.warning("[PI05][ee-loss] dataset stats are unavailable, EE losses are skipped")
                self._warned_missing_ee_stats = True
            if self._pose_loss_enabled() and not self._warned_missing_pose_stats:
                logging.warning("[PI05][pose-loss] dataset stats are unavailable, pose losses are skipped")
                self._warned_missing_pose_stats = True
            return None

        norm_mode = self.config.normalization_mapping.get("ACTION", NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY:
            return action

        stats = self._dataset_stats
        device = action.device
        dtype = action.dtype
        eps = 1e-8

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = _to_tensor_stat(stats, ACTION, "mean", device=device, dtype=dtype)
            std = _to_tensor_stat(stats, ACTION, "std", device=device, dtype=dtype)
            if mean is None or std is None:
                return None
            return action * std + mean

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = _to_tensor_stat(stats, ACTION, "min", device=device, dtype=dtype)
            max_val = _to_tensor_stat(stats, ACTION, "max", device=device, dtype=dtype)
            if min_val is None or max_val is None:
                return None
            denom = torch.where(
                (max_val - min_val) == 0,
                torch.full_like(max_val, eps),
                max_val - min_val,
            )
            return (action + 1.0) / 2.0 * denom + min_val

        if norm_mode == NormalizationMode.QUANTILES:
            q01 = _to_tensor_stat(stats, ACTION, "q01", device=device, dtype=dtype)
            q99 = _to_tensor_stat(stats, ACTION, "q99", device=device, dtype=dtype)
            if q01 is None or q99 is None:
                return None
            denom = torch.where((q99 - q01) == 0, torch.full_like(q99, eps), q99 - q01)
            return (action + 1.0) * denom / 2.0 + q01

        if norm_mode == NormalizationMode.QUANTILE10:
            q10 = _to_tensor_stat(stats, ACTION, "q10", device=device, dtype=dtype)
            q90 = _to_tensor_stat(stats, ACTION, "q90", device=device, dtype=dtype)
            if q10 is None or q90 is None:
                return None
            denom = torch.where((q90 - q10) == 0, torch.full_like(q90, eps), q90 - q10)
            return (action + 1.0) * denom / 2.0 + q10

        return None

    def _recover_action_for_ee(
        self,
        action: Tensor,
        batch: dict[str, Tensor],
        action_dim: int,
    ) -> Tensor | None:
        action_for_ee = action[..., :action_dim]
        if self.config.action_target_mode == "delta_from_state":
            state_for_action = self._prepare_state_for_action(batch, action.device, action.dtype)
            state_for_action = state_for_action.expand(-1, action.shape[1], -1)[:, :, :action_dim]
            # Training targets live in normalized space; recover absolute normalized actions first,
            # then unnormalize with ACTION stats before feeding FK.
            action_for_ee = action_for_ee + state_for_action
        return self._unnormalize_action_for_ee(action_for_ee.float())

    def _compute_ee_pose_losses(
        self,
        predicted_action: Tensor,
        target_action: Tensor,
        batch: dict[str, Tensor],
        action_dim: int,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if not self._ee_loss_enabled():
            return None, None, None, None

        predicted_action_raw = self._recover_action_for_ee(predicted_action, batch, action_dim)
        target_action_raw = self._recover_action_for_ee(target_action, batch, action_dim)
        if predicted_action_raw is None or target_action_raw is None:
            return None, None, None, None

        joint_groups = self._resolve_ee_joint_groups(action_dim)
        if not joint_groups:
            return None, None, None, None

        position_losses = []
        orientation_losses = []
        for group in joint_groups:
            pred_joint = predicted_action_raw[..., group]
            tgt_joint = target_action_raw[..., group]
            if self.config.ee_kinematics_type != "piper":
                raise ValueError(f"Unsupported ee_kinematics_type: {self.config.ee_kinematics_type}")
            pred_pos, pred_rot = _piper_forward_kinematics_torch(pred_joint, self.config.ee_joint_unit_scale)
            tgt_pos, tgt_rot = _piper_forward_kinematics_torch(tgt_joint, self.config.ee_joint_unit_scale)
            position_losses.append((pred_pos - tgt_pos).pow(2).mean(dim=-1))
            orientation_losses.append(_rotation_geodesic_distance(pred_rot, tgt_rot).pow(2))

        position_loss_tensor = torch.stack(position_losses, dim=-1).mean(dim=-1)
        orientation_loss_tensor = torch.stack(orientation_losses, dim=-1).mean(dim=-1)
        return (
            position_loss_tensor.mean(),
            position_loss_tensor.mean(dim=1),
            orientation_loss_tensor.mean(),
            orientation_loss_tensor.mean(dim=1),
        )

    @staticmethod
    def _masked_pose_mean(
        values: Tensor,
        action_valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if action_valid_mask is None:
            return values.mean(), values.mean(dim=1)
        if action_valid_mask.shape != values.shape:
            raise ValueError(
                "action_valid_mask must match pose loss [batch,time]: "
                f"{tuple(action_valid_mask.shape)} != {tuple(values.shape)}"
            )
        weights = action_valid_mask.to(device=values.device, dtype=values.dtype)
        per_sample = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        scalar = (values * weights).sum() / weights.sum().clamp_min(1.0)
        return scalar, per_sample

    def _compute_pose_action_losses(
        self,
        predicted_action: Tensor,
        target_action: Tensor,
        action_valid_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Physical translation L2 and SO(3) geodesic losses for pose20."""

        if not self._pose_loss_enabled():
            return None, None, None, None
        if predicted_action.shape[-1] != 20 or target_action.shape[-1] != 20:
            raise ValueError(
                "pose semantic losses require 20D pose actions, got "
                f"{predicted_action.shape[-1]} and {target_action.shape[-1]}"
            )
        predicted_raw = self._unnormalize_action_for_ee(predicted_action.float())
        target_raw = self._unnormalize_action_for_ee(target_action.float())
        if predicted_raw is None or target_raw is None:
            return None, None, None, None

        side_slices = {
            "left": (slice(0, 3), slice(3, 9)),
            "right": (slice(10, 13), slice(13, 19)),
        }
        translation_values: list[Tensor] = []
        rotation_values: list[Tensor] = []
        for side in getattr(self.config, "pose_loss_active_sides", ["left"]):
            translation_slice, rotation_slice = side_slices[side]
            translation_values.append(
                torch.linalg.vector_norm(
                    predicted_raw[..., translation_slice] - target_raw[..., translation_slice],
                    dim=-1,
                )
            )
            predicted_rotation = _rotation_6d_to_matrix(predicted_raw[..., rotation_slice])
            target_rotation = _rotation_6d_to_matrix(target_raw[..., rotation_slice])
            rotation_values.append(_rotation_geodesic_angle(predicted_rotation, target_rotation))

        translation_error = torch.stack(translation_values, dim=-1).mean(dim=-1)
        rotation_error = torch.stack(rotation_values, dim=-1).mean(dim=-1)
        translation_loss, translation_per_sample = self._masked_pose_mean(
            translation_error, action_valid_mask
        )
        rotation_loss, rotation_per_sample = self._masked_pose_mean(
            rotation_error, action_valid_mask
        )
        return translation_loss, translation_per_sample, rotation_loss, rotation_per_sample

    def _add_ee_readable_metrics(
        self,
        loss_dict: dict,
        ee_position_loss: Tensor | None,
        ee_orientation_loss: Tensor | None,
    ) -> None:
        if ee_position_loss is not None:
            # `ee_position_loss` is mean squared xyz error in mm^2; convert to Euclidean mm error for logging.
            ee_position_error_mm = torch.sqrt(torch.clamp(3.0 * ee_position_loss.detach().float(), min=0.0))
            loss_dict["ee_position_error_mm"] = ee_position_error_mm.item()
        if ee_orientation_loss is not None:
            # `ee_orientation_loss` is squared geodesic angle in rad^2; convert to degrees for readability.
            ee_orientation_error_deg = (
                torch.sqrt(torch.clamp(ee_orientation_loss.detach().float(), min=0.0)) * (180.0 / torch.pi)
            )
            loss_dict["ee_orientation_error_deg"] = ee_orientation_error_deg.item()

    def _save_clip_status_curve_png(self, pred_curve: Tensor, gt_curve: Tensor, out_path: Path) -> None:
        from PIL import Image, ImageDraw, ImageFont

        pred = pred_curve.detach().float().cpu().numpy().reshape(-1)
        gt = gt_curve.detach().float().cpu().numpy().reshape(-1)
        t_n = int(min(pred.shape[0], gt.shape[0]))
        if t_n <= 0:
            return
        pred = pred[:t_n]
        gt = gt[:t_n]

        w = 1200
        h = 360
        margin_l = 70
        margin_r = 30
        margin_t = 50
        margin_b = 45
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        draw.text((12, 12), "clip_running_status curve", fill=(0, 0, 0), font=font)
        draw.text((70, 12), "GT", fill=(0, 0, 0), font=font)
        draw.text((95, 12), "Pred", fill=(220, 0, 0), font=font)

        x0 = margin_l
        y0 = margin_t
        x1 = margin_l + plot_w
        y1 = margin_t + plot_h
        draw.rectangle([x0, y0, x1, y1], outline=(220, 220, 220), width=1)

        for tick, val in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            yy = int(y1 - val * plot_h)
            color = (240, 240, 240) if tick not in (0, 4) else (220, 220, 220)
            draw.line([(x0, yy), (x1, yy)], fill=color, width=1)
            draw.text((8, yy - 6), f"{val:.2f}", fill=(90, 90, 90), font=font)

        def _series_to_points(vals):
            if t_n == 1:
                xs = torch.tensor([x0 + plot_w // 2], dtype=torch.float32).numpy()
            else:
                xs = x0 + (torch.arange(t_n, dtype=torch.float32).numpy() / float(t_n - 1)) * float(plot_w)
            ys = y1 - torch.clamp(torch.from_numpy(vals), 0.0, 1.0).numpy() * float(plot_h)
            return [(int(x), int(y)) for x, y in zip(xs, ys)]

        gt_pts = _series_to_points(gt)
        pred_pts = _series_to_points(pred)
        if len(gt_pts) >= 2:
            draw.line(gt_pts, fill=(0, 0, 0), width=2)
        if len(pred_pts) >= 2:
            draw.line(pred_pts, fill=(220, 0, 0), width=2)

        draw.text((x0, y1 + 12), "t=0", fill=(90, 90, 90), font=font)
        draw.text((x1 - 65, y1 + 12), f"t={t_n - 1}", fill=(90, 90, 90), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    def _save_action_curve_png(self, pred_chunk: Tensor, gt_chunk: Tensor, out_path: Path) -> None:
        from PIL import Image, ImageDraw, ImageFont

        pred = pred_chunk.detach().float().cpu()
        gt = gt_chunk.detach().float().cpu()
        if pred.ndim == 1:
            pred = pred[:, None]
        if gt.ndim == 1:
            gt = gt[:, None]
        if pred.ndim != 2 or gt.ndim != 2:
            pred = pred.reshape(pred.shape[0], -1)
            gt = gt.reshape(gt.shape[0], -1)

        t_n = int(min(pred.shape[0], gt.shape[0]))
        d_n = int(min(pred.shape[1], gt.shape[1]))
        if t_n <= 0 or d_n <= 0:
            return
        pred = pred[:t_n, :d_n]
        gt = gt[:t_n, :d_n]

        cols = min(4, d_n)
        rows = int(math.ceil(d_n / cols))
        cell_w = 290
        cell_h = 170
        pad = 18
        title_h = 36
        w = cols * cell_w + (cols + 1) * pad
        h = rows * cell_h + (rows + 1) * pad + title_h

        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((pad, 10), "action curve (GT vs Pred)", fill=(0, 0, 0), font=font)

        def _series_to_points(vals: Tensor, px0: int, py0: int, pw: int, ph: int, vmin: float, vmax: float):
            if t_n == 1:
                xs = torch.tensor([px0 + pw // 2], dtype=torch.float32)
            else:
                xs = px0 + (torch.arange(t_n, dtype=torch.float32) / float(t_n - 1)) * float(pw)
            denom = max(vmax - vmin, 1e-6)
            ys = py0 + ph - ((vals - vmin) / denom).clamp(0.0, 1.0) * float(ph)
            return [(int(x.item()), int(y.item())) for x, y in zip(xs, ys)]

        for d in range(d_n):
            r = d // cols
            c = d % cols
            x = pad + c * (cell_w + pad)
            y = title_h + pad + r * (cell_h + pad)
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(210, 210, 210), width=1)
            draw.text((x + 8, y + 6), f"dim {d}", fill=(0, 0, 0), font=font)

            px0 = x + 8
            py0 = y + 26
            pw = cell_w - 16
            ph = cell_h - 36
            draw.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(230, 230, 230), width=1)

            gt_v = gt[:, d]
            pred_v = pred[:, d]
            vmin = float(torch.min(torch.stack([gt_v.min(), pred_v.min()])).item())
            vmax = float(torch.max(torch.stack([gt_v.max(), pred_v.max()])).item())
            if not (math.isfinite(vmin) and math.isfinite(vmax)):
                vmin = -1.0
                vmax = 1.0
            if abs(vmax - vmin) < 1e-6:
                delta = 1.0 if abs(vmax) < 1.0 else abs(vmax) * 0.2
                vmin = vmin - delta
                vmax = vmax + delta

            gt_pts = _series_to_points(gt_v, px0, py0, pw, ph, vmin, vmax)
            pred_pts = _series_to_points(pred_v, px0, py0, pw, ph, vmin, vmax)
            if len(gt_pts) >= 2:
                draw.line(gt_pts, fill=(0, 0, 0), width=2)
            elif len(gt_pts) == 1:
                x1, y1 = gt_pts[0]
                draw.ellipse([x1 - 2, y1 - 2, x1 + 2, y1 + 2], fill=(0, 0, 0))
            if len(pred_pts) >= 2:
                draw.line(pred_pts, fill=(220, 0, 0), width=2)
            elif len(pred_pts) == 1:
                x1, y1 = pred_pts[0]
                draw.ellipse([x1 - 2, y1 - 2, x1 + 2, y1 + 2], fill=(220, 0, 0))
            draw.text((px0, py0 + ph - 12), "t=0", fill=(90, 90, 90), font=font)
            draw.text((px0 + pw - 45, py0 + ph - 12), f"t={t_n - 1}", fill=(90, 90, 90), font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    def _maybe_dump_clip_status_curves(
        self,
        pred_status: Tensor | None,
        tgt_status: Tensor | None,
        pred_action: Tensor | None,
        tgt_action: Tensor | None,
        batch: dict[str, Tensor],
    ) -> None:
        if not self.training:
            return
        dump_dir = self.config.clip_running_status_curve_dump_dir
        if dump_dir is None or str(dump_dir).strip() == "":
            return
        root = Path(dump_dir)
        pid = os.getpid()
        batch_idx = self._clip_status_curve_batch_idx
        self._clip_status_curve_batch_idx += 1
        step_dir = root / f"step_{batch_idx:08d}_p{pid}"
        bsz_candidates: list[int] = []
        if pred_status is not None and tgt_status is not None:
            bsz_candidates.append(int(min(pred_status.shape[0], tgt_status.shape[0])))
        if pred_action is not None and tgt_action is not None:
            bsz_candidates.append(int(min(pred_action.shape[0], tgt_action.shape[0])))
        if not bsz_candidates:
            return
        bsz = int(min(bsz_candidates))
        for bi in range(bsz):
            idx = self._clip_status_curve_dump_idx
            self._clip_status_curve_dump_idx += 1
            suffix = f"p{pid}_n{idx:08d}_b{bi:02d}"
            if "index" in batch:
                try:
                    sample_idx = int(batch["index"][bi].item())
                    suffix = f"{suffix}_idx{sample_idx:09d}"
                except Exception:
                    pass
            stem = f"{time.time_ns()}_{suffix}"
            if pred_status is not None and tgt_status is not None:
                status_out_path = step_dir / f"{stem}_clip_running_status_curve.png"
                self._save_clip_status_curve_png(pred_status[bi], tgt_status[bi], status_out_path)
            if pred_action is not None and tgt_action is not None:
                action_out_path = step_dir / f"{stem}_action_curve.png"
                self._save_action_curve_png(pred_action[bi], tgt_action[bi], action_out_path)


    @staticmethod
    def _extract_visual_debug_image(image_tensor: Tensor) -> Tensor:
        image = image_tensor.detach().float().cpu()
        while image.ndim > 3:
            image = image[-1]
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims after squeeze, got shape {tuple(image.shape)}")
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.clamp(0.0, 1.0)
        return image

    @classmethod
    def _extract_visual_debug_frames(cls, image_tensor: Tensor) -> list[Tensor]:
        image = image_tensor.detach().float().cpu()
        while image.ndim > 4:
            image = image[-1]
        if image.ndim == 3:
            return [cls._extract_visual_debug_image(image)]
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(image.shape)}")

        frames = []
        for frame_idx in range(image.shape[0]):
            frames.append(cls._extract_visual_debug_image(image[frame_idx]))
        return frames

    @staticmethod
    def _normalize_pred_coords_for_draw(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().float().cpu().clamp(0.0, 1.0)

    @staticmethod
    def _normalize_box_coords_for_draw(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        boxes = tensor.detach().float().cpu().clone()
        boxes = boxes.clamp(0.0, 1.0)
        if boxes.shape[-1] != 4:
            return boxes
        x_min = torch.minimum(boxes[..., 0], boxes[..., 2])
        y_min = torch.minimum(boxes[..., 1], boxes[..., 3])
        x_max = torch.maximum(boxes[..., 0], boxes[..., 2])
        y_max = torch.maximum(boxes[..., 1], boxes[..., 3])
        boxes[..., 0] = x_min
        boxes[..., 1] = y_min
        boxes[..., 2] = x_max
        boxes[..., 3] = y_max
        return boxes

    @staticmethod
    def _draw_box(draw, box: Tensor, width: int, height: int, color: tuple[int, int, int], label: str, y_offset: int):
        x1 = int(float(box[0].item()) * width)
        y1 = int(float(box[1].item()) * height)
        x2 = int(float(box[2].item()) * width)
        y2 = int(float(box[3].item()) * height)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((max(4, x1 + 3), max(4, y1 + y_offset)), label, fill=color)

    @staticmethod
    def _draw_point(draw, point: Tensor, width: int, height: int, color: tuple[int, int, int], label: str):
        cx = int(float(point[0].item()) * width)
        cy = int(float(point[1].item()) * height)
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        draw.line([(cx - 8, cy), (cx + 8, cy)], fill=color, width=2)
        draw.line([(cx, cy - 8), (cx, cy + 8)], fill=color, width=2)
        draw.text((min(width - 80, cx + 8), max(4, cy - 10)), label, fill=color)

    def _save_box_cross_debug_png(
        self,
        image_tensor: Tensor,
        out_path: Path,
        *,
        camera_key: str,
        predicted_boxes: Tensor | None,
        target_boxes: Tensor | None,
        box_valid_mask: Tensor | None,
        predicted_cross_center: Tensor | None,
        target_cross_center: Tensor | None,
        cross_center_valid: bool,
        sample_idx: int | None,
        dataset_index: int | None,
    ) -> None:
        from PIL import Image, ImageDraw, ImageFont

        image = self._extract_visual_debug_image(image_tensor)
        image_np = (image.numpy() * 255.0).round().astype("uint8")
        img = Image.fromarray(image_np)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        width, height = img.size

        header_lines = [
            f"camera={camera_key}",
            f"sample_idx={sample_idx}" if sample_idx is not None else "sample_idx=NA",
            f"dataset_index={dataset_index}" if dataset_index is not None else "dataset_index=NA",
        ]
        draw.rectangle([0, 0, width - 1, min(height - 1, 36 + 14 * len(header_lines))], fill=(255, 255, 255))
        for line_idx, text in enumerate(header_lines):
            draw.text((8, 8 + line_idx * 12), text, fill=(0, 0, 0), font=font)

        if target_boxes is not None and box_valid_mask is not None:
            for box_idx, is_valid in enumerate(box_valid_mask.tolist()):
                if not is_valid:
                    continue
                self._draw_box(
                    draw,
                    target_boxes[box_idx],
                    width,
                    height,
                    (0, 200, 0),
                    f"gt_box_{box_idx}",
                    y_offset=0,
                )
        if predicted_boxes is not None:
            for box_idx in range(predicted_boxes.shape[0]):
                self._draw_box(
                    draw,
                    predicted_boxes[box_idx],
                    width,
                    height,
                    (220, 0, 0),
                    f"pred_box_{box_idx}",
                    y_offset=12,
                )

        if target_cross_center is not None and cross_center_valid:
            self._draw_point(draw, target_cross_center, width, height, (0, 200, 0), "gt_center")
        if predicted_cross_center is not None:
            self._draw_point(draw, predicted_cross_center, width, height, (220, 0, 0), "pred_center")

        legend_y = max(4, height - 36)
        draw.rectangle([0, legend_y - 4, min(width - 1, 220), height - 1], fill=(255, 255, 255))
        draw.text((8, legend_y), "green=GT  red=Pred", fill=(0, 0, 0), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    def _save_input_image_debug_png(
        self,
        image_tensors: list[Tensor],
        out_path: Path,
        *,
        camera_keys: list[str],
        pad_masks: list[Tensor | None] | None,
        frame_source_texts: list[list[str] | None] | None,
        sample_idx: int | None,
        dataset_index: int | None,
        episode_index_text: str | None,
        frame_index_text: str | None,
        timestamp_text: str | None,
    ) -> None:
        from PIL import Image, ImageDraw, ImageFont

        image_sequences = [self._extract_visual_debug_frames(image_tensor) for image_tensor in image_tensors]
        if not image_sequences:
            return

        max_frames = max(len(sequence) for sequence in image_sequences)
        first_frame = image_sequences[0][0]
        frame_height, frame_width = int(first_frame.shape[0]), int(first_frame.shape[1])
        row_label_width = 220
        frame_label_height = 52
        header_height = 40
        canvas_width = row_label_width + max_frames * frame_width
        canvas_height = header_height + len(image_sequences) * (frame_height + frame_label_height)

        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        header_lines = [
            f"sample_idx={sample_idx}" if sample_idx is not None else "sample_idx=NA",
            f"dataset_index={dataset_index}" if dataset_index is not None else "dataset_index=NA",
            f"episode_index={episode_index_text}" if episode_index_text else "episode_index=NA",
            f"frame_index={frame_index_text}" if frame_index_text else "frame_index=NA",
            f"timestamp={timestamp_text}" if timestamp_text else "timestamp=NA",
        ]
        for line_idx, text in enumerate(header_lines):
            draw.text((8, 6 + line_idx * 12), text, fill=(0, 0, 0), font=font)

        for row_idx, (image_sequence, camera_key) in enumerate(zip(image_sequences, camera_keys, strict=True)):
            y_offset = header_height + row_idx * (frame_height + frame_label_height)
            draw.rectangle(
                [0, y_offset, row_label_width - 1, y_offset + frame_height + frame_label_height - 1],
                fill=(245, 245, 245),
            )
            draw.text((8, y_offset + 6), camera_key, fill=(0, 0, 0), font=font)
            draw.text((8, y_offset + 20), f"frames={len(image_sequence)}", fill=(80, 80, 80), font=font)

            pad_mask = None if pad_masks is None else pad_masks[row_idx]
            pad_list = None
            if isinstance(pad_mask, torch.Tensor):
                pad_list = pad_mask.detach().bool().cpu().tolist()
            source_text_list = None if frame_source_texts is None else frame_source_texts[row_idx]

            for frame_idx in range(max_frames):
                x_offset = row_label_width + frame_idx * frame_width
                tile_bottom = y_offset + frame_height
                draw.rectangle(
                    [x_offset, y_offset, x_offset + frame_width - 1, tile_bottom - 1],
                    fill=(235, 235, 235),
                    outline=(180, 180, 180),
                )
                if frame_idx < len(image_sequence):
                    image = image_sequence[frame_idx]
                    image_array = (image.numpy() * 255.0).round().astype("uint8")
                    pil_image = Image.fromarray(image_array)
                    canvas.paste(pil_image, (x_offset, y_offset))
                    is_pad = bool(pad_list[frame_idx]) if pad_list is not None and frame_idx < len(pad_list) else False
                    if is_pad:
                        draw.rectangle(
                            [x_offset, y_offset, x_offset + frame_width - 1, tile_bottom - 1],
                            fill=(255, 220, 220),
                            outline=(200, 0, 0),
                            width=2,
                        )
                        draw.text((x_offset + 6, y_offset + 6), "pad", fill=(200, 0, 0), font=font)
                draw.text(
                    (x_offset + 6, tile_bottom + 2),
                    f"t{frame_idx}",
                    fill=(0, 0, 0),
                    font=font,
                )
                if source_text_list is not None and frame_idx < len(source_text_list):
                    source_text = source_text_list[frame_idx]
                    for line_idx, text in enumerate(source_text.split("\n")[:3]):
                        draw.text(
                            (x_offset + 6, tile_bottom + 14 + line_idx * 12),
                            text[:30],
                            fill=(70, 70, 70),
                            font=font,
                        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)

    @staticmethod
    def _collect_frame_source_debug_payload(
        batch: dict[str, Tensor], camera_keys: list[str], batch_index: int
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        for camera_key in camera_keys:
            debug_key = f"{camera_key}_debug_frame_sources"
            if debug_key not in batch:
                payload[camera_key] = None
                continue
            try:
                value = batch[debug_key][batch_index]
                if isinstance(value, str) and value.strip():
                    payload[camera_key] = json.loads(value)
                else:
                    payload[camera_key] = value
            except Exception as exc:
                payload[camera_key] = {"error": str(exc)}
        return payload

    @staticmethod
    def _format_debug_batch_value(batch: dict[str, Tensor], key: str, batch_index: int) -> tuple[str | None, str | None]:
        if key not in batch:
            return None, None
        value = batch[key]
        try:
            if isinstance(value, Tensor):
                sample_value = value[batch_index]
                if sample_value.ndim == 0:
                    scalar = sample_value.item()
                    if isinstance(scalar, float):
                        return f"{scalar:.6f}", f"{scalar:.6f}".replace(".", "p").replace("-", "m")
                    return str(int(scalar)), str(int(scalar))

                flat_values = sample_value.reshape(-1).tolist()
                if len(flat_values) == 0:
                    return None, None
                if any(isinstance(v, float) for v in flat_values):
                    text = "[" + ",".join(f"{float(v):.6f}" for v in flat_values) + "]"
                    safe = "seq" + "-".join(f"{float(v):.6f}".replace(".", "p").replace("-", "m") for v in flat_values[:8])
                    if len(flat_values) > 8:
                        safe = f"{safe}-n{len(flat_values)}"
                    return text, safe
                text = "[" + ",".join(str(int(v)) for v in flat_values) + "]"
                safe = "seq" + "-".join(str(int(v)) for v in flat_values[:8])
                if len(flat_values) > 8:
                    safe = f"{safe}-n{len(flat_values)}"
                return text, safe
        except Exception:
            return None, None
        return None, None

    @staticmethod
    def _format_frame_source_debug_lines(batch: dict[str, Tensor], key: str, batch_index: int) -> list[str] | None:
        if key not in batch:
            return None
        try:
            payload = batch[key][batch_index]
            if not isinstance(payload, str) or payload.strip() == "":
                return None
            entries = json.loads(payload)
        except Exception:
            return None

        formatted: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                formatted.append(str(entry))
                continue
            video_name = str(entry.get("video", "NA"))
            source_episode = entry.get("source_episode", "NA")
            source_task = entry.get("source_task", "NA")
            source_frame = entry.get("source_frame", "NA")
            source_ts = entry.get("source_ts", "NA")
            pad_text = " PAD" if bool(entry.get("pad", False)) else ""
            formatted.append(
                f"{video_name}{pad_text}\n"
                f"ep={source_episode} task={source_task}\n"
                f"fr={source_frame} ts={source_ts}"
            )
        return formatted

    def _maybe_dump_input_image_visualizations(self, *, batch: dict[str, Tensor]) -> None:
        if not self.training:
            return
        dump_dir = self.config.input_image_debug_dump_dir
        if dump_dir is None or str(dump_dir).strip() == "":
            return
        interval = int(getattr(self.config, "input_image_debug_dump_interval", 0))
        if interval <= 0:
            return

        batch_idx = self._input_image_debug_batch_idx
        self._input_image_debug_batch_idx += 1
        if batch_idx % interval != 0:
            return

        present_img_keys = [key for key in self.config.image_features if key in batch]
        if not present_img_keys:
            return
        max_cameras = int(getattr(self.config, "input_image_debug_max_cameras", 3))
        if max_cameras > 0:
            present_img_keys = present_img_keys[:max_cameras]
        if not present_img_keys:
            return

        first_image_batch = batch[present_img_keys[0]]
        bsz = int(first_image_batch.shape[0])
        max_samples = int(getattr(self.config, "input_image_debug_max_samples_per_batch", 1))
        if max_samples > 0:
            bsz = min(bsz, max_samples)

        pid = os.getpid()
        step_dir = Path(dump_dir) / f"step_{batch_idx:08d}_p{pid}"
        for bi in range(bsz):
            idx = self._input_image_debug_dump_idx
            self._input_image_debug_dump_idx += 1
            suffix = f"p{pid}_n{idx:08d}_b{bi:02d}"
            sample_idx = None
            dataset_index = None
            if "index" in batch:
                try:
                    sample_idx = int(batch["index"][bi].item())
                    suffix = f"{suffix}_idx{sample_idx:09d}"
                except Exception:
                    sample_idx = None
            if "dataset_index" in batch:
                try:
                    dataset_index = int(batch["dataset_index"][bi].item())
                    suffix = f"{suffix}_ds{dataset_index:02d}"
                except Exception:
                    dataset_index = None
            episode_index_text, episode_index_safe = self._format_debug_batch_value(batch, "episode_index", bi)
            if episode_index_safe:
                suffix = f"{suffix}_ep{episode_index_safe}"
            frame_index_text, frame_index_safe = self._format_debug_batch_value(batch, "frame_index", bi)
            if frame_index_safe:
                suffix = f"{suffix}_fr{frame_index_safe}"
            timestamp_text, timestamp_safe = self._format_debug_batch_value(batch, "timestamp", bi)
            if timestamp_safe:
                suffix = f"{suffix}_ts{timestamp_safe}"
            out_path = step_dir / f"{time.time_ns()}_{suffix}_input_images.png"
            json_out_path = out_path.with_suffix(".json")
            image_tensors = [batch[camera_key][bi] for camera_key in present_img_keys]
            pad_masks = [
                batch[f"{camera_key}_is_pad"][bi] if f"{camera_key}_is_pad" in batch else None
                for camera_key in present_img_keys
            ]
            frame_source_texts = [
                self._format_frame_source_debug_lines(batch, f"{camera_key}_debug_frame_sources", bi)
                for camera_key in present_img_keys
            ]
            self._save_input_image_debug_png(
                image_tensors,
                out_path,
                camera_keys=present_img_keys,
                pad_masks=pad_masks,
                frame_source_texts=frame_source_texts,
                sample_idx=sample_idx,
                dataset_index=dataset_index,
                episode_index_text=episode_index_text,
                frame_index_text=frame_index_text,
                timestamp_text=timestamp_text,
            )
            debug_payload = {
                "sample_idx": sample_idx,
                "dataset_index": dataset_index,
                "episode_index": episode_index_text,
                "frame_index": frame_index_text,
                "timestamp": timestamp_text,
                "camera_keys": present_img_keys,
                "frame_sources": self._collect_frame_source_debug_payload(batch, present_img_keys, bi),
            }
            json_out_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _maybe_dump_box_cross_visualizations(
        self,
        *,
        batch: dict[str, Tensor],
        predicted_boxes: Tensor | None,
        target_boxes: Tensor | None,
        box_valid_mask: Tensor | None,
        predicted_cross_center: Tensor | None,
        target_cross_center: Tensor | None,
        cross_center_valid_mask: Tensor | None,
    ) -> None:
        if not self.training:
            return
        dump_dir = self.config.box_cross_debug_dump_dir
        if dump_dir is None or str(dump_dir).strip() == "":
            return
        interval = int(getattr(self.config, "box_cross_debug_dump_interval", 0))
        if interval <= 0:
            return

        batch_idx = self._box_cross_debug_batch_idx
        self._box_cross_debug_batch_idx += 1
        if batch_idx % interval != 0:
            return

        camera_key = getattr(self.config, "box_cross_debug_camera_key", None)
        if camera_key is None or str(camera_key).strip() == "":
            present_img_keys = [key for key in self.config.image_features if key in batch]
            if not present_img_keys:
                return
            camera_key = present_img_keys[0]
        if camera_key not in batch:
            return

        image_batch = batch[camera_key]
        bsz = int(image_batch.shape[0])
        max_samples = int(getattr(self.config, "box_cross_debug_max_samples_per_batch", 0))
        if max_samples > 0:
            bsz = min(bsz, max_samples)

        pid = os.getpid()
        step_dir = Path(dump_dir) / f"step_{batch_idx:08d}_p{pid}"
        norm_pred_boxes = self._normalize_box_coords_for_draw(predicted_boxes)
        norm_pred_cross_center = self._normalize_pred_coords_for_draw(predicted_cross_center)
        target_boxes_cpu = self._normalize_box_coords_for_draw(target_boxes)
        box_valid_cpu = box_valid_mask.detach().bool().cpu() if box_valid_mask is not None else None
        target_cross_center_cpu = target_cross_center.detach().float().cpu() if target_cross_center is not None else None
        cross_valid_cpu = (
            cross_center_valid_mask.detach().bool().cpu() if cross_center_valid_mask is not None else None
        )

        for bi in range(bsz):
            idx = self._box_cross_debug_dump_idx
            self._box_cross_debug_dump_idx += 1
            suffix = f"p{pid}_n{idx:08d}_b{bi:02d}"
            sample_idx = None
            dataset_index = None
            if "index" in batch:
                try:
                    sample_idx = int(batch["index"][bi].item())
                    suffix = f"{suffix}_idx{sample_idx:09d}"
                except Exception:
                    sample_idx = None
            if "dataset_index" in batch:
                try:
                    dataset_index = int(batch["dataset_index"][bi].item())
                    suffix = f"{suffix}_ds{dataset_index:02d}"
                except Exception:
                    dataset_index = None
            stem = f"{time.time_ns()}_{suffix}"
            out_path = step_dir / f"{stem}_box_cross_debug.png"
            self._save_box_cross_debug_png(
                image_batch[bi],
                out_path,
                camera_key=camera_key,
                predicted_boxes=None if norm_pred_boxes is None else norm_pred_boxes[bi],
                target_boxes=None if target_boxes_cpu is None else target_boxes_cpu[bi],
                box_valid_mask=None if box_valid_cpu is None else box_valid_cpu[bi],
                predicted_cross_center=None if norm_pred_cross_center is None else norm_pred_cross_center[bi],
                target_cross_center=None if target_cross_center_cpu is None else target_cross_center_cpu[bi],
                cross_center_valid=False if cross_valid_cpu is None else bool(cross_valid_cpu[bi].item()),
                sample_idx=sample_idx,
                dataset_index=dataset_index,
            )

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            checkpoint_has_rynnbrain_lora_weights = any(
                key.startswith("model.paligemma_with_expert.rynnbrain.") and ".lora_" in key
                for key in remapped_state_dict
            )
            checkpoint_has_wrapped_rynnbrain_base = any(
                key.startswith("model.paligemma_with_expert.rynnbrain.base_model.model.")
                for key in remapped_state_dict
            )
            should_skip_rynnbrain_checkpoint_weights = (
                getattr(model.config, "use_rynnbrain", False)
                and not getattr(model.config, "load_rynnbrain_from_pretrained", True)
                and not checkpoint_has_rynnbrain_lora_weights
                and not checkpoint_has_wrapped_rynnbrain_base
            )

            if should_skip_rynnbrain_checkpoint_weights:
                rynnbrain_prefix = "model.paligemma_with_expert.rynnbrain."
                original_key_count = len(remapped_state_dict)
                remapped_state_dict = {
                    key: value for key, value in remapped_state_dict.items() if not key.startswith(rynnbrain_prefix)
                }
                skipped_key_count = original_key_count - len(remapped_state_dict)
                print(
                    "Skipping RynnBrain weights from pretrained checkpoint and keeping "
                    f"default weights from rynnbrain_path. Filtered {skipped_key_count} keys."
                )

            # Load the remapped state dict into the model.
            # NOTE: old checkpoints do not contain the optional aux heads added for coordinate prediction.
            # We retry with strict=False only when the remaining mismatches are limited to known-optional keys.
            optional_key_prefixes = (
                "model.box_out_proj.",
                "model.cross_center_out_proj.",
                "model.behavior_b1k_action_out_projs.",
                "model.joint_action_out_proj.",
                "model.gripper_action_out_proj.",
            )
            optional_exact_keys = {
                "model.paligemma_with_expert.gemma_expert.model.embed_tokens.weight",
                "model.paligemma_with_expert.action_k_mix_weights",
                "model.paligemma_with_expert.action_v_mix_weights",
                "model.paligemma_with_expert.action_k_mix_bias",
                "model.paligemma_with_expert.action_v_mix_bias",
            }

            def _is_optional_key(key: str) -> bool:
                if key in optional_exact_keys or key.startswith(optional_key_prefixes):
                    return True
                if should_skip_rynnbrain_checkpoint_weights and key.startswith(
                    "model.paligemma_with_expert.rynnbrain."
                ):
                    return True
                return False

            split_head_keys = (
                "model.joint_action_out_proj.weight",
                "model.joint_action_out_proj.bias",
                "model.gripper_action_out_proj.weight",
                "model.gripper_action_out_proj.bias",
            )
            behavior_b1k_semantic_head_keys = tuple(
                key
                for group_name, _group_indices in BEHAVIOR_B1K_SEMANTIC_ACTION_HEAD_GROUPS
                for key in (
                    f"model.behavior_b1k_action_out_projs.{group_name}.weight",
                    f"model.behavior_b1k_action_out_projs.{group_name}.bias",
                )
            )

            def _maybe_initialize_split_action_heads(*, force: bool = False, missing_keys: list[str] | tuple[str, ...] = ()) -> None:
                if not getattr(model.config, "enable_split_action_heads", False):
                    return
                if not force and not any(key in missing_keys for key in split_head_keys):
                    return
                present_split_head_keys = [key for key in split_head_keys if key in remapped_state_dict]
                if present_split_head_keys:
                    print(
                        "Skipping legacy split-head initialization because checkpoint already contains "
                        f"split action head weights: {present_split_head_keys}"
                    )
                    return

                legacy_weight = remapped_state_dict.get("model.action_out_proj.weight")
                legacy_bias = remapped_state_dict.get("model.action_out_proj.bias")
                if legacy_weight is not None and legacy_bias is not None:
                    print("Initializing split action heads from legacy checkpoint action_out_proj weights")
                    model.model.initialize_split_action_heads_from_legacy_tensors(legacy_weight, legacy_bias)
                    return

                print("Initializing split action heads from in-memory legacy action_out_proj weights")
                model.model.initialize_split_action_heads_from_legacy()

            def _maybe_initialize_behavior_b1k_semantic_action_heads(
                *, force: bool = False, missing_keys: list[str] | tuple[str, ...] = ()
            ) -> None:
                if not getattr(model.config, "enable_behavior_b1k_semantic_action_heads", False):
                    return
                if not force and not any(key in missing_keys for key in behavior_b1k_semantic_head_keys):
                    return
                present_behavior_head_keys = [
                    key for key in behavior_b1k_semantic_head_keys if key in remapped_state_dict
                ]
                if present_behavior_head_keys:
                    print(
                        "Skipping legacy Behavior semantic-head initialization because checkpoint already contains "
                        f"semantic action head weights: {present_behavior_head_keys[:4]}"
                    )
                    return

                legacy_weight = remapped_state_dict.get("model.action_out_proj.weight")
                legacy_bias = remapped_state_dict.get("model.action_out_proj.bias")
                if legacy_weight is not None and legacy_bias is not None:
                    print("Initializing Behavior semantic action heads from legacy checkpoint action_out_proj weights")
                    model.model.initialize_behavior_b1k_semantic_action_heads_from_legacy_tensors(
                        legacy_weight, legacy_bias
                    )
                    return

                print("Initializing Behavior semantic action heads from in-memory legacy action_out_proj weights")
                model.model.initialize_behavior_b1k_semantic_action_heads_from_legacy()

            try:
                missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)
            except RuntimeError as e:
                if strict:
                    relaxed_missing_keys, relaxed_unexpected_keys = model.load_state_dict(
                        remapped_state_dict, strict=False
                    )
                    if all(_is_optional_key(key) for key in relaxed_missing_keys) and all(
                        _is_optional_key(key) for key in relaxed_unexpected_keys
                    ):
                        print(
                            "Warning: strict checkpoint loading skipped optional aux/tied-weight keys. "
                            "Retrying with strict=False so remaining weights load."
                        )
                        missing_keys, unexpected_keys = relaxed_missing_keys, relaxed_unexpected_keys
                    else:
                        raise e
                else:
                    print(
                        "Warning: strict=False loading retained mismatched keys outside the optional aux heads. "
                        "Proceeding because strict loading is disabled."
                    )
                    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")
            _maybe_initialize_split_action_heads(missing_keys=missing_keys)
            _maybe_initialize_behavior_b1k_semantic_action_heads(missing_keys=missing_keys)

        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")
            _maybe_initialize_split_action_heads(force=True)
            _maybe_initialize_behavior_b1k_semantic_action_heads(force=True)

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(
        self,
        batch: dict[str, Tensor],
        *,
        selected_img_keys: list[str] | tuple[str, ...] | None = None,
        return_images_by_key: bool = False,
    ) -> tuple[list[Tensor], list[Tensor]] | tuple[list[Tensor], list[Tensor], dict[str, Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        The legacy PaliGemma/SigLIP path expects [-1, 1], while the RynnBrain static-image
        processor path can stay in [0, 1] until its final uint8 packing step.
        """
        images = []
        img_masks = []
        images_by_key: dict[str, Tensor] = {}

        # Get device from model parameters
        device = next(self.parameters()).device

        configured_img_keys = list(selected_img_keys) if selected_img_keys is not None else list(self.config.image_features)
        present_img_keys = [key for key in configured_img_keys if key in batch]
        missing_img_keys = [] if selected_img_keys is not None else [key for key in configured_img_keys if key not in batch]
        if selected_img_keys is not None:
            missing_selected_keys = [key for key in configured_img_keys if key not in batch]
            if missing_selected_keys:
                raise ValueError(
                    "Selected image keys are missing from the batch: "
                    f"{missing_selected_keys}. Available keys: {list(batch.keys())}"
                )

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {configured_img_keys})"
            )

        use_rynnbrain = bool(getattr(self.config, "use_rynnbrain", False))

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            is_temporal = img.ndim == 5
            keep_static_range_for_processor = use_rynnbrain and not is_temporal

            # Ensure tensor is on the same device as the model unless the static-image
            # RynnBrain processor path will repack it on CPU immediately afterwards.
            if not keep_static_range_for_processor and img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency.
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            if is_temporal:
                bsize, tsize = img.shape[:2]
                is_channels_first = img.shape[2] == 3
                if is_channels_first:
                    img = img.permute(0, 1, 3, 4, 2)
                flat_img = img.reshape(bsize * tsize, img.shape[-3], img.shape[-2], img.shape[-1])
                if flat_img.shape[1:3] != self.config.image_resolution:
                    flat_img = resize_with_pad_torch(flat_img, *self.config.image_resolution)
                flat_img = flat_img * 2.0 - 1.0
                if is_channels_first:
                    flat_img = flat_img.permute(0, 3, 1, 2)
                img = flat_img.reshape(bsize, tsize, *flat_img.shape[1:])
            else:
                # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
                is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

                if is_channels_first:
                    # Convert [B, C, H, W] to [B, H, W, C] for processing
                    img = img.permute(0, 2, 3, 1)

                # from openpi preprocess_observation_pytorch: Resize with padding if needed
                if img.shape[1:3] != self.config.image_resolution:
                    img = resize_with_pad_torch(img, *self.config.image_resolution)

                if not keep_static_range_for_processor:
                    # Normalize from [0,1] to [-1,1] as expected by the legacy path.
                    img = img * 2.0 - 1.0

                # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
                if is_channels_first:
                    img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            images_by_key[key] = img
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        if return_images_by_key:
            return images, img_masks, images_by_key
        return images, img_masks

    @staticmethod
    def _extract_raw_texts_from_batch(batch: dict[str, Tensor]) -> list[str] | None:
        task_value = batch.get("task")
        if isinstance(task_value, str):
            return [task_value]
        if isinstance(task_value, tuple):
            task_value = list(task_value)
        if isinstance(task_value, list) and all(isinstance(task, str) for task in task_value):
            return list(task_value)
        return None

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        if self.config.action_target_mode == "delta_from_state":
            state = self._prepare_state_for_action(batch, actions.device, actions.dtype)
            actions = actions - state.expand(-1, actions.shape[1], -1)
        return self.model._zero_unused_action_dims(actions)

    @staticmethod
    def _prepare_action_valid_mask(batch: dict[str, Tensor], actions: Tensor) -> Tensor | None:
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            return None
        action_is_pad = action_is_pad.to(device=actions.device, dtype=torch.bool)
        if action_is_pad.ndim == 3 and action_is_pad.shape[-1] == 1:
            action_is_pad = action_is_pad.squeeze(-1)
        if action_is_pad.shape != actions.shape[:2]:
            raise ValueError(
                "action_is_pad must have shape [batch, time]: "
                f"{tuple(action_is_pad.shape)} != {tuple(actions.shape[:2])}"
            )
        return ~action_is_pad

    def _prepare_state_for_action(self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype) -> Tensor:
        if OBS_STATE not in batch:
            raise ValueError(f"{OBS_STATE} is required when action_target_mode=delta_from_state")
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim == 3:
            state = state[:, -1, :]
        state = pad_vector(state, self.config.max_action_dim)
        return state[:, None, :]

    def _prepare_clip_running_status(self, batch: dict[str, Tensor]) -> Tensor | None:
        status_key = self.config.clip_running_status_key
        if status_key not in batch:
            return None
        status = batch[status_key]
        device = next(self.parameters()).device
        status = status.to(device=device, dtype=torch.float32)
        if status.ndim == 0:
            status = status[None, None]
        elif status.ndim == 1:
            status = status[:, None]
        else:
            status = status.reshape(status.shape[0], -1)
        return status.clamp(0.0, 1.0)

    def _prepare_task_id_target(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.enable_task_aux_loss:
            return None
        return batch["task_id"]

    def _prepare_box_target(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.enable_box_aux_loss:
            return None, None
        key = self.config.box_aux_key
        if key not in batch:
            return None, None
        device = next(self.parameters()).device
        boxes = batch[key].to(device=device, dtype=torch.float32)
        if boxes.ndim == 2:
            boxes = boxes[:, None, :]
        valid_mask = (boxes >= 0).all(dim=-1)
        return boxes, valid_mask

    def _prepare_cross_center_target(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.enable_cross_center_aux_loss:
            return None, None
        key = self.config.cross_center_aux_key
        if key not in batch:
            return None, None
        device = next(self.parameters()).device
        cross_center = batch[key].to(device=device, dtype=torch.float32)
        if cross_center.ndim == 1:
            cross_center = cross_center[None, :]
        valid_mask = (cross_center >= 0).all(dim=-1)
        return cross_center, valid_mask

    @staticmethod
    def _compute_box_aux_loss(
        predicted_boxes: Tensor | None,
        target_boxes: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if predicted_boxes is None or target_boxes is None or valid_mask is None:
            return None, None
        valid_mask = valid_mask.to(dtype=torch.bool)
        loss_per_box = F.smooth_l1_loss(predicted_boxes, target_boxes, reduction="none").mean(dim=-1)
        valid_float = valid_mask.to(dtype=loss_per_box.dtype)
        valid_count = valid_float.sum(dim=1)
        per_sample_loss = (loss_per_box * valid_float).sum(dim=1) / valid_count.clamp_min(1.0)
        if not torch.any(valid_mask):
            return predicted_boxes.sum() * 0.0, per_sample_loss * 0.0
        scalar_loss = (loss_per_box * valid_float).sum() / valid_float.sum().clamp_min(1.0)
        return scalar_loss, per_sample_loss

    @staticmethod
    def _compute_cross_center_aux_loss(
        predicted_cross_center: Tensor | None,
        target_cross_center: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if predicted_cross_center is None or target_cross_center is None or valid_mask is None:
            return None, None
        valid_mask = valid_mask.to(dtype=torch.bool)
        loss_per_sample = F.smooth_l1_loss(predicted_cross_center, target_cross_center, reduction="none").mean(dim=-1)
        valid_float = valid_mask.to(dtype=loss_per_sample.dtype)
        per_sample_loss = loss_per_sample * valid_float
        if not torch.any(valid_mask):
            return predicted_cross_center.sum() * 0.0, per_sample_loss * 0.0
        scalar_loss = per_sample_loss.sum() / valid_float.sum().clamp_min(1.0)
        return scalar_loss, per_sample_loss

    def _prepare_state_for_denoise_condition(
        self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        if not self.config.enable_state_in_action_time_emb:
            return None
        if OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state[None, :]
        return pad_vector(state, self.config.max_state_dim)

    def _prepare_state_history_for_prefix(
        self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        if not getattr(self.config, "enable_state_in_prefix_tokens", False):
            return None
        if OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim == 3 and not getattr(self.config, "state_prefix_use_history", False):
            state = state[:, -1:, :]
        return state

    def _prepare_state_history_for_flow_source(
        self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        if getattr(self.config, "flow_source_mode", "gaussian") not in {"state_history", "blend"}:
            return None
        if OBS_STATE not in batch:
            raise ValueError(f"{OBS_STATE} is required when flow_source_mode uses state history")
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state[None, None, :]
        elif state.ndim == 2:
            state = state[:, None, :]
        elif state.ndim != 3:
            raise ValueError(f"Unsupported state ndim for flow source: {state.ndim}")
        target_frames = int(getattr(self.config, "flow_source_state_num_frames", 1))
        if state.shape[1] >= target_frames:
            return state[:, -target_frames:, :]
        pad_frames = state[:, :1, :].expand(-1, target_frames - state.shape[1], -1)
        return torch.cat([pad_frames, state], dim=1)

    def _prepare_vggt_images(
        self,
        batch: dict[str, Tensor],
        *,
        preprocessed_images_by_key: dict[str, Tensor] | None = None,
    ) -> list[Tensor] | None:
        if not getattr(self.config, "enable_vggt_omega_prefix_token", False):
            return None
        selected_img_keys = list(getattr(self.config, "vggt_omega_camera_keys", []))
        if preprocessed_images_by_key is not None:
            missing_selected_keys = [key for key in selected_img_keys if key not in preprocessed_images_by_key]
            if not missing_selected_keys:
                return [preprocessed_images_by_key[key] for key in selected_img_keys]

        vggt_images, _ = self._preprocess_images(batch, selected_img_keys=selected_img_keys)
        return vggt_images

    @staticmethod
    def _prepare_short_term_memory_debug_info(batch: dict[str, Tensor]) -> dict[str, int] | None:
        keys = (
            "short_term_memory_strategy_idx",
            "short_term_memory_num_frames",
            "short_term_memory_stride",
        )
        if not all(key in batch for key in keys):
            return None

        debug_info = {}
        for key in keys:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                flat_value = value.reshape(-1)
                if flat_value.numel() == 0:
                    return None
                debug_info[key] = int(flat_value[0].item())
            else:
                debug_info[key] = int(value)
        return debug_info

    def _prepare_image_pad_masks(
        self,
        batch: dict[str, Tensor],
        selected_img_keys: list[str] | None = None,
    ) -> list[Tensor | None]:
        configured_img_keys = list(selected_img_keys) if selected_img_keys is not None else list(self.config.image_features)
        present_img_keys = [key for key in configured_img_keys if key in batch]
        missing_img_keys = [] if selected_img_keys is not None else [key for key in configured_img_keys if key not in batch]

        image_pad_masks: list[Tensor | None] = []
        for key in present_img_keys:
            pad_key = f"{key}_is_pad"
            pad_mask = batch.get(pad_key)
            if isinstance(pad_mask, torch.Tensor):
                image_pad_masks.append(pad_mask.to(dtype=torch.bool))
            else:
                image_pad_masks.append(None)

        image_pad_masks.extend([None] * len(missing_img_keys))
        return image_pad_masks

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks, images_by_key = self._preprocess_images(batch, return_images_by_key=True)
        img_pad_masks = self._prepare_image_pad_masks(batch)
        vggt_images = self._prepare_vggt_images(batch, preprocessed_images_by_key=images_by_key)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        raw_texts = self._extract_raw_texts_from_batch(batch)

        clip_running_status = self._prepare_clip_running_status(batch)
        state_cond = self._prepare_state_for_denoise_condition(batch, tokens.device, torch.float32)
        state_history = self._prepare_state_history_for_prefix(batch, tokens.device, torch.float32)
        flow_source_state_history = self._prepare_state_history_for_flow_source(batch, tokens.device, torch.float32)
        actions = self.model.sample_actions(
            images,
            img_masks,
            img_pad_masks,
            tokens,
            masks,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status,
            state=state_cond,
            state_history=state_history,
            flow_source_state_history=flow_source_state_history,
            **kwargs,
        )
        if self.config.action_target_mode == "delta_from_state":
            state = self._prepare_state_for_action(batch, actions.device, actions.dtype)
            actions = actions + state.expand(-1, actions.shape[1], -1)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def predict_action_chunk_with_status(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        images, img_masks, images_by_key = self._preprocess_images(batch, return_images_by_key=True)
        img_pad_masks = self._prepare_image_pad_masks(batch)
        vggt_images = self._prepare_vggt_images(batch, preprocessed_images_by_key=images_by_key)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        raw_texts = self._extract_raw_texts_from_batch(batch)
        clip_running_status = self._prepare_clip_running_status(batch)
        state_cond = self._prepare_state_for_denoise_condition(batch, tokens.device, torch.float32)
        state_history = self._prepare_state_history_for_prefix(batch, tokens.device, torch.float32)
        flow_source_state_history = self._prepare_state_history_for_flow_source(batch, tokens.device, torch.float32)
        actions, predicted_clip_running_status_chunk = self.model.sample_actions_and_status(
            images,
            img_masks,
            img_pad_masks,
            tokens,
            masks,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status,
            state=state_cond,
            state_history=state_history,
            flow_source_state_history=flow_source_state_history,
            **kwargs,
        )
        if self.config.action_target_mode == "delta_from_state":
            state = self._prepare_state_for_action(batch, actions.device, actions.dtype)
            actions = actions + state.expand(-1, actions.shape[1], -1)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions, predicted_clip_running_status_chunk

    @torch.no_grad()
    def predict_action_chunk_with_status_and_task(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.eval()
        images, img_masks, images_by_key = self._preprocess_images(batch, return_images_by_key=True)
        img_pad_masks = self._prepare_image_pad_masks(batch)
        vggt_images = self._prepare_vggt_images(batch, preprocessed_images_by_key=images_by_key)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        raw_texts = self._extract_raw_texts_from_batch(batch)
        clip_running_status = self._prepare_clip_running_status(batch)
        state_cond = self._prepare_state_for_denoise_condition(batch, tokens.device, torch.float32)
        state_history = self._prepare_state_history_for_prefix(batch, tokens.device, torch.float32)
        flow_source_state_history = self._prepare_state_history_for_flow_source(batch, tokens.device, torch.float32)
        actions, predicted_clip_running_status_chunk, task_id_logits = self.model.sample_actions_and_status(
            images,
            img_masks,
            img_pad_masks,
            tokens,
            masks,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status,
            state=state_cond,
            state_history=state_history,
            flow_source_state_history=flow_source_state_history,
            return_task_logits=True,
            **kwargs,
        )
        if self.config.action_target_mode == "delta_from_state":
            state = self._prepare_state_for_action(batch, actions.device, actions.dtype)
            actions = actions + state.expand(-1, actions.shape[1], -1)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions, predicted_clip_running_status_chunk, task_id_logits

    @torch.no_grad()
    def predict_action_chunk_with_status_task_and_aux(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        self.eval()
        images, img_masks, images_by_key = self._preprocess_images(batch, return_images_by_key=True)
        img_pad_masks = self._prepare_image_pad_masks(batch)
        vggt_images = self._prepare_vggt_images(batch, preprocessed_images_by_key=images_by_key)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        raw_texts = self._extract_raw_texts_from_batch(batch)
        clip_running_status = self._prepare_clip_running_status(batch)
        state_cond = self._prepare_state_for_denoise_condition(batch, tokens.device, torch.float32)
        state_history = self._prepare_state_history_for_prefix(batch, tokens.device, torch.float32)
        flow_source_state_history = self._prepare_state_history_for_flow_source(batch, tokens.device, torch.float32)
        actions, predicted_clip_running_status_chunk, task_id_logits, predicted_boxes, predicted_cross_center = (
            self.model.sample_actions_and_status(
                images,
                img_masks,
                img_pad_masks,
                tokens,
                masks,
                vggt_images=vggt_images,
                raw_texts=raw_texts,
                clip_running_status=clip_running_status,
                state=state_cond,
                state_history=state_history,
                flow_source_state_history=flow_source_state_history,
                return_task_logits=True,
                return_aux_predictions=True,
                **kwargs,
            )
        )
        if self.config.action_target_mode == "delta_from_state":
            state = self._prepare_state_for_action(batch, actions.device, actions.dtype)
            actions = actions + state.expand(-1, actions.shape[1], -1)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions, predicted_clip_running_status_chunk, task_id_logits, predicted_boxes, predicted_cross_center

    def get_last_inference_timing_ms(self) -> dict[str, float]:
        timing = getattr(self.model, "last_inference_timing_ms", {})
        if isinstance(timing, dict):
            return dict(timing)
        return {}

    def forward(
        self, batch: dict[str, Tensor], reduction: str = "mean", enable_train_rollout_debug: bool = True
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        selected_group_camera_keys = (
            list(getattr(self.config, "group_consistency_camera_keys", []))
            if getattr(self.config, "enable_group_consistency_loss", False)
            else None
        )
        images, img_masks, images_by_key = self._preprocess_images(
            batch,
            selected_img_keys=selected_group_camera_keys,
            return_images_by_key=True,
        )
        img_pad_masks = self._prepare_image_pad_masks(batch, selected_img_keys=selected_group_camera_keys)
        batch_for_loss, images, img_masks, img_pad_masks, consistency_meta = self._maybe_expand_group_consistency_inputs(
            batch=batch,
            images=images,
            img_masks=img_masks,
            img_pad_masks=img_pad_masks,
        )
        if batch_for_loss is batch and consistency_meta is None:
            vggt_images = self._prepare_vggt_images(batch_for_loss, preprocessed_images_by_key=images_by_key)
        else:
            vggt_images = self._prepare_vggt_images(batch_for_loss)
        tokens, masks = batch_for_loss[f"{OBS_LANGUAGE_TOKENS}"], batch_for_loss[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        raw_texts = self._extract_raw_texts_from_batch(batch_for_loss)

        actions = self.prepare_action(batch_for_loss)
        action_valid_mask = self._prepare_action_valid_mask(batch_for_loss, actions)
        action_dim_valid_mask = self._prepare_action_dim_valid_mask(actions)
        clip_running_status_target = self._prepare_clip_running_status(batch_for_loss)
        box_target, box_valid_mask = self._prepare_box_target(batch_for_loss)
        cross_center_target, cross_center_valid_mask = self._prepare_cross_center_target(batch_for_loss)

        state_cond = self._prepare_state_for_denoise_condition(batch_for_loss, actions.device, actions.dtype)
        state_history = self._prepare_state_history_for_prefix(batch_for_loss, actions.device, torch.float32)
        flow_source_state_history = self._prepare_state_history_for_flow_source(batch_for_loss, actions.device, torch.float32)
        short_term_memory_debug_info = self._prepare_short_term_memory_debug_info(batch_for_loss)
        (
            losses,
            predicted_clip_running_status,
            predicted_action,
            task_id_logits,
            predicted_boxes,
            predicted_cross_center,
            future_action_aux_loss,
            future_action_aux_per_sample_loss,
            flow_source,
            pooled_feature,
        ) = self.model.forward(
            images,
            img_masks,
            img_pad_masks,
            tokens,
            masks,
            actions,
            action_valid_mask=action_valid_mask,
            action_dim_valid_mask=action_dim_valid_mask,
            vggt_images=vggt_images,
            raw_texts=raw_texts,
            clip_running_status=clip_running_status_target,
            state=state_cond,
            state_history=state_history,
            flow_source_state_history=flow_source_state_history,
            short_term_memory_debug_info=short_term_memory_debug_info,
        )
        task_id_target = self._prepare_task_id_target(batch_for_loss)
        box_loss, box_per_sample_loss = self._compute_box_aux_loss(predicted_boxes, box_target, box_valid_mask)
        cross_center_loss, cross_center_per_sample_loss = self._compute_cross_center_aux_loss(
            predicted_cross_center, cross_center_target, cross_center_valid_mask
        )
        (
            group_consistency_loss,
            group_consistency_per_sample_loss,
            group_consistency_cos_sim,
        ) = self._compute_group_consistency_loss(
            self.model.project_group_consistency_feature(pooled_feature),
            consistency_meta,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        target_action = actions[:, :, :original_action_dim]
        predicted_action = predicted_action[:, :, :original_action_dim]
        (
            ee_position_loss,
            ee_position_per_sample_loss,
            ee_orientation_loss,
            ee_orientation_per_sample_loss,
        ) = self._compute_ee_pose_losses(predicted_action, target_action, batch_for_loss, original_action_dim)
        (
            pose_translation_loss,
            pose_translation_per_sample_loss,
            pose_rotation_loss,
            pose_rotation_per_sample_loss,
        ) = self._compute_pose_action_losses(
            predicted_action,
            target_action,
            action_valid_mask,
        )
        if enable_train_rollout_debug:
            self._schedule_train_rollout_debug(
                images=images,
                img_masks=img_masks,
                vggt_images=vggt_images,
                tokens=tokens,
                masks=masks,
                batch=batch_for_loss,
                state_cond=state_cond,
                state_history=state_history,
                flow_source_state_history=flow_source_state_history,
                flow_source=flow_source,
                predicted_action=predicted_action,
                target_action=target_action,
                original_action_dim=original_action_dim,
            )

        if action_valid_mask is None:
            loss_per_dim = losses.mean(dim=(0, 1))
            action_valid_ratio = 1.0
        else:
            loss_weights = action_valid_mask.to(device=losses.device, dtype=losses.dtype).unsqueeze(-1)
            loss_per_dim = (losses * loss_weights).sum(dim=(0, 1)) / loss_weights.sum().clamp_min(1.0)
            action_valid_ratio = float(action_valid_mask.float().mean().item())
        loss_dict = {
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            "action_valid_ratio": action_valid_ratio,
        }
        self._add_ee_readable_metrics(loss_dict, ee_position_loss, ee_orientation_loss)
        if pose_translation_loss is not None:
            loss_dict["pose_translation_l2_m"] = pose_translation_loss.item()
            loss_dict["pose_translation_loss_m"] = pose_translation_loss.item()
            loss_dict["pose_translation_weighted_loss"] = (
                self.config.pose_translation_loss_weight * pose_translation_loss.detach()
            ).item()
        if pose_rotation_loss is not None:
            loss_dict["pose_rotation_geodesic_rad"] = pose_rotation_loss.item()
            rotation_deg = (
                pose_rotation_loss.detach() * (180.0 / torch.pi)
            ).item()
            loss_dict["pose_rotation_geodesic_deg"] = rotation_deg
            loss_dict["pose_rotation_loss_rad"] = pose_rotation_loss.item()
            loss_dict["pose_rotation_error_deg"] = rotation_deg
            loss_dict["pose_rotation_weighted_loss"] = (
                self.config.pose_rotation_loss_weight * pose_rotation_loss.detach()
            ).item()
        if box_valid_mask is not None:
            valid_sample_mask = box_valid_mask.any(dim=1) if box_valid_mask.ndim > 1 else box_valid_mask
            loss_dict[BOX_AUX_VALID_COUNT] = float(valid_sample_mask.sum().item())
            loss_dict[BOX_AUX_VALID_RATIO] = float(valid_sample_mask.float().mean().item())
        if cross_center_valid_mask is not None:
            loss_dict[CROSS_CENTER_AUX_VALID_COUNT] = float(cross_center_valid_mask.sum().item())
            loss_dict[CROSS_CENTER_AUX_VALID_RATIO] = float(cross_center_valid_mask.float().mean().item())
        if consistency_meta is not None:
            loss_dict["group_consistency_group_size"] = float(consistency_meta["group_size"])
        if group_consistency_cos_sim is not None:
            loss_dict["group_consistency_cos_sim"] = group_consistency_cos_sim
        self._maybe_dump_box_cross_visualizations(
            batch=batch_for_loss,
            predicted_boxes=predicted_boxes,
            target_boxes=box_target,
            box_valid_mask=box_valid_mask,
            predicted_cross_center=predicted_cross_center,
            target_cross_center=cross_center_target,
            cross_center_valid_mask=cross_center_valid_mask,
        )
        self._maybe_dump_input_image_visualizations(batch=batch_for_loss)
        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            (
                action_per_sample_loss,
                joint_action_per_sample_loss,
                gripper_action_per_sample_loss,
            ) = self._compute_action_per_sample_loss(losses, action_valid_mask=action_valid_mask)
            per_sample_loss = action_per_sample_loss
            if joint_action_per_sample_loss is not None:
                loss_dict[self._non_gripper_action_loss_name()] = joint_action_per_sample_loss.mean().item()
            if gripper_action_per_sample_loss is not None:
                loss_dict["gripper_action_loss"] = gripper_action_per_sample_loss.mean().item()
            if task_id_target is not None and self.config.task_aux_loss_weight > 0:
                task_ce_loss = F.cross_entropy(task_id_logits, task_id_target, reduction="none")
                per_sample_loss = per_sample_loss + self.config.task_aux_loss_weight * task_ce_loss
                loss_dict["task_id_loss"] = task_ce_loss.mean().item()
            if box_per_sample_loss is not None and self.config.box_aux_loss_weight > 0:
                per_sample_loss = per_sample_loss + self.config.box_aux_loss_weight * box_per_sample_loss
                loss_dict[BOX_AUX_LOSS] = box_per_sample_loss.mean().item()
            if cross_center_per_sample_loss is not None and self.config.cross_center_aux_loss_weight > 0:
                per_sample_loss = (
                    per_sample_loss + self.config.cross_center_aux_loss_weight * cross_center_per_sample_loss
                )
                loss_dict[CROSS_CENTER_AUX_LOSS] = cross_center_per_sample_loss.mean().item()
            if future_action_aux_per_sample_loss is not None and self.config.future_action_aux_loss_weight > 0:
                per_sample_loss = per_sample_loss + self.config.future_action_aux_loss_weight * future_action_aux_per_sample_loss
                loss_dict[FUTURE_ACTION_AUX_LOSS] = future_action_aux_per_sample_loss.mean().item()
            if group_consistency_per_sample_loss is not None and self.config.group_consistency_loss_weight > 0:
                per_sample_loss = per_sample_loss + self.config.group_consistency_loss_weight * group_consistency_per_sample_loss
                loss_dict[GROUP_CONSISTENCY_AUX_LOSS] = group_consistency_per_sample_loss.mean().item()
            if ee_position_per_sample_loss is not None and self.config.ee_position_loss_weight > 0:
                per_sample_loss = per_sample_loss + self.config.ee_position_loss_weight * ee_position_per_sample_loss
                loss_dict["ee_position_loss"] = ee_position_per_sample_loss.mean().item()
            if ee_orientation_per_sample_loss is not None and self.config.ee_orientation_loss_weight > 0:
                per_sample_loss = (
                    per_sample_loss + self.config.ee_orientation_loss_weight * ee_orientation_per_sample_loss
                )
                loss_dict["ee_orientation_loss"] = ee_orientation_per_sample_loss.mean().item()
            if (
                pose_translation_per_sample_loss is not None
                and self.config.pose_translation_loss_weight > 0
            ):
                per_sample_loss = (
                    per_sample_loss
                    + self.config.pose_translation_loss_weight * pose_translation_per_sample_loss
                )
            if pose_rotation_per_sample_loss is not None and self.config.pose_rotation_loss_weight > 0:
                per_sample_loss = (
                    per_sample_loss
                    + self.config.pose_rotation_loss_weight * pose_rotation_per_sample_loss
                )

            if clip_running_status_target is not None and predicted_clip_running_status is not None:
                if clip_running_status_target.shape[1] == 1:
                    pred_status = predicted_clip_running_status[:, :1]
                    tgt_status = clip_running_status_target
                else:
                    target_t = min(int(clip_running_status_target.shape[1]), int(predicted_clip_running_status.shape[1]))
                    pred_status = predicted_clip_running_status[:, :target_t]
                    tgt_status = clip_running_status_target[:, :target_t]
                self._maybe_dump_clip_status_curves(pred_status, tgt_status, predicted_action, target_action, batch_for_loss)
                status_loss = F.mse_loss(pred_status, tgt_status, reduction="none").mean(dim=1)
                per_sample_loss = per_sample_loss + self.config.clip_running_status_loss_weight * status_loss
                loss_dict[CLIP_RUNNING_STATUS] = pred_status.mean().item()
                loss_dict[f"{CLIP_RUNNING_STATUS}_loss"] = status_loss.mean().item()
            else:
                self._maybe_dump_clip_status_curves(None, None, predicted_action, target_action, batch_for_loss)
            loss_dict["loss"] = per_sample_loss.mean().item()
            loss_dict["action_loss"] = action_per_sample_loss.mean().item()
            loss_dict["_action_per_sample_loss"] = action_per_sample_loss
            loss_dict["_task_id_logits"] = task_id_logits
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            action_loss, joint_action_loss, gripper_action_loss = self._compute_action_loss(
                losses, action_valid_mask=action_valid_mask
            )
            loss = action_loss
            loss_dict["action_loss"] = action_loss.item()
            if joint_action_loss is not None:
                loss_dict[self._non_gripper_action_loss_name()] = joint_action_loss.item()
            if gripper_action_loss is not None:
                loss_dict["gripper_action_loss"] = gripper_action_loss.item()
            if task_id_target is not None and self.config.task_aux_loss_weight > 0:
                task_ce_loss = F.cross_entropy(task_id_logits, task_id_target)
                loss = loss + self.config.task_aux_loss_weight * task_ce_loss
                loss_dict["task_id_loss"] = task_ce_loss.item()
            if box_loss is not None and self.config.box_aux_loss_weight > 0:
                loss = loss + self.config.box_aux_loss_weight * box_loss
                loss_dict[BOX_AUX_LOSS] = box_loss.item()
            if cross_center_loss is not None and self.config.cross_center_aux_loss_weight > 0:
                loss = loss + self.config.cross_center_aux_loss_weight * cross_center_loss
                loss_dict[CROSS_CENTER_AUX_LOSS] = cross_center_loss.item()
            if future_action_aux_loss is not None and self.config.future_action_aux_loss_weight > 0:
                loss = loss + self.config.future_action_aux_loss_weight * future_action_aux_loss
                loss_dict[FUTURE_ACTION_AUX_LOSS] = future_action_aux_loss.item()
            if group_consistency_loss is not None and self.config.group_consistency_loss_weight > 0:
                loss = loss + self.config.group_consistency_loss_weight * group_consistency_loss
                loss_dict[GROUP_CONSISTENCY_AUX_LOSS] = group_consistency_loss.item()
            if ee_position_loss is not None and self.config.ee_position_loss_weight > 0:
                loss = loss + self.config.ee_position_loss_weight * ee_position_loss
                loss_dict["ee_position_loss"] = ee_position_loss.item()
            if ee_orientation_loss is not None and self.config.ee_orientation_loss_weight > 0:
                loss = loss + self.config.ee_orientation_loss_weight * ee_orientation_loss
                loss_dict["ee_orientation_loss"] = ee_orientation_loss.item()
            if pose_translation_loss is not None and self.config.pose_translation_loss_weight > 0:
                loss = loss + self.config.pose_translation_loss_weight * pose_translation_loss
            if pose_rotation_loss is not None and self.config.pose_rotation_loss_weight > 0:
                loss = loss + self.config.pose_rotation_loss_weight * pose_rotation_loss
            if clip_running_status_target is not None and predicted_clip_running_status is not None:
                if clip_running_status_target.shape[1] == 1:
                    pred_status = predicted_clip_running_status[:, :1]
                    tgt_status = clip_running_status_target
                else:
                    target_t = min(int(clip_running_status_target.shape[1]), int(predicted_clip_running_status.shape[1]))
                    pred_status = predicted_clip_running_status[:, :target_t]
                    tgt_status = clip_running_status_target[:, :target_t]
                self._maybe_dump_clip_status_curves(pred_status, tgt_status, predicted_action, target_action, batch_for_loss)
                status_loss = F.mse_loss(pred_status, tgt_status)
                loss = loss + self.config.clip_running_status_loss_weight * status_loss
                loss_dict[CLIP_RUNNING_STATUS] = pred_status.mean().item()
                loss_dict[f"{CLIP_RUNNING_STATUS}_loss"] = status_loss.item()
            else:
                self._maybe_dump_clip_status_curves(None, None, predicted_action, target_action, batch_for_loss)
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
