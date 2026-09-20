#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import dataclasses
import json
import logging
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler, StepGroupedStrategyBatchSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_MISMATCH_ATTENTION_MASK,
    OBS_LANGUAGE_MISMATCH_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

STEP_PROFILE_METRIC_SPECS = (
    ("train_step_s", "step_s", "step"),
    ("dataloader_next_s", "data_s", "step"),
    ("preprocess_s", "prep_s", "step"),
    ("update_s", "updt_s", "step"),
    ("forward_s", "fwd_s", "update"),
    ("backward_s", "bwd_s", "update"),
    ("grad_clip_s", "clip_s", "update"),
    ("optimizer_step_s", "opt_s", "update"),
    ("zero_grad_s", "zero_s", "update"),
    ("debug_hook_s", "dbg_s", "update"),
    ("scheduler_s", "sch_s", "update"),
    ("policy_update_s", "pupd_s", "update"),
    ("update_misc_s", "umisc_s", "update"),
    ("train_step_misc_s", "smisc_s", "step"),
)

STEP_PROFILE_UPDATE_COMPONENT_KEYS = (
    "forward_s",
    "backward_s",
    "grad_clip_s",
    "optimizer_step_s",
    "zero_grad_s",
    "debug_hook_s",
    "scheduler_s",
    "policy_update_s",
)

STEP_PROFILE_RANK_AGG_KEYS = (
    "train_step_s",
    "dataloader_next_s",
    "preprocess_s",
    "update_s",
    "forward_s",
    "backward_s",
)


def _to_python_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _default_gripper_action_indices(action_dim: int) -> list[int]:
    if action_dim == 14:
        return [6, 13]
    if action_dim == 7:
        return [6]
    return []


def _resolve_gripper_action_indices(policy_cfg: Any, action_dim: int) -> list[int]:
    configured = list(getattr(policy_cfg, "gripper_action_indices", []) or [])
    if configured:
        return [int(idx) for idx in configured if 0 <= int(idx) < action_dim]
    return _default_gripper_action_indices(action_dim)


def _stack_action_column(action_column: Any) -> torch.Tensor:
    return torch.stack([action if isinstance(action, torch.Tensor) else torch.as_tensor(action) for action in action_column])


def _build_relative_episode_ranges(episode_ids: list[Any]) -> list[tuple[int, int]]:
    if len(episode_ids) == 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    prev_episode_id = _to_python_int(episode_ids[0])
    for idx in range(1, len(episode_ids)):
        episode_id = _to_python_int(episode_ids[idx])
        if episode_id != prev_episode_id:
            ranges.append((start, idx))
            start = idx
            prev_episode_id = episode_id
    ranges.append((start, len(episode_ids)))
    return ranges


def _build_valid_sampling_weights(
    num_frames: int,
    episode_ranges: list[tuple[int, int]],
    *,
    drop_n_first_frames: int,
    drop_n_last_frames: int,
) -> torch.Tensor:
    weights = torch.zeros(num_frames, dtype=torch.double)
    for start, end in episode_ranges:
        valid_start = min(end, start + drop_n_first_frames)
        valid_end = max(valid_start, end - drop_n_last_frames)
        if valid_end > valid_start:
            weights[valid_start:valid_end] = 1.0
    return weights


def _get_optimizer_lr_stats(optimizer: Optimizer) -> tuple[float, dict[str, float]]:
    base_lr = float(optimizer.param_groups[0]["lr"])
    named_group_lrs: dict[str, float] = {}

    for param_group in optimizer.param_groups:
        group_name = param_group.get("name")
        if isinstance(group_name, str) and group_name:
            named_group_lrs[group_name] = float(param_group["lr"])

    if "other" in named_group_lrs:
        base_lr = named_group_lrs["other"]

    extra_group_lrs = {name: lr for name, lr in named_group_lrs.items() if name != "other"}
    return base_lr, extra_group_lrs


def _step_profile_mark(accelerator: Accelerator, sync_cuda: bool) -> float:
    if sync_cuda and accelerator.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    return time.perf_counter()


def _make_step_profile_meters() -> dict[str, AverageMeter]:
    return {metric_name: AverageMeter(display_name, ":.4f") for metric_name, display_name, _ in STEP_PROFILE_METRIC_SPECS}


def _make_rank_profile_meters() -> dict[str, dict[str, AverageMeter]]:
    return {
        metric_name: {
            "min": AverageMeter(f"{metric_name}_rank_min", ":.4f"),
            "mean": AverageMeter(f"{metric_name}_rank_mean", ":.4f"),
            "max": AverageMeter(f"{metric_name}_rank_max", ":.4f"),
            "spread": AverageMeter(f"{metric_name}_rank_spread", ":.4f"),
        }
        for metric_name in STEP_PROFILE_RANK_AGG_KEYS
    }


def _gather_rank_profile_stats(
    accelerator: Accelerator,
    step_profile_metrics: dict[str, float],
) -> dict[str, dict[str, float]] | None:
    metric_values = torch.tensor(
        [float(step_profile_metrics.get(metric_name, 0.0)) for metric_name in STEP_PROFILE_RANK_AGG_KEYS],
        device=accelerator.device,
        dtype=torch.float64,
    )
    gathered_values = accelerator.gather(metric_values)
    gathered_values = gathered_values.view(accelerator.num_processes, len(STEP_PROFILE_RANK_AGG_KEYS)).cpu()
    if not accelerator.is_main_process:
        return None

    rank_profile_stats: dict[str, dict[str, float]] = {}
    for metric_idx, metric_name in enumerate(STEP_PROFILE_RANK_AGG_KEYS):
        per_rank_values = gathered_values[:, metric_idx]
        min_value = torch.min(per_rank_values).item()
        max_value = torch.max(per_rank_values).item()
        mean_value = torch.mean(per_rank_values).item()
        rank_profile_stats[metric_name] = {
            "min": float(min_value),
            "mean": float(mean_value),
            "max": float(max_value),
            "spread": float(max_value - min_value),
        }
    return rank_profile_stats


def _build_step_profile_summary(
    profile_meters: dict[str, AverageMeter],
    *,
    step_start: int,
    step_end: int,
    rank_profile_meters: dict[str, dict[str, AverageMeter]] | None = None,
) -> tuple[str, dict[str, Any], dict[str, float | int]]:
    avg_seconds = {metric_name: meter.avg for metric_name, meter in profile_meters.items() if meter.count > 0}
    profiled_steps = int(next(iter(profile_meters.values())).count) if profile_meters else 0
    step_total = max(avg_seconds.get("train_step_s", 0.0), 1e-12)
    update_total = max(avg_seconds.get("update_s", 0.0), 1e-12)

    stage_parts = [f"step={step_total * 1000.0:.1f}ms"]
    percent_of_step: dict[str, float] = {}
    percent_of_update: dict[str, float] = {}
    for metric_name, _, denominator in STEP_PROFILE_METRIC_SPECS:
        if metric_name == "train_step_s" or metric_name not in avg_seconds:
            continue
        value = avg_seconds[metric_name]
        if denominator == "step":
            percent = value / step_total * 100.0
            percent_of_step[metric_name] = percent
        else:
            percent = value / update_total * 100.0
            percent_of_update[metric_name] = percent
        short_name = metric_name.removesuffix("_s")
        stage_parts.append(f"{short_name}={value * 1000.0:.1f}ms ({percent:.1f}%)")

    summary_payload = {
        "step_start": step_start,
        "step_end": step_end,
        "profiled_steps": profiled_steps,
        "avg_seconds": avg_seconds,
        "percent_of_step": percent_of_step,
        "percent_of_update": percent_of_update,
    }
    rank_avg_seconds: dict[str, dict[str, float]] = {}
    if rank_profile_meters is not None:
        for metric_name, stat_meters in rank_profile_meters.items():
            if stat_meters["mean"].count == 0:
                continue
            rank_avg_seconds[metric_name] = {
                stat_name: meter.avg for stat_name, meter in stat_meters.items() if meter.count > 0
            }
        if rank_avg_seconds:
            if "dataloader_next_s" in rank_avg_seconds:
                data_rank_stats = rank_avg_seconds["dataloader_next_s"]
                stage_parts.append(
                    "rank_data[min/mean/max]="
                    f"{data_rank_stats['min'] * 1000.0:.1f}/"
                    f"{data_rank_stats['mean'] * 1000.0:.1f}/"
                    f"{data_rank_stats['max'] * 1000.0:.1f}ms"
                )
            if "backward_s" in rank_avg_seconds:
                backward_rank_stats = rank_avg_seconds["backward_s"]
                stage_parts.append(
                    "rank_bwd[min/mean/max]="
                    f"{backward_rank_stats['min'] * 1000.0:.1f}/"
                    f"{backward_rank_stats['mean'] * 1000.0:.1f}/"
                    f"{backward_rank_stats['max'] * 1000.0:.1f}ms"
                )
            summary_payload["rank_avg_seconds"] = rank_avg_seconds
    summary_line = (
        f"Step profile avg [{step_start}, {step_end}] over {profiled_steps} steps: " + " | ".join(stage_parts)
    )
    wandb_payload: dict[str, float | int] = {
        "profile/profiled_steps": profiled_steps,
        "profile/step_start": step_start,
        "profile/step_end": step_end,
    }
    wandb_payload.update({f"profile/{metric_name}": value for metric_name, value in avg_seconds.items()})
    wandb_payload.update({f"profile/percent_of_step/{metric_name}": value for metric_name, value in percent_of_step.items()})
    wandb_payload.update(
        {f"profile/percent_of_update/{metric_name}": value for metric_name, value in percent_of_update.items()}
    )
    if rank_avg_seconds:
        for metric_name, stats in rank_avg_seconds.items():
            for stat_name, value in stats.items():
                wandb_payload[f"profile/rank/{metric_name}/{stat_name}"] = value
    return summary_line, summary_payload, wandb_payload


def _write_step_profile_summary(profile_path: Path, summary_payload: dict[str, Any]) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")


def _build_local_gripper_change_centers(
    actions: torch.Tensor,
    *,
    gripper_indices: list[int],
    threshold: float,
    episode_ranges: list[tuple[int, int]],
) -> set[int]:
    valid_gripper_indices = [idx for idx in gripper_indices if 0 <= idx < actions.shape[-1]]
    if not valid_gripper_indices:
        return set()

    change_centers: set[int] = set()
    for start, end in episode_ranges:
        if end - start <= 1:
            continue
        gripper_actions = actions[start:end, valid_gripper_indices].float()
        if gripper_actions.ndim == 1:
            gripper_actions = gripper_actions.unsqueeze(-1)
        diffs = torch.abs(gripper_actions[1:] - gripper_actions[:-1]).amax(dim=-1)
        change_offsets = torch.nonzero(diffs >= threshold, as_tuple=False).flatten().tolist()
        for change_offset in change_offsets:
            change_centers.add(start + int(change_offset) + 1)
    return change_centers


def _build_local_gripper_transition_indices(
    dataset: LeRobotDataset,
    *,
    gripper_indices: list[int],
    threshold: float,
    window_before: int,
    min_before: int,
    window_after: int,
) -> set[int]:
    if len(dataset) == 0 or "action" not in dataset.hf_dataset.column_names:
        return set()

    action_column = dataset.hf_dataset["action"]
    actions = _stack_action_column(action_column)
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)

    episode_ranges = _build_relative_episode_ranges(dataset.hf_dataset["episode_index"])
    change_centers = _build_local_gripper_change_centers(
        actions,
        gripper_indices=gripper_indices,
        threshold=threshold,
        episode_ranges=episode_ranges,
    )
    transition_indices: set[int] = set()
    for start, end in episode_ranges:
        local_centers = sorted(idx for idx in change_centers if start <= idx < end)
        for center in local_centers:
            left = max(start, center - window_before)
            right = min(end - 1, center + window_after)
            if min_before > 0:
                right = min(right, center - min_before)
            transition_indices.update(range(left, right + 1))

    return transition_indices


def _configure_single_dataset_gripper_transition_focus(
    dataset: LeRobotDataset,
    policy_cfg: Any,
) -> tuple[torch.Tensor, int]:
    extra_aug_repeats = int(getattr(policy_cfg, "gripper_transition_extra_aug_repeats", 0))
    drop_n_first_frames = int(getattr(policy_cfg, "drop_n_first_frames", 0))
    drop_n_last_frames = int(getattr(policy_cfg, "drop_n_last_frames", 0))
    oversample_weight = float(getattr(policy_cfg, "gripper_transition_oversample_weight", 1.0))
    threshold = float(getattr(policy_cfg, "gripper_transition_threshold", 1e-3))
    window_before = int(getattr(policy_cfg, "gripper_transition_window_before", 2))
    min_before = int(getattr(policy_cfg, "gripper_transition_min_before", 0))
    window_after = int(getattr(policy_cfg, "gripper_transition_window_after", 2))

    episode_ranges = _build_relative_episode_ranges(dataset.hf_dataset["episode_index"])
    weights = _build_valid_sampling_weights(
        len(dataset),
        episode_ranges,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
    )

    action_column = dataset.hf_dataset["action"]
    if len(action_column) == 0:
        dataset.gripper_transition_indices = set()
        dataset.gripper_transition_extra_aug_repeats = extra_aug_repeats
        return weights, 0

    first_action = action_column[0]
    first_action = first_action if isinstance(first_action, torch.Tensor) else torch.as_tensor(first_action)
    action_dim = int(first_action.shape[-1]) if first_action.ndim > 0 else 1
    gripper_indices = _resolve_gripper_action_indices(policy_cfg, action_dim)
    local_transition_indices = _build_local_gripper_transition_indices(
        dataset,
        gripper_indices=gripper_indices,
        threshold=threshold,
        window_before=window_before,
        min_before=min_before,
        window_after=window_after,
    )
    dataset.gripper_transition_indices = local_transition_indices
    dataset.gripper_transition_extra_aug_repeats = extra_aug_repeats

    if oversample_weight > 1.0 and local_transition_indices:
        valid_transition_indices = [idx for idx in local_transition_indices if weights[idx] > 0]
        if valid_transition_indices:
            weights[torch.tensor(valid_transition_indices, dtype=torch.long)] *= oversample_weight

    return weights, len(local_transition_indices)


def _configure_gripper_transition_focus(
    dataset: LeRobotDataset | MultiLeRobotDataset,
    policy_cfg: Any,
) -> tuple[torch.Tensor | None, int]:
    oversample_weight = float(getattr(policy_cfg, "gripper_transition_oversample_weight", 1.0))
    extra_aug_repeats = int(getattr(policy_cfg, "gripper_transition_extra_aug_repeats", 0))
    if oversample_weight <= 1.0 and extra_aug_repeats <= 0:
        return None, 0

    if isinstance(dataset, MultiLeRobotDataset):
        combined_weights = torch.zeros(len(dataset), dtype=torch.double)
        transition_count = 0
        offset = 0
        for sub_dataset in dataset._datasets:
            local_weights, local_transition_count = _configure_single_dataset_gripper_transition_focus(
                sub_dataset, policy_cfg
            )
            combined_weights[offset : offset + len(sub_dataset)] = local_weights
            transition_count += local_transition_count
            offset += len(sub_dataset)
        if oversample_weight > 1.0 and transition_count > 0:
            return combined_weights, transition_count
        return None, transition_count

    local_weights, transition_count = _configure_single_dataset_gripper_transition_focus(dataset, policy_cfg)
    if oversample_weight > 1.0 and transition_count > 0:
        return local_weights, transition_count
    return None, transition_count


def _resolve_dataset_and_local_idx(
    dataset: LeRobotDataset | MultiLeRobotDataset, absolute_idx: int
) -> tuple[LeRobotDataset, int, int | None]:
    if isinstance(dataset, MultiLeRobotDataset):
        start_idx = 0
        for dataset_idx, sub_dataset in enumerate(dataset._datasets):
            end_idx = start_idx + len(sub_dataset)
            if absolute_idx < end_idx:
                return sub_dataset, absolute_idx - start_idx, dataset_idx
            start_idx = end_idx
        raise IndexError(f"Index {absolute_idx} out of bounds for concatenated dataset")
    return dataset, absolute_idx, None


def _collect_transition_absolute_indices(dataset: LeRobotDataset | MultiLeRobotDataset) -> list[int]:
    if isinstance(dataset, MultiLeRobotDataset):
        indices: list[int] = []
        offset = 0
        for sub_dataset in dataset._datasets:
            indices.extend(offset + idx for idx in sorted(sub_dataset.gripper_transition_indices))
            offset += len(sub_dataset)
        return indices
    return sorted(dataset.gripper_transition_indices)


def _select_debug_indices(indices: list[int], max_samples: int) -> list[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return indices

    positions = torch.linspace(0, len(indices) - 1, max_samples).round().to(torch.int64).tolist()
    selected: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        idx = indices[int(pos)]
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    if len(selected) < max_samples:
        for idx in indices:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= max_samples:
                    break
    return selected


def _extract_visual_debug_image(image_tensor: torch.Tensor) -> torch.Tensor:
    image = image_tensor.detach().float().cpu()
    while image.ndim > 3:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims after squeeze, got shape {tuple(image.shape)}")
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    return image.clamp(0.0, 1.0)


def _save_gripper_transition_image_png(
    image_tensor: torch.Tensor,
    out_path: Path,
    *,
    camera_key: str,
    absolute_idx: int,
    local_idx: int,
    dataset_index: int | None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = _extract_visual_debug_image(image_tensor)
    image_np = (image.numpy() * 255.0).round().astype("uint8")
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    width, height = img.size
    header_lines = [
        f"camera={camera_key}",
        f"abs_idx={absolute_idx}",
        f"local_idx={local_idx}",
        f"dataset_index={dataset_index}" if dataset_index is not None else "dataset_index=NA",
    ]
    draw.rectangle([0, 0, width - 1, min(height - 1, 36 + 14 * len(header_lines))], fill=(255, 255, 255))
    for line_idx, text in enumerate(header_lines):
        draw.text((8, 8 + line_idx * 12), text, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _save_gripper_transition_action_curve_png(
    actions: torch.Tensor,
    out_path: Path,
    *,
    gripper_indices: list[int],
    sample_local_idx: int,
    change_centers: set[int],
    plot_start: int,
    plot_end: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_actions = actions[plot_start:plot_end].detach().float().cpu()
    xs = list(range(plot_start, plot_end))
    fig, ax = plt.subplots(figsize=(14, 6))
    valid_gripper_indices = [dim for dim in gripper_indices if 0 <= dim < window_actions.shape[-1]]
    if not valid_gripper_indices:
        plt.close(fig)
        return
    for dim in valid_gripper_indices:
        ax.plot(
            xs,
            window_actions[:, dim].numpy(),
            linewidth=2.0,
            alpha=0.95,
            label=f"gripper_d{dim}",
        )
    for center in sorted(change_centers):
        if plot_start <= center < plot_end:
            ax.axvline(center, color="red", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.axvline(sample_local_idx, color="black", linestyle="-", linewidth=2.0, alpha=0.9, label="sample")
    prev_diff_parts: list[str] = []
    if sample_local_idx > 0:
        prev_actions = actions[sample_local_idx - 1].detach().float().cpu()
        curr_actions = actions[sample_local_idx].detach().float().cpu()
        for dim in valid_gripper_indices:
            prev_diff = float(torch.abs(curr_actions[dim] - prev_actions[dim]).item())
            prev_diff_parts.append(f"d{dim}={prev_diff:.3f}")
    else:
        prev_diff_parts.append("sample_is_first_frame")
    center_hit = "yes" if sample_local_idx in change_centers else "no"
    prev_diff_text = ", ".join(prev_diff_parts) if prev_diff_parts else "NA"
    ax.set_title(
        f"Action curve from local_idx={sample_local_idx}\n"
        f"prev_diff: {prev_diff_text}, center_hit={center_hit}, future_steps={plot_end - plot_start - 1}"
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("action value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dump_gripper_transition_focus_debug(
    dataset: LeRobotDataset | MultiLeRobotDataset,
    policy_cfg: Any,
    output_dir: str | Path,
) -> Path | None:
    max_samples = int(getattr(policy_cfg, "gripper_transition_debug_max_samples", 24))
    if max_samples <= 0:
        return None

    all_transition_indices = _collect_transition_absolute_indices(dataset)
    if not all_transition_indices:
        return None

    dump_dir = getattr(policy_cfg, "gripper_transition_debug_dump_dir", None)
    root = (
        Path(dump_dir)
        if dump_dir is not None and str(dump_dir).strip() != ""
        else Path(output_dir) / "gripper_transition_focus"
    )
    root.mkdir(parents=True, exist_ok=True)

    selected_indices = _select_debug_indices(all_transition_indices, max_samples)
    summary_lines = [
        f"transition_samples={len(all_transition_indices)}",
        f"dumped_samples={len(selected_indices)}",
        f"window_before={getattr(policy_cfg, 'gripper_transition_window_before', 0)}",
        f"min_before={getattr(policy_cfg, 'gripper_transition_min_before', 0)}",
        f"window_after={getattr(policy_cfg, 'gripper_transition_window_after', 0)}",
        f"threshold={getattr(policy_cfg, 'gripper_transition_threshold', 0.0)}",
        f"debug_future_steps={getattr(policy_cfg, 'gripper_transition_debug_future_steps', 50)}",
    ]

    for sample_rank, absolute_idx in enumerate(selected_indices):
        single_dataset, local_idx, dataset_index = _resolve_dataset_and_local_idx(dataset, absolute_idx)
        action_column = single_dataset.hf_dataset["action"]
        if len(action_column) == 0:
            continue
        actions = _stack_action_column(action_column)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        gripper_indices = _resolve_gripper_action_indices(policy_cfg, int(actions.shape[-1]))
        episode_ranges = _build_relative_episode_ranges(single_dataset.hf_dataset["episode_index"])
        change_centers = _build_local_gripper_change_centers(
            actions,
            gripper_indices=gripper_indices,
            threshold=float(getattr(policy_cfg, "gripper_transition_threshold", 1e-3)),
            episode_ranges=episode_ranges,
        )

        image_transforms = single_dataset.image_transforms
        single_dataset.image_transforms = None
        try:
            item = single_dataset[local_idx]
        finally:
            single_dataset.image_transforms = image_transforms

        sample_dir = root / f"sample_{sample_rank:03d}_abs_{absolute_idx:09d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for camera_key in single_dataset.meta.camera_keys:
            if camera_key not in item:
                continue
            _save_gripper_transition_image_png(
                item[camera_key],
                sample_dir / f"{camera_key.replace('.', '_')}.png",
                camera_key=camera_key,
                absolute_idx=absolute_idx,
                local_idx=local_idx,
                dataset_index=dataset_index,
            )

        plot_after = max(int(getattr(policy_cfg, "gripper_transition_debug_future_steps", 50)), 1)
        plot_start = local_idx
        plot_end = min(actions.shape[0], local_idx + plot_after + 1)
        _save_gripper_transition_action_curve_png(
            actions,
            sample_dir / "action_curve.png",
            gripper_indices=gripper_indices,
            sample_local_idx=local_idx,
            change_centers=change_centers,
            plot_start=plot_start,
            plot_end=plot_end,
        )
        summary_lines.append(
            f"sample_{sample_rank:03d}: abs_idx={absolute_idx}, local_idx={local_idx}, dataset_index={dataset_index}"
        )

    (root / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return root


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
    language_mismatch_cfg: dict[str, Any] | None = None,
    step_profile: dict[str, float] | None = None,
    profile_cuda_sync: bool = False,
) -> tuple[MetricsTracker, dict, dict[str, float] | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    timing_mark = (
        (lambda: _step_profile_mark(accelerator, profile_cuda_sync)) if step_profile is not None else time.perf_counter
    )
    start_time = timing_mark()
    policy.train()
    profile_metrics = None
    if step_profile is not None:
        profile_metrics = {metric_name: 0.0 for metric_name in STEP_PROFILE_UPDATE_COMPONENT_KEYS}
        profile_metrics["update_misc_s"] = 0.0

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    use_language_mismatch = bool(
        language_mismatch_cfg
        and language_mismatch_cfg.get("enabled", False)
        and language_mismatch_cfg.get("ratio", 0.0) > 0
        and OBS_LANGUAGE_TOKENS in batch
        and OBS_LANGUAGE_ATTENTION_MASK in batch
        and OBS_LANGUAGE_MISMATCH_TOKENS in batch
        and OBS_LANGUAGE_MISMATCH_ATTENTION_MASK in batch
    )

    # Let accelerator handle mixed precision
    forward_start_time = timing_mark()
    with accelerator.autocast():
        if rabc_batch_weights is not None or use_language_mismatch:
            epsilon = 1e-6
            if use_language_mismatch:
                batch_size = int(batch[OBS_LANGUAGE_TOKENS].shape[0])
                ratio = float(language_mismatch_cfg["ratio"])
                mismatch_mode = str(language_mismatch_cfg.get("mode", "action_rank"))
                num_src = min(batch_size, max(1, int(batch_size * ratio)))
                src_idx = torch.randperm(batch_size, device=batch[OBS_LANGUAGE_TOKENS].device)[:num_src]
                src_idx_list = src_idx.detach().cpu().tolist()
                per_sample_loss, output_dict = policy.forward(batch, reduction="none", enable_train_rollout_debug=True)
                pos_action_per_sample_loss = output_dict.pop("_action_per_sample_loss", per_sample_loss)

                mismatch_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                        mismatch_batch[key] = value[src_idx]
                    elif isinstance(value, list) and len(value) == batch_size:
                        mismatch_batch[key] = [value[i] for i in src_idx_list]
                    elif isinstance(value, tuple) and len(value) == batch_size:
                        mismatch_batch[key] = [value[i] for i in src_idx_list]
                    else:
                        mismatch_batch[key] = value

                mismatch_batch[OBS_LANGUAGE_TOKENS] = batch[OBS_LANGUAGE_MISMATCH_TOKENS][src_idx]
                mismatch_batch[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_MISMATCH_ATTENTION_MASK][src_idx]
                mismatch_task_value = batch.get("task_mismatch")
                if isinstance(mismatch_task_value, tuple):
                    mismatch_task_value = list(mismatch_task_value)
                if isinstance(mismatch_task_value, list) and len(mismatch_task_value) == batch_size:
                    mismatch_batch["task"] = [mismatch_task_value[i] for i in src_idx_list]

                mismatch_per_sample_loss, mismatch_output_dict = policy.forward(
                    mismatch_batch, reduction="none", enable_train_rollout_debug=False
                )
                if mismatch_mode == "action_rank":
                    neg_selected_action_per_sample_loss = mismatch_output_dict.pop(
                        "_action_per_sample_loss", mismatch_per_sample_loss
                    )
                else:
                    mismatch_output_dict.pop("_action_per_sample_loss", None)
            else:
                per_sample_loss, output_dict = policy.forward(batch, reduction="none", enable_train_rollout_debug=True)
                output_dict.pop("_action_per_sample_loss", None)

            if rabc_batch_weights is not None:
                loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
                output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
                output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
                output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
            else:
                loss = per_sample_loss.mean()
            if use_language_mismatch:
                reg_weight = float(language_mismatch_cfg.get("weight", 0.3))
                if mismatch_mode == "action_rank":
                    margin = float(language_mismatch_cfg.get("margin", 0.05))
                    rank_loss_per_sample = torch.relu(
                        margin + pos_action_per_sample_loss[src_idx] - neg_selected_action_per_sample_loss
                    )
                    if rabc_batch_weights is not None:
                        selected_weights = rabc_batch_weights[src_idx]
                        mismatch_loss = (rank_loss_per_sample * selected_weights).sum() / (
                            selected_weights.sum() + epsilon
                        )
                    else:
                        mismatch_loss = rank_loss_per_sample.mean()
                    output_dict["lang_mismatch_rank_violation"] = (
                        rank_loss_per_sample.gt(0).float().mean().item()
                    )
                elif mismatch_mode == "task_id_not_correct":
                    mismatch_task_id_logits = mismatch_output_dict.pop("_task_id_logits", None)
                    if mismatch_task_id_logits is None:
                        raise RuntimeError("Language mismatch mode `task_id_not_correct` requires task_id logits.")
                    correct_task_id = mismatch_batch["task_id"].to(device=mismatch_task_id_logits.device, dtype=torch.long)
                    mismatch_probs = torch.softmax(mismatch_task_id_logits.float(), dim=-1)
                    correct_task_prob = mismatch_probs.gather(1, correct_task_id[:, None]).squeeze(1)
                    pred_task_id = mismatch_probs.argmax(dim=-1)
                    not_correct_loss_per_sample = -torch.log1p(-(correct_task_prob.clamp(min=0.0, max=1.0 - 1e-6)))
                    if rabc_batch_weights is not None:
                        selected_weights = rabc_batch_weights[src_idx]
                        mismatch_loss = (not_correct_loss_per_sample * selected_weights).sum() / (
                            selected_weights.sum() + epsilon
                        )
                    else:
                        mismatch_loss = not_correct_loss_per_sample.mean()
                    output_dict["lang_mismatch_correct_task_prob"] = correct_task_prob.mean().item()
                    output_dict["lang_mismatch_not_correct_rate"] = (
                        pred_task_id.ne(correct_task_id).float().mean().item()
                    )
                else:
                    raise ValueError(f"Unsupported language mismatch mode: {mismatch_mode}")
                loss = loss + reg_weight * mismatch_loss
                output_dict["lang_mismatch_loss"] = mismatch_loss.item()
                output_dict["lang_mismatch_pairs"] = int(src_idx.numel())
                output_dict["lang_mismatch_mode"] = mismatch_mode
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    if profile_metrics is not None:
        profile_metrics["forward_s"] = timing_mark() - forward_start_time

    # Use accelerator's backward method
    backward_start_time = timing_mark()
    accelerator.backward(loss)
    if profile_metrics is not None:
        profile_metrics["backward_s"] = timing_mark() - backward_start_time

    # Clip gradients if specified
    grad_clip_start_time = timing_mark()
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )
    if profile_metrics is not None:
        profile_metrics["grad_clip_s"] = timing_mark() - grad_clip_start_time

    # Optimizer step
    optimizer_step_start_time = timing_mark()
    with lock if lock is not None else nullcontext():
        optimizer.step()
    if profile_metrics is not None:
        profile_metrics["optimizer_step_s"] = timing_mark() - optimizer_step_start_time

    zero_grad_start_time = timing_mark()
    optimizer.zero_grad()
    if profile_metrics is not None:
        profile_metrics["zero_grad_s"] = timing_mark() - zero_grad_start_time

    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    debug_hook_start_time = timing_mark()
    if has_method(unwrapped_policy, "run_pending_train_rollout_debug"):
        unwrapped_policy.run_pending_train_rollout_debug()
    if profile_metrics is not None:
        profile_metrics["debug_hook_s"] = timing_mark() - debug_hook_start_time

    # Step through pytorch scheduler at every batch instead of epoch
    scheduler_start_time = timing_mark()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if profile_metrics is not None:
        profile_metrics["scheduler_s"] = timing_mark() - scheduler_start_time

    # Update internal buffers if policy has update method
    policy_update_start_time = timing_mark()
    if has_method(unwrapped_policy, "update"):
        unwrapped_policy.update()
    if profile_metrics is not None:
        profile_metrics["policy_update_s"] = timing_mark() - policy_update_start_time

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    base_lr, extra_group_lrs = _get_optimizer_lr_stats(optimizer)
    train_metrics.lr = base_lr
    for group_name, group_lr in extra_group_lrs.items():
        metric_name = f"{group_name}_lr"
        if metric_name in train_metrics.metrics:
            setattr(train_metrics, metric_name, group_lr)
    train_metrics.update_s = timing_mark() - start_time
    if profile_metrics is not None:
        profile_metrics["update_s"] = train_metrics.update_s.val
        measured_update_s = sum(profile_metrics[key] for key in STEP_PROFILE_UPDATE_COMPONENT_KEYS)
        profile_metrics["update_misc_s"] = max(profile_metrics["update_s"] - measured_update_s, 0.0)
    return train_metrics, output_dict, profile_metrics


def get_default_peft_configuration(policy_type):
    """Build a basic PEFT configuration for the given policy type assuming that we train a policy from a checkpoint."""

    common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"

    if policy_type == "smolvla":
        return {
            "target_modules": rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))",
            "modules_to_save": [],
        }
    elif policy_type in ("pi0", "pi05"):
        return {
            "target_modules": rf"(.*\.gemma_expert\..*\.self_attn.(q|v)_proj|model\.({common_projections}))",
            "modules_to_save": [],
        }

    return {"modules_to_save": None}


def wrap_policy_in_peft_model(cfg, policy):
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType, get_peft_model

    # Disable all gradients because we'll only train the parameters selected by the PEFT method.
    # Layers that should receive gradients anyway need to be listed in `modules_to_save`.
    for p in policy.parameters():
        p.requires_grad_(False)

    if not cfg.policy.pretrained_path:
        raise ValueError(
            "Training from scratch using PEFT. This is unlikely to yield good results. "
            "Supply a `policy.path` to fine-tune an existing model."
        )

    if cfg.policy.type == "smolvla" and not cfg.policy.load_vlm_weights:
        logging.warning(
            "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. Set "
            "`load_vlm_weights=True` to fine-tune the existing policy."
        )

    peft_config_policy = get_default_peft_configuration(cfg.policy.type)
    peft_config_cli = dataclasses.asdict(cfg.peft) if cfg.peft else {}
    peft_config_cli["modules_to_save"] = peft_config_cli["full_training_modules"]  # compatibility with PEFT
    peft_method_type = PeftType[peft_config_cli["method_type"].upper()]
    peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

    # Handle specific CLI overrides
    for key in ["target_modules", "modules_to_save", "r"]:
        if peft_config_cli[key] is not None:
            peft_config_policy[key] = peft_config_cli[key]

    if "target_modules" not in peft_config_policy:
        raise ValueError(
            f"There is no default `target_modules` value for policy {cfg.policy.type}. Please pass it manually."
        )

    # Init method depends on the used PEFT method, your specific PEFT method
    # might not be considered here, in that case an error is raised.
    if peft_config_cli["init_type"] is not None:
        if peft_method_type == "LORA":
            peft_config_policy["init_lora_weights"] = peft_config_cli["init_type"]
        elif peft_method_type == "MISS":
            peft_config_policy["init_weights"] = peft_config_cli["init_type"]
        else:
            raise ValueError(
                f"Init type {peft_config_cli['init_type']} unknown for PEFT method {peft_method_type}."
            )

    # PEFT uses this attribute to set adapter_config.base_name_or_path which we use for loading the
    # correct base model in `make_policy` since in a PEFT loading setting we only get the path to the
    # adapter, not the base model.
    if policy.config.pretrained_path:
        policy.name_or_path = str(policy.config.pretrained_path)

    # Finally wrap the policy in a PEFT model
    policy = get_peft_model(
        policy,
        peft_config_cls(**peft_config_policy),
    )

    # Make sure that the config is tagged as using PEFT so that the loading code can take the
    # appropriate steps to use the adapter weights and the PEFT config instead of the full model weights.
    policy.config.use_peft = True

    return policy


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs

        ddp_find_unused_parameters = os.getenv("DDP_FIND_UNUSED_PARAMETERS", "true").lower() == "true"
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused_parameters)
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=10))
        dataloader_config = None
        if cfg.dataset.streaming:
            # Streaming datasets may yield non-tensor metadata such as task strings.
            # Let each rank fetch its own shard instead of having accelerate concatenate
            # batches on the main process, which only supports tensor-like leaves.
            dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs, process_group_kwargs],
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    cfg.validate()

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        policy = wrap_policy_in_peft_model(cfg, policy)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type in {"sarm", "pi05"}:
        processor_kwargs["dataset_meta"] = dataset.meta
    if cfg.policy.type == "pi05":
        task_pool = None
        if hasattr(dataset.meta, "tasks") and dataset.meta.tasks is not None:
            task_pool = list(dataset.meta.tasks.index)
        elif hasattr(dataset, "_datasets"):
            all_tasks = set()
            for sub_dataset in dataset._datasets:
                if hasattr(sub_dataset.meta, "tasks") and sub_dataset.meta.tasks is not None:
                    all_tasks.update(list(sub_dataset.meta.tasks.index))
            if all_tasks:
                task_pool = sorted(all_tasks)
        if task_pool is not None:
            processor_kwargs["task_pool"] = task_pool
        if getattr(cfg.policy, "task_id_map", None):
            repo_ids = cfg.dataset.repo_id if isinstance(cfg.dataset.repo_id, list) else [cfg.dataset.repo_id]
            dataset_index_to_task_id: dict[int, int] = {}
            for dataset_index, repo_id in enumerate(repo_ids):
                if repo_id in cfg.policy.task_id_map:
                    dataset_index_to_task_id[dataset_index] = int(cfg.policy.task_id_map[repo_id])
                    continue
                repo_id_as_str = str(repo_id)
                if repo_id_as_str in cfg.policy.task_id_map:
                    dataset_index_to_task_id[dataset_index] = int(cfg.policy.task_id_map[repo_id_as_str])
                    continue
                index_key = str(dataset_index)
                if index_key in cfg.policy.task_id_map:
                    dataset_index_to_task_id[dataset_index] = int(cfg.policy.task_id_map[index_key])
            if dataset_index_to_task_id:
                processor_kwargs["dataset_index_to_task_id"] = dataset_index_to_task_id

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    if cfg.policy.type == "pi05":
        preprocessor_overrides = processor_kwargs.setdefault("preprocessor_overrides", {})
        preprocessor_overrides["pi05_prepare_state_tokenizer_processor_step"] = {
            "include_state_in_prompt": bool(cfg.policy.include_state_in_language_prompt)
        }

    processor_pretrained_path = cfg.policy.pretrained_path
    if cfg.policy.type == "pi05":
        # PI05 preprocessing depends on runtime dataset stats and task mapping kwargs,
        # and newer configs may exclude aux targets like boxes/cross_center from
        # observation normalization. Rebuild the processor to avoid stale checkpoint
        # preprocessor configs overriding these settings.
        processor_pretrained_path = None

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    camera_keys = getattr(dataset.meta, "camera_keys", [])
    total_images = dataset.num_frames * len(camera_keys)
    total_clips = dataset.num_episodes
    total_tasks = None
    if hasattr(dataset.meta, "tasks") and dataset.meta.tasks is not None:
        total_tasks = len(dataset.meta.tasks)
    elif hasattr(dataset.meta, "total_tasks"):
        total_tasks = dataset.meta.total_tasks
    elif hasattr(dataset, "_datasets"):
        task_names = set()
        for sub_dataset in dataset._datasets:
            if hasattr(sub_dataset.meta, "tasks") and sub_dataset.meta.tasks is not None:
                task_names.update(list(sub_dataset.meta.tasks.index))
        total_tasks = len(task_names)
    if total_tasks is None:
        total_tasks = 0

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"dataset.total_images={total_images}")
        logging.info(f"dataset.total_tasks={total_tasks}")
        logging.info(f"dataset.total_clips={total_clips}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    num_stm_candidates = len(getattr(cfg.policy, "observation_delta_indices_candidates", []) or [])
    transition_sampling_weights, transition_sample_count = _configure_gripper_transition_focus(dataset, cfg.policy)
    if is_main_process and transition_sample_count > 0:
        logging.info(
            "Gripper transition focus enabled: "
            f"transition_samples={transition_sample_count}, "
            f"oversample_weight={getattr(cfg.policy, 'gripper_transition_oversample_weight', 1.0)}, "
            f"extra_aug_repeats={getattr(cfg.policy, 'gripper_transition_extra_aug_repeats', 0)}"
        )
        debug_root = _dump_gripper_transition_focus_debug(dataset, cfg.policy, cfg.output_dir)
        if debug_root is not None:
            logging.info(f"Gripper transition debug dump saved to: {debug_root}")

    # create dataloader for offline training
    if transition_sampling_weights is not None:
        shuffle = False
        valid_sample_count = int((transition_sampling_weights > 0).sum().item())
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=transition_sampling_weights,
            num_samples=valid_sample_count,
            replacement=True,
        )
    elif hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    batch_sampler = None
    if not cfg.dataset.streaming and num_stm_candidates > 1:
        if sampler is None:
            sampler = torch.utils.data.RandomSampler(dataset)
            shuffle = False
        batch_sampler = StepGroupedStrategyBatchSampler(
            sampler=sampler,
            batch_size=cfg.batch_size,
            drop_last=False,
            num_strategies=num_stm_candidates,
            process_group_size=accelerator.num_processes,
            seed=cfg.seed,
        )

    prefetch_factor = int(os.environ.get("LEROBOT_DATALOADER_PREFETCH_FACTOR", "2"))
    dataloader_kwargs = dict(
        dataset=dataset,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor if cfg.num_workers > 0 else None,
    )
    if batch_sampler is not None:
        dataloader = torch.utils.data.DataLoader(
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            drop_last=False,
            **dataloader_kwargs,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()
    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
    }
    _, extra_group_lrs = _get_optimizer_lr_stats(optimizer)
    for group_name in extra_group_lrs:
        train_metrics[f"{group_name}_lr"] = AverageMeter(f"{group_name}_lr", ":0.1e")
    train_metrics["update_s"] = AverageMeter("updt_s", ":.3f")
    train_metrics["dataloading_s"] = AverageMeter("data_s", ":.3f")

    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    step_profile_enabled = bool(cfg.step_profile and cfg.step_profile_steps > 0)
    step_profile_start = max(1, cfg.step_profile_start)
    step_profile_end = min(cfg.steps, step_profile_start + cfg.step_profile_steps - 1)
    step_profile_meters: dict[str, AverageMeter] | None = None
    rank_profile_meters: dict[str, dict[str, AverageMeter]] | None = None
    step_profile_summary_path: Path | None = None
    step_profile_summary_written = False
    step_profile_last_step = 0
    if step_profile_enabled and step_profile_start <= cfg.steps:
        if is_main_process:
            step_profile_meters = _make_step_profile_meters()
            rank_profile_meters = _make_rank_profile_meters()
            step_profile_summary_path = Path(cfg.output_dir) / "step_profile_summary.json"
            logging.info(
                "Step profile enabled from step %d for %d steps (cuda_sync=%s). Summary: %s",
                step_profile_start,
                step_profile_end - step_profile_start + 1,
                cfg.step_profile_cuda_sync,
                step_profile_summary_path,
            )
    else:
        step_profile_enabled = False

    for _ in range(step, cfg.steps):
        current_step = step + 1
        profile_this_step = step_profile_enabled and step_profile_start <= current_step <= step_profile_end
        timing_mark = (
            (lambda: _step_profile_mark(accelerator, cfg.step_profile_cuda_sync))
            if profile_this_step
            else time.perf_counter
        )
        start_time = timing_mark()
        batch = next(dl_iter)
        dataloader_next_s = timing_mark() - start_time
        preprocess_start_time = timing_mark()
        batch = preprocessor(batch)
        preprocess_s = timing_mark() - preprocess_start_time
        train_tracker.dataloading_s = dataloader_next_s + preprocess_s
        step_profile_metrics = None
        if profile_this_step:
            step_profile_metrics = {
                "dataloader_next_s": dataloader_next_s,
                "preprocess_s": preprocess_s,
            }

        train_tracker, output_dict, update_profile_metrics = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
            language_mismatch_cfg={
                "enabled": cfg.use_language_mismatch_regularization,
                "ratio": cfg.language_mismatch_ratio,
                "mode": cfg.language_mismatch_mode,
                "margin": cfg.language_mismatch_margin,
                "weight": cfg.language_mismatch_weight,
            },
            step_profile=step_profile_metrics,
            profile_cuda_sync=cfg.step_profile_cuda_sync and profile_this_step,
        )
        if profile_this_step and step_profile_meters is not None and step_profile_metrics is not None:
            if update_profile_metrics is not None:
                step_profile_metrics.update(update_profile_metrics)
            step_profile_metrics["train_step_s"] = timing_mark() - start_time
            step_profile_metrics["train_step_misc_s"] = max(
                step_profile_metrics["train_step_s"]
                - step_profile_metrics["dataloader_next_s"]
                - step_profile_metrics["preprocess_s"]
                - step_profile_metrics["update_s"],
                0.0,
            )
            rank_profile_stats = _gather_rank_profile_stats(accelerator, step_profile_metrics)
            for metric_name, meter in step_profile_meters.items():
                meter.update(float(step_profile_metrics.get(metric_name, 0.0)))
            if rank_profile_meters is not None and rank_profile_stats is not None:
                for metric_name, stats in rank_profile_stats.items():
                    if metric_name not in rank_profile_meters:
                        continue
                    for stat_name, value in stats.items():
                        if stat_name in rank_profile_meters[metric_name]:
                            rank_profile_meters[metric_name][stat_name].update(float(value))
            step_profile_last_step = current_step
        elif profile_this_step and step_profile_metrics is not None:
            if update_profile_metrics is not None:
                step_profile_metrics.update(update_profile_metrics)
            step_profile_metrics["train_step_s"] = timing_mark() - start_time
            step_profile_metrics["train_step_misc_s"] = max(
                step_profile_metrics["train_step_s"]
                - step_profile_metrics["dataloader_next_s"]
                - step_profile_metrics["preprocess_s"]
                - step_profile_metrics["update_s"],
                0.0,
            )
            _gather_rank_profile_stats(accelerator, step_profile_metrics)

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if profile_this_step and step_profile_meters is not None and step == step_profile_end:
            summary_line, summary_payload, summary_wandb_payload = _build_step_profile_summary(
                step_profile_meters,
                step_start=step_profile_start,
                step_end=step_profile_last_step,
                rank_profile_meters=rank_profile_meters,
            )
            logging.info(summary_line)
            if step_profile_summary_path is not None:
                _write_step_profile_summary(step_profile_summary_path, summary_payload)
                logging.info("Step profile summary saved to: %s", step_profile_summary_path)
            if wandb_logger:
                wandb_logger.log_dict(summary_wandb_payload, step)
            step_profile_summary_written = True
            step_profile_enabled = False

        if is_log_step:
            logging.info(train_tracker)
            if output_dict:
                loss_items = {
                    k: v
                    for k, v in output_dict.items()
                    if (
                        "loss" in k.lower()
                        or k.endswith("_error_mm")
                        or k.endswith("_error_deg")
                        or k == "mse"
                        or k == "mae"
                        or k.endswith("_mse")
                        or k.endswith("_mae")
                        or k.startswith("lang_mismatch_")
                    )
                }
                if loss_items:
                    formatted_loss_items = []
                    for key in sorted(loss_items.keys()):
                        value = loss_items[key]
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                value = float(value.detach().cpu().item())
                            else:
                                value = value.detach().cpu().tolist()
                        if isinstance(value, float):
                            formatted_loss_items.append(f"{key}={value:.6f}")
                        else:
                            formatted_loss_items.append(f"{key}={value}")
                    logging.info("Loss Breakdown: %s", " | ".join(formatted_loss_items))
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    try:
                        wandb_logger.log_policy(checkpoint_dir)
                    except Exception:
                        logging.exception(
                            "Failed to upload checkpoint artifact to WandB for %s; local checkpoint is kept.",
                            checkpoint_dir,
                        )

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if (
        is_main_process
        and step_profile_meters is not None
        and not step_profile_summary_written
        and step_profile_last_step >= step_profile_start
    ):
        summary_line, summary_payload, summary_wandb_payload = _build_step_profile_summary(
            step_profile_meters,
            step_start=step_profile_start,
            step_end=step_profile_last_step,
            rank_profile_meters=rank_profile_meters,
        )
        logging.info(summary_line)
        if step_profile_summary_path is not None:
            _write_step_profile_summary(step_profile_summary_path, summary_payload)
            logging.info("Step profile summary saved to: %s", step_profile_summary_path)
        if wandb_logger:
            wandb_logger.log_dict(summary_wandb_payload, step)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
