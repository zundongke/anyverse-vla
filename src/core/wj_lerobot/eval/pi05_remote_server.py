#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PI05 remote inference server (HTTP).

Goal:
- Edge device sends images + robot state (+ optional task string)
- Server runs PI05 inference (same path as eval_pi05_dataset.py)
- Returns an action chunk (and first-step action) as JSON

Notes:
- This server intentionally stays self-contained under Anyverse-VLA.
- It relies on the LeRobot codebase being importable (recommended via PYTHONPATH).

Example:
  PYTHONPATH=/mnt/dataset/chao-liang/lerobot/src \
  python pi05_remote_server.py --host 0.0.0.0 --port 6007 \
    --model-dir /path/to/pretrained_model \
    --dataset-root /mnt/dataset/chao-liang/lerobot_datasets/piper_dual_arm \
    --dataset-repo-id piper_dual_arm

Request:
  POST /v1/predict (multipart/form-data)
    - cam_high: one or more image files
    - wrist_left: one or more image files
    - wrist_right: one or more image files
    - state: JSON list[float] length matches the loaded model state feature
    - task: (optional) string

Response:
  {
    "action": [..14..],
    "clip_running_status": [0.0,1.0],
    "action_chunk": [[..14..]*T],
    "pred_boxes": [[x1, y1, x2, y2], ...],
    "pred_cross_center": [cx, cy],
    ...
  }
"""

import argparse
import hashlib
import json
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _lazy_imports() -> dict[str, Any]:
    """Import heavy dependencies lazily to improve CLI error messages."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: torch") from exc

    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import JSONResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: fastapi (and its deps). Install with: pip install fastapi uvicorn"
        ) from exc

    # Required by FastAPI when using File/Form (multipart/form-data)
    try:
        import multipart  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            'Form data requires "python-multipart". Install with: pip install python-multipart'
        ) from exc

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pillow. Install with: pip install pillow") from exc

    try:
        # LeRobot imports
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        # 与 main 的 `factory.py` 保持一致：不修改 upstream；此处侧载 processor 以注册
        # ProcessorStepRegistry，使 `make_pre_post_processors(..., pretrained_path=...)` 能解析
        # checkpoint 里的 pre/post JSON（否则可能出现 KeyError）。
        import lerobot.policies.pi05.processor_pi05  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import lerobot. "
            "Recommended: export PYTHONPATH=/mnt/dataset/chao-liang/lerobot/src:$PYTHONPATH"
        ) from exc

    return {
        "torch": torch,
        "FastAPI": FastAPI,
        "File": File,
        "Form": Form,
        "UploadFile": UploadFile,
        "JSONResponse": JSONResponse,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
        "PreTrainedConfig": PreTrainedConfig,
        "FeatureType": FeatureType,
        "LeRobotDataset": LeRobotDataset,
        "dataset_to_policy_features": dataset_to_policy_features,
        "make_pre_post_processors": make_pre_post_processors,
        "PI05Policy": PI05Policy,
    }


def _find_local_paligemma_tokenizer() -> Path | None:
    """Return a local snapshot dir for google/paligemma-3b-pt-224 if present in HF cache."""
    candidates: list[Path] = []
    home = Path.home()
    candidates.append(home / ".cache" / "huggingface" / "hub")
    candidates.append(Path("/root/.cache/huggingface/hub"))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")

    for hub_root in candidates:
        snap_root = hub_root / "models--google--paligemma-3b-pt-224" / "snapshots"
        if snap_root.exists():
            snaps = sorted(snap_root.glob("*"))
            if snaps:
                return snaps[-1]
    return None


def _load_image_tensor(
    image_bytes: bytes, *, image_key: str, image_size: int, masker: Any = None, pad: bool = False
):
    mod = _lazy_imports()
    torch = mod["torch"]
    Image = mod["Image"]

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if masker is not None:
        masked, _ = masker.process([img])
        img = masked[0]
    if pad:
        from PIL import ImageOps

        # Match the offline converter, including its resampling and black padding.
        img = ImageOps.pad(img, (image_size, image_size), color=(0, 0, 0))
    elif img.size != (image_size, image_size):
        img = img.resize((image_size, image_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"Bad image for {image_key}: got shape {tuple(t.shape)}")

    return t


def _stack_temporal_images(image_tensors: list[Any], *, torch: Any) -> Any:
    if len(image_tensors) == 0:
        raise ValueError("At least one image is required per camera")
    if len(image_tensors) == 1:
        return image_tensors[0]
    # Explicit temporal input: [1, T, C, H, W]. This bypasses PI05Policy's
    # internal short-term-memory cache augmentation and matches training layout.
    return torch.stack(image_tensors, dim=0).unsqueeze(0)


def _project_behavior_b1k_state(state: np.ndarray) -> np.ndarray | None:
    """Project raw BEHAVIOR proprio to the 23-dim state used by Behavior-B1K training."""
    if state.ndim != 1 or state.shape[0] < 234:
        return None

    base_qvel = state[253:256]
    trunk_qpos = state[236:240]
    left_arm_qpos = state[158:165]
    left_gripper = np.asarray([state[193:195].sum()], dtype=np.float32)
    right_arm_qpos = state[197:204]
    right_gripper = np.asarray([state[232:234].sum()], dtype=np.float32)
    return np.concatenate(
        [base_qvel, trunk_qpos, left_arm_qpos, left_gripper, right_arm_qpos, right_gripper],
        axis=0,
    ).astype(np.float32, copy=False)


def _resolve_policy_image_keys(input_feature_keys: list[str]) -> tuple[str, str | None, str | None]:
    image_keys = [
        key
        for key in input_feature_keys
        if key.startswith("observation.images.") or key.startswith("observation.rgb.")
    ]
    if not image_keys:
        raise ValueError("No image features found in policy input_features.")

    if image_keys == ["observation.images.top_head"]:
        return image_keys[0], None, None

    def _pick(preferred: list[str], fuzzy_tokens: list[str], camera_name: str) -> str:
        for key in preferred:
            if key in image_keys:
                return key
        for key in image_keys:
            if all(token in key for token in fuzzy_tokens):
                return key
        raise ValueError(f"Failed to resolve {camera_name} image key from policy image features: {image_keys}")

    head_key = _pick(
        preferred=[
            "observation.images.cam_high",
            "observation.rgb.zed_link_camera_0",
        ],
        fuzzy_tokens=["zed"],
        camera_name="head",
    )
    left_key = _pick(
        preferred=[
            "observation.images.wrist_left",
            "observation.rgb.left_realsense_link_camera_0",
        ],
        fuzzy_tokens=["left", "camera"],
        camera_name="left",
    )
    right_key = _pick(
        preferred=[
            "observation.images.wrist_right",
            "observation.rgb.right_realsense_link_camera_0",
        ],
        fuzzy_tokens=["right", "camera"],
        camera_name="right",
    )
    return head_key, left_key, right_key


def _safe_upload_name(name: str | None, *, fallback: str) -> str:
    base = Path(str(name or fallback)).name.strip()
    if not base:
        base = fallback
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    return safe or fallback


def _parse_optional_int(value: str | None, *, name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    return parsed


def _parse_optional_seed(value: str | None) -> int | None:
    seed = _parse_optional_int(value, name="flow_noise_seed")
    if seed is not None and not 0 <= seed < 2**63:
        raise ValueError("flow_noise_seed must be in [0, 2**63)")
    return seed


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


@contextmanager
def _fixed_flow_noise_seed(torch: Any, *, device: str, seed: int | None):
    """Temporarily seed Torch RNGs without changing the server's normal RNG stream."""
    if seed is None:
        yield
        return

    device_obj = torch.device(device)
    cuda_devices: list[int] = []
    if device_obj.type == "cuda":
        device_index = device_obj.index
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        cuda_devices = [int(device_index)]

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_task_id_to_label(task_id_map: Any, num_task_classes: int) -> list[str]:
    size = max(0, int(num_task_classes))
    labels = [f"id_{i}" for i in range(size)]
    if not isinstance(task_id_map, dict):
        return labels
    for label, raw_idx in task_id_map.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < size:
            labels[idx] = str(label)
    return labels


def _save_line_plot(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    y_label: str,
    colors: list[tuple[int, int, int]] | None = None,
    series_labels: list[str] | None = None,
) -> None:
    mod = _lazy_imports()
    Image = mod["Image"]
    ImageDraw = mod["ImageDraw"]
    ImageFont = mod["ImageFont"]

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return

    width, height = 1280, 720
    margin_left, margin_right = 90, 30
    margin_top, margin_bottom = 60, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    palette = colors or [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0], dtype=np.float32)
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if abs(y_max - y_min) < 1e-6:
        pad = 1.0 if abs(y_max) < 1e-6 else abs(y_max) * 0.1
        y_min -= pad
        y_max += pad

    for frac in np.linspace(0.0, 1.0, 6):
        y = margin_top + int((1.0 - frac) * plot_h)
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill=(230, 230, 230), width=1)
        value = y_min + frac * (y_max - y_min)
        draw.text((10, y - 6), f"{value:.3f}", fill=(80, 80, 80), font=font)

    draw.line(
        [(margin_left, margin_top), (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h)],
        fill=(40, 40, 40),
        width=2,
    )
    draw.text((margin_left, 18), title, fill=(20, 20, 20), font=font)
    draw.text((10, 18), y_label, fill=(90, 90, 90), font=font)

    steps = arr.shape[0]
    if steps == 1:
        xs = np.array([margin_left + plot_w // 2], dtype=np.float32)
    else:
        xs = np.linspace(margin_left, margin_left + plot_w, steps, dtype=np.float32)

    def to_y(v: float) -> float:
        ratio = (float(v) - y_min) / (y_max - y_min)
        return margin_top + (1.0 - ratio) * plot_h

    for dim in range(arr.shape[1]):
        color = palette[dim % len(palette)]
        pts = [(float(xs[i]), float(to_y(arr[i, dim]))) for i in range(steps)]
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)
        else:
            draw.line(pts, fill=color, width=2)
        if series_labels and dim < len(series_labels):
            ly = margin_top + 18 * dim
            draw.text((margin_left + plot_w - 180, ly), series_labels[dim], fill=color, font=font)

    for i in range(min(steps, 8)):
        idx = 0 if steps == 1 else int(round(i * (steps - 1) / max(1, min(steps, 8) - 1)))
        x = float(xs[idx])
        draw.line([(x, margin_top + plot_h), (x, margin_top + plot_h + 6)], fill=(50, 50, 50), width=1)
        draw.text((x - 8, margin_top + plot_h + 10), str(idx), fill=(80, 80, 80), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _save_task_probability_plot(
    task_probs: list[float],
    task_labels: list[str],
    output_path: Path,
    *,
    title: str = "Task ID Probabilities",
    top_k: int = 8,
) -> None:
    mod = _lazy_imports()
    Image = mod["Image"]
    ImageDraw = mod["ImageDraw"]
    ImageFont = mod["ImageFont"]

    probs = np.asarray(task_probs, dtype=np.float32)
    if probs.ndim != 1 or probs.size == 0:
        return
    k = min(int(top_k), int(probs.size))
    top_ids = np.argsort(probs)[::-1][:k]
    top_vals = probs[top_ids]
    top_names = [task_labels[i] if 0 <= i < len(task_labels) else f"id_{i}" for i in top_ids.tolist()]

    width, height = 1280, max(420, 120 + 52 * k)
    margin_left, margin_right = 320, 80
    margin_top, margin_bottom = 60, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    row_h = plot_h / max(1, k)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((margin_left, 18), title, fill=(20, 20, 20), font=font)
    for frac in np.linspace(0.0, 1.0, 6):
        x = margin_left + int(frac * plot_w)
        draw.line([(x, margin_top), (x, margin_top + plot_h)], fill=(235, 235, 235), width=1)
        draw.text((x - 8, margin_top + plot_h + 8), f"{frac:.1f}", fill=(80, 80, 80), font=font)

    for row, (idx, value, name) in enumerate(zip(top_ids.tolist(), top_vals.tolist(), top_names, strict=False)):
        y = margin_top + int(row * row_h + row_h * 0.2)
        bar_h = max(16, int(row_h * 0.55))
        bar_w = int(max(0.0, min(1.0, value)) * plot_w)
        color = (214, 39, 40) if row == 0 else (31, 119, 180)
        draw.text((20, y), f"{idx:>3}  {name}", fill=(40, 40, 40), font=font)
        draw.rectangle((margin_left, y, margin_left + bar_w, y + bar_h), fill=color)
        draw.text((margin_left + bar_w + 8, y), f"{value:.4f}", fill=(60, 60, 60), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _save_action_chunk_plot(action_chunk: list[list[float]] | list[float], output_path: Path) -> None:
    arr = np.asarray(action_chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return

    mod = _lazy_imports()
    Image = mod["Image"]
    ImageDraw = mod["ImageDraw"]
    ImageFont = mod["ImageFont"]
    font = ImageFont.load_default()

    cols = 2
    rows = int(np.ceil(arr.shape[1] / cols))
    cell_w, cell_h = 620, 160
    width = cols * cell_w + 40
    height = rows * cell_h + 60
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), "Action Chunk by Dimension", fill=(20, 20, 20), font=font)

    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]
    steps = arr.shape[0]

    for dim in range(arr.shape[1]):
        row, col = divmod(dim, cols)
        x0 = 20 + col * cell_w
        y0 = 50 + row * cell_h
        margin_left, margin_right = 50, 20
        margin_top, margin_bottom = 22, 28
        plot_w = cell_w - margin_left - margin_right
        plot_h = cell_h - margin_top - margin_bottom
        cell = arr[:, dim]
        finite = cell[np.isfinite(cell)]
        if finite.size == 0:
            finite = np.array([0.0, 1.0], dtype=np.float32)
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if abs(y_max - y_min) < 1e-6:
            pad = 1.0 if abs(y_max) < 1e-6 else abs(y_max) * 0.1
            y_min -= pad
            y_max += pad

        draw.rectangle((x0, y0, x0 + cell_w - 10, y0 + cell_h - 10), outline=(225, 225, 225), width=1)
        draw.text((x0 + 6, y0 + 2), f"a[{dim}]  min={y_min:.3f}  max={y_max:.3f}", fill=(40, 40, 40), font=font)
        axis_x = x0 + margin_left
        axis_y = y0 + margin_top + plot_h
        draw.line([(axis_x, y0 + margin_top), (axis_x, axis_y), (axis_x + plot_w, axis_y)], fill=(60, 60, 60), width=1)
        xs = np.array([axis_x + plot_w / 2.0], dtype=np.float32) if steps == 1 else np.linspace(axis_x, axis_x + plot_w, steps, dtype=np.float32)

        def to_y(v: float) -> float:
            ratio = (float(v) - y_min) / (y_max - y_min)
            return y0 + margin_top + (1.0 - ratio) * plot_h

        pts = [(float(xs[i]), float(to_y(cell[i]))) for i in range(steps)]
        color = palette[dim % len(palette)]
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)
        else:
            draw.line(pts, fill=color, width=2)
        draw.text((axis_x - 40, y0 + margin_top - 4), f"{y_max:.2f}", fill=(90, 90, 90), font=font)
        draw.text((axis_x - 40, axis_y - 4), f"{y_min:.2f}", fill=(90, 90, 90), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _save_prediction_artifacts(request_dir: Path, output: dict[str, Any]) -> None:
    _write_json_atomic(request_dir / "output.json", output)
    viz_dir = request_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    clip_chunk = output.get("clip_running_status_chunk")
    if isinstance(clip_chunk, list) and clip_chunk:
        _save_line_plot(
            np.asarray(clip_chunk, dtype=np.float32),
            viz_dir / "progress.png",
            title="Clip Running Status",
            y_label="progress",
            colors=[(31, 119, 180)],
            series_labels=["progress"],
        )

    task_probs = output.get("task_probabilities")
    task_labels = output.get("task_probability_labels")
    if isinstance(task_probs, list) and task_probs and isinstance(task_labels, list):
        _save_task_probability_plot(task_probs, [str(v) for v in task_labels], viz_dir / "task_id.png")

    action_chunk = output.get("action_chunk")
    if isinstance(action_chunk, list) and action_chunk:
        _save_action_chunk_plot(action_chunk, viz_dir / "action.png")


def _save_raw_request_payload(
    args: "ServerArgs",
    *,
    task: str,
    state_list: list[float],
    clip_running_status: float | None,
    include_action_chunk: str | None,
    chunk_max_steps: int | None,
    flow_noise_seed: int | None,
    client_request_id: str | None,
    observation_stamp_ns: int | None,
    client_request_started_unix_ns: int | None,
    client_received_monotonic_ns: int | None,
    raw_uploads: dict[str, list[dict[str, Any]]],
) -> Path | None:
    if args.save_request_dir is None:
        return None

    root = Path(args.save_request_dir)
    root.mkdir(parents=True, exist_ok=True)

    request_id = f"request_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    request_dir = root / request_id
    request_dir.mkdir(parents=True, exist_ok=False)

    images_meta: dict[str, list[dict[str, Any]]] = {}
    images_root = request_dir / "images"
    for camera_key, items in raw_uploads.items():
        cam_dir = images_root / camera_key
        cam_dir.mkdir(parents=True, exist_ok=True)
        cam_meta: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            saved_name = f"{index:02d}_{_safe_upload_name(item.get('filename'), fallback=f'{camera_key}_{index:02d}.jpg')}"
            saved_path = cam_dir / saved_name
            saved_path.write_bytes(item["bytes"])
            cam_meta.append(
                {
                    "index": index,
                    "filename": str(item.get("filename") or ""),
                    "saved_path": str(saved_path),
                    "content_type": str(item.get("content_type") or ""),
                    "num_bytes": int(len(item["bytes"])),
                    "sha256": hashlib.sha256(item["bytes"]).hexdigest(),
                }
            )
        images_meta[camera_key] = cam_meta

    (request_dir / "prompt.txt").write_text(task, encoding="utf-8")
    _write_json_atomic(
        request_dir / "payload.json",
        {
            "schema": "pi05_remote_request_dump/v2",
            "request_id": request_id,
            "saved_at_unix_ns": time.time_ns(),
            "client_request_id": client_request_id,
            "observation_stamp_ns": observation_stamp_ns,
            "client_request_started_unix_ns": client_request_started_unix_ns,
            "client_received_monotonic_ns": client_received_monotonic_ns,
            "task": task,
            "state": [float(v) for v in state_list],
            "state_dim": int(len(state_list)),
            "clip_running_status": clip_running_status,
            "include_action_chunk": None if include_action_chunk is None else str(include_action_chunk),
            "chunk_max_steps": chunk_max_steps,
            "flow_noise_seed": flow_noise_seed,
            "image_counts": {k: len(v) for k, v in raw_uploads.items()},
            "images": images_meta,
        },
    )
    return request_dir


def _normalize_box_coords_for_response(boxes: np.ndarray | None) -> np.ndarray | None:
    if boxes is None:
        return None
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes = np.clip(boxes, 0.0, 1.0)
    if boxes.shape[-1] != 4:
        return boxes
    x_min = np.minimum(boxes[..., 0], boxes[..., 2])
    y_min = np.minimum(boxes[..., 1], boxes[..., 3])
    x_max = np.maximum(boxes[..., 0], boxes[..., 2])
    y_max = np.maximum(boxes[..., 1], boxes[..., 3])
    boxes[..., 0] = x_min
    boxes[..., 1] = y_min
    boxes[..., 2] = x_max
    boxes[..., 3] = y_max
    return boxes


def _normalize_point_coords_for_response(point: np.ndarray | None) -> np.ndarray | None:
    if point is None:
        return None
    return np.clip(np.asarray(point, dtype=np.float32), 0.0, 1.0)


# io is used in _load_image_tensor, keep import after module doc
import io  # noqa: E402


@dataclass
class ServerArgs:
    host: str = "0.0.0.0"
    port: int = 6007

    model_dir: Path = Path(
        "/mnt/dataset/chao-liang/outputs/20260110_134241_pi05_finetune/checkpoints/last/pretrained_model"
    )

    dataset_root: Path = Path("/mnt/dataset/wj-dataset/dataset_coffee/lerobot/coffee_megamix")
    dataset_repo_id: str = "coffee_megamix"

    device: str = "cuda"
    dtype: str = "bfloat16"  # float32 | float16 | bfloat16

    tokenizer_path: Path | None = None

    # If request doesn't include task, fall back to this.
    default_task: str = "pick things and place on the box"

    # Image resize size (must match training; dataset is 224)
    image_size: int = 224

    # If configured, raw head images are masked before resize/tokenization.
    dino_model: Path | None = None
    dino_prompt: str = "robot arm . robot gripper ."
    dino_threshold: float = 0.30
    dino_max_boxes: int = 4
    dino_device: str | None = None
    dino_short_edge: int | None = None

    # Whether to return full action chunk in JSON (can be large)
    return_action_chunk: bool = True
    num_inference_steps: int = 10

    # If > 0, only the first N steps of action_chunk / clip_running_status_chunk are returned (0 = full chunk).
    chunk_max_steps: int = 0

    # If set, save the raw multipart request (prompt/state/images) for debugging.
    save_request_dir: Path | None = None


class PI05RemoteService:
    def __init__(self, args: ServerArgs):
        mod = _lazy_imports()

        torch = mod["torch"]
        PreTrainedConfig = mod["PreTrainedConfig"]
        FeatureType = mod["FeatureType"]
        LeRobotDataset = mod["LeRobotDataset"]
        dataset_to_policy_features = mod["dataset_to_policy_features"]
        make_pre_post_processors = mod["make_pre_post_processors"]
        PI05Policy = mod["PI05Policy"]

        self.args = args
        self._lock = threading.Lock()

        if not args.model_dir.exists():
            raise FileNotFoundError(f"model-dir not found: {args.model_dir}")
        if not args.dataset_root.exists():
            raise FileNotFoundError(f"dataset-root not found: {args.dataset_root}")

        # Load dataset metadata/stats (for normalization and feature mapping)
        self.dataset = LeRobotDataset(args.dataset_repo_id, root=str(args.dataset_root), episodes=[0])

        # Load policy config from checkpoint config.json
        policy_cfg = PreTrainedConfig.from_pretrained(str(args.model_dir))
        policy_cfg.pretrained_path = args.model_dir
        policy_cfg.device = args.device
        policy_cfg.num_inference_steps = int(args.num_inference_steps)
        policy_cfg.push_to_hub = False
        if policy_cfg.repo_id is None:
            policy_cfg.repo_id = "local/remote"

        # Ensure features are populated (PI05Policy.__init__ expects them)
        features = dataset_to_policy_features(self.dataset.meta.features)
        policy_cfg.output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
        if not policy_cfg.input_features:
            policy_cfg.input_features = {key: ft for key, ft in features.items() if key not in policy_cfg.output_features}
        action_feature = policy_cfg.output_features.get("action")
        self.expected_action_dim = (
            int(action_feature.shape[0]) if action_feature is not None else None
        )
        self.action_target_mode = str(
            getattr(policy_cfg, "action_target_mode", "absolute")
        )

        # Tokenizer path needs to be available before policy/model creation in RynnBrain mode.
        tokenizer_path = args.tokenizer_path or _find_local_paligemma_tokenizer()
        if tokenizer_path is not None:
            policy_cfg.paligemma_tokenizer_path = str(tokenizer_path)

        # Load weights
        self.policy = PI05Policy.from_pretrained(str(args.model_dir), config=policy_cfg, strict=False)
        self.policy.config.num_inference_steps = int(args.num_inference_steps)
        self.policy.model.config.num_inference_steps = int(args.num_inference_steps)
        self.policy.eval()
        self.policy.to(args.device)
        state_feature = policy_cfg.input_features.get("observation.state")
        self.expected_state_dim = int(state_feature.shape[0]) if state_feature is not None else None
        self.head_image_key, self.left_image_key, self.right_image_key = _resolve_policy_image_keys(
            list(policy_cfg.input_features.keys())
        )
        print(
            "[pi05_remote_server] Resolved policy image keys:",
            f"head={self.head_image_key}",
            f"left={self.left_image_key}",
            f"right={self.right_image_key}",
        )
        self._warned_behavior_b1k_projection = False
        self.task_id_to_label = _build_task_id_to_label(
            getattr(policy_cfg, "task_id_map", None),
            int(getattr(policy_cfg, "task_aux_num_classes", 0) or 0),
        )

        # Build processors
        pre_overrides: dict[str, Any] = {}
        if tokenizer_path is not None:
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
        # Keep PI05 prompt construction aligned with training config even when
        # loading processors from checkpoint files.
        pre_overrides["pi05_prepare_state_tokenizer_processor_step"] = {
            "include_state_in_prompt": bool(getattr(policy_cfg, "include_state_in_language_prompt", True)),
            "task_id_key": str(getattr(policy_cfg, "task_id_key", "task_id")),
            "num_task_classes": int(getattr(policy_cfg, "task_aux_num_classes", 100)),
        }
        if args.device != "cuda":
            pre_overrides["device_processor"] = {"device": args.device}

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(args.model_dir),
            dataset_stats=self.dataset.meta.stats,
            preprocessor_overrides=pre_overrides,
        )

        self.pad_input_images = policy_cfg.action_space == "revo2_eef_pose"
        self.dino_masker = None
        if args.dino_model is not None:
            from lerobot.utils.dino_masking import DinoMasker

            self.dino_masker = DinoMasker(
                args.dino_model, args.dino_prompt, args.dino_threshold,
                args.dino_device or args.device, args.dino_max_boxes,
                short_edge=args.dino_short_edge,
            )

        # Autocast dtype
        if args.dtype == "float32":
            self.amp_dtype = torch.float32
        elif args.dtype == "float16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16

    def reset(self) -> None:
        with self._lock:
            self.policy.reset()

    def predict(
        self,
        sample: dict[str, Any],
        chunk_max_steps_override: int | None = None,
        flow_noise_seed: int | None = None,
    ) -> dict[str, Any]:
        mod = _lazy_imports()
        torch = mod["torch"]

        # One policy instance => serialize access
        with self._lock:
            t0 = time.perf_counter()
            revo2_relative = (
                self.policy.config.action_space == "revo2_eef_pose"
                and self.action_target_mode == "relative_pose"
            )
            original_state = sample["observation.state"].detach().clone() if revo2_relative else None
            model_in = self.preprocessor(sample)
            t1 = time.perf_counter()

            with _fixed_flow_noise_seed(
                torch, device=self.args.device, seed=flow_noise_seed
            ):
                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda" if "cuda" in self.args.device else "cpu",
                        dtype=self.amp_dtype,
                        enabled=True,
                    ):
                        pred_chunk, predicted_clip_running_status_chunk, task_id_logits, predicted_boxes, predicted_cross_center = (
                            self.policy.predict_action_chunk_with_status_task_and_aux(
                                model_in,
                                num_steps=int(self.args.num_inference_steps),
                            )
                        )
            model_timing_ms = self.policy.get_last_inference_timing_ms()

            t2 = time.perf_counter()

            # Postprocess (de-normalize) actions.
            # IMPORTANT: This must match eval_pi05_dataset.py, otherwise MSE vs GT will be meaningless.
            # pred_chunk: (B, T, A)
            post_first_obj = self.postprocessor(pred_chunk[:, 0])

            post_chunk_obj: Any
            try:
                post_chunk_obj = self.postprocessor(pred_chunk)
            except Exception:
                steps = []
                for t in range(pred_chunk.shape[1]):
                    steps.append(self.postprocessor(pred_chunk[:, t]))
                post_chunk_obj = steps

            t3 = time.perf_counter()

        def _as_action_array(x: object) -> np.ndarray:
            """Match eval_pi05_dataset.py: best-effort conversion to numpy action array."""
            if isinstance(x, dict):
                if "action" in x:
                    x = x["action"]
                elif len(x) == 1:
                    x = next(iter(x.values()))
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy()
            return np.asarray(x)

        def _to_chunk_t_a(x: object) -> np.ndarray:
            """Convert postprocessed output to a (T, A) numpy array."""
            arr = _as_action_array(x)
            if arr.ndim == 3 and arr.shape[0] == 1:
                return arr[0].astype(np.float32)
            if arr.ndim == 2:
                return arr.astype(np.float32)
            if arr.ndim == 1:
                return arr[None, :].astype(np.float32)
            raise ValueError(f"Unexpected action array shape: {arr.shape}")

        # Prefer chunk output (covers both action + action_chunk consistently)
        try:
            action_chunk_t_a = _to_chunk_t_a(post_chunk_obj)
        except Exception:
            # If post_chunk_obj is a list of per-step outputs, stack them
            if isinstance(post_chunk_obj, list) and post_chunk_obj:
                steps = [_as_action_array(s) for s in post_chunk_obj]
                steps = [s[0] if (isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] == 1) else s for s in steps]
                action_chunk_t_a = np.stack(steps, axis=0).astype(np.float32)
            else:
                # Fall back to first-step only
                first = _as_action_array(post_first_obj)
                if first.ndim == 2 and first.shape[0] == 1:
                    first = first[0]
                action_chunk_t_a = first[None, :].astype(np.float32)

        chunk_steps_total = int(action_chunk_t_a.shape[0])
        eff_max = int(self.args.chunk_max_steps)
        if chunk_max_steps_override is not None and chunk_max_steps_override > 0:
            eff_max = int(chunk_max_steps_override)
        if eff_max > 0 and action_chunk_t_a.shape[0] > eff_max:
            action_chunk_t_a = action_chunk_t_a[:eff_max].copy()
            if isinstance(post_chunk_obj, list) and len(post_chunk_obj) > eff_max:
                post_chunk_obj = post_chunk_obj[:eff_max]

        if original_state is not None:
            from lerobot.policies.pi05.revo2_relative import restore_revo2_absolute

            # HTTP samples contain unbatched state or observation history.
            action_chunk_t_a = restore_revo2_absolute(
                original_state.unsqueeze(0), torch.from_numpy(action_chunk_t_a).unsqueeze(0)
            )[0].numpy()

        first_action = action_chunk_t_a[0]
        clip_running_status = float(predicted_clip_running_status_chunk[0, 0].detach().float().cpu().item())
        task_logits_1d = task_id_logits[0].detach().float().cpu()
        task_probs_1d = torch.softmax(task_logits_1d, dim=0)
        task_probs_np = task_probs_1d.detach().float().cpu().numpy().astype(np.float32)
        task_labels = list(self.task_id_to_label)
        if len(task_labels) < int(task_probs_np.shape[0]):
            task_labels.extend(f"id_{i}" for i in range(len(task_labels), int(task_probs_np.shape[0])))
        pred_task_id = int(torch.argmax(task_probs_1d).item())
        pred_task_confidence = float(task_probs_1d[pred_task_id].item())
        pred_task_label = (
            task_labels[pred_task_id]
            if 0 <= pred_task_id < len(task_labels)
            else f"id_{pred_task_id}"
        )
        top_k = min(8, int(task_probs_np.shape[0]))
        top_ids = np.argsort(task_probs_np)[::-1][:top_k]
        task_probabilities_topk = [
            {
                "task_id": int(task_id),
                "label": task_labels[int(task_id)]
                if 0 <= int(task_id) < len(task_labels)
                else f"id_{int(task_id)}",
                "probability": float(task_probs_np[int(task_id)]),
            }
            for task_id in top_ids.tolist()
        ]
        pred_boxes_1d = None
        if predicted_boxes is not None:
            pred_boxes_np = predicted_boxes[0].detach().float().cpu().numpy()
            pred_boxes_1d = _normalize_box_coords_for_response(pred_boxes_np)
        pred_cross_center_1d = None
        if predicted_cross_center is not None:
            pred_cross_center_np = predicted_cross_center[0].detach().float().cpu().numpy()
            pred_cross_center_1d = _normalize_point_coords_for_response(pred_cross_center_np)

        def _jsonify(x: Any):
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy().tolist()
            if isinstance(x, np.ndarray):
                return x.astype(np.float32).tolist()
            if isinstance(x, dict):
                return {k: _jsonify(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_jsonify(v) for v in x]
            return x

        clip_chunk_list = predicted_clip_running_status_chunk[0].detach().float().cpu().numpy().tolist()
        if eff_max > 0 and len(clip_chunk_list) > eff_max:
            clip_chunk_list = clip_chunk_list[:eff_max]

        resp: dict[str, Any] = {
            "action_pose_frame": "absolute" if original_state is not None else self.action_target_mode,
            "flow_noise_seed": flow_noise_seed,
            "action": first_action.astype(np.float32).tolist(),
            "clip_running_status": clip_running_status,
            "clip_running_status_chunk": clip_chunk_list,
            "chunk_steps_total": chunk_steps_total,
            "chunk_steps_returned": int(action_chunk_t_a.shape[0]),
            "pred_task_id": pred_task_id,
            "pred_task_label": pred_task_label,
            "pred_task_confidence": pred_task_confidence,
            "task_probability_labels": task_labels,
            "task_probabilities": task_probs_np.tolist(),
            "task_probabilities_topk": task_probabilities_topk,
            "pred_boxes": None if pred_boxes_1d is None else pred_boxes_1d.tolist(),
            "pred_cross_center": None if pred_cross_center_1d is None else pred_cross_center_1d.tolist(),
            "timing_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "predict": (t2 - t1) * 1000.0,
                "postprocess": (t3 - t2) * 1000.0,
                "total": (t3 - t0) * 1000.0,
                **model_timing_ms,
            },
        }

        if self.args.return_action_chunk:
            resp["action_chunk"] = action_chunk_t_a.tolist()

        # Keep a debug field with postprocessor outputs (fully JSON-safe)
        resp["postprocessed"] = _jsonify({"first": post_first_obj, "chunk": post_chunk_obj})
        if original_state is not None:
            # Debug tensors remain relative; public action fields are absolute.
            resp["postprocessed"]["action_pose_frame"] = "relative_pose"

        return resp


def build_app(service: PI05RemoteService):
    mod = _lazy_imports()
    FastAPI = mod["FastAPI"]
    File = mod["File"]
    Form = mod["Form"]
    UploadFile = mod["UploadFile"]
    JSONResponse = mod["JSONResponse"]
    torch = mod["torch"]

    app = FastAPI(title="PI05 Remote Inference", version="0.1")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": service.args.device,
            "dtype": service.args.dtype,
            "model_dir": str(service.args.model_dir),
            "dataset_repo_id": service.args.dataset_repo_id,
            "expected_state_dim": service.expected_state_dim,
            "expected_action_dim": service.expected_action_dim,
            "action_target_mode": service.action_target_mode,
            "dino_mask": {
                "enabled": service.dino_masker is not None,
                "model": str(service.args.dino_model) if service.args.dino_model else None,
                "prompt": service.args.dino_prompt,
                "threshold": service.args.dino_threshold,
                "max_boxes": service.args.dino_max_boxes,
                "short_edge": service.args.dino_short_edge,
            },
            "head_image_key": service.head_image_key,
            "left_image_key": service.left_image_key,
            "right_image_key": service.right_image_key,
            "chunk_max_steps": int(service.args.chunk_max_steps),
            "request_dump_enabled": service.args.save_request_dir is not None,
            "request_dump_dir": (
                None
                if service.args.save_request_dir is None
                else str(service.args.save_request_dir)
            ),
            "request_dump_schema": "pi05_remote_request_dump/v2",
            "supports_flow_noise_seed": True,
        }

    @app.post("/v1/reset")
    def reset():
        service.reset()
        return {"status": "ok"}

    @app.post("/v1/predict")
    async def predict(
        cam_high: list[UploadFile] = File(...),
        wrist_left: list[UploadFile] | None = File(None),
        wrist_right: list[UploadFile] | None = File(None),
        state: str = Form(...),
        task: str | None = Form(None),
        clip_running_status: str | None = Form(None),
        include_action_chunk: str | None = Form(None),
        chunk_max_steps: str | None = Form(None),
        flow_noise_seed: str | None = Form(None),
        save_request: str | None = Form(None),
        client_request_id: str | None = Form(None),
        observation_stamp_ns: str | None = Form(None),
        client_request_started_unix_ns: str | None = Form(None),
        client_received_monotonic_ns: str | None = Form(None),
    ):
        try:
            state_list = json.loads(state)
            if not isinstance(state_list, list):
                raise TypeError("state must be a JSON list")
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": f"bad state: {exc}"})

        state_arr = np.asarray(state_list, dtype=np.float32).reshape(-1)
        expected_state_dim = service.expected_state_dim
        if expected_state_dim is not None and state_arr.shape[0] != expected_state_dim:
            if expected_state_dim == 23:
                projected_state = _project_behavior_b1k_state(state_arr)
                if projected_state is not None and projected_state.shape[0] == expected_state_dim:
                    if not service._warned_behavior_b1k_projection:
                        print(
                            "[pi05_remote_server] State dim mismatch: got raw BEHAVIOR proprio dim "
                            f"{state_arr.shape[0]}, expected {expected_state_dim}. "
                            "Applying built-in Behavior-B1K 256->23 projection."
                        )
                        service._warned_behavior_b1k_projection = True
                    state_arr = projected_state
            if state_arr.shape[0] != expected_state_dim:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"state length must be {expected_state_dim}, got {len(state_list)}"},
                )

        clip_running_status_value: float | None = None
        if clip_running_status is not None and str(clip_running_status).strip() != "":
            try:
                clip_running_status_value = float(json.loads(clip_running_status))
            except Exception:
                try:
                    clip_running_status_value = float(str(clip_running_status).strip())
                except Exception as exc:
                    return JSONResponse(status_code=400, content={"error": f"bad clip_running_status: {exc}"})

        async def _read_upload_group(uploads: list[UploadFile], *, image_key: str) -> tuple[list[Any], list[dict[str, Any]]]:
            tensors: list[Any] = []
            raw_items: list[dict[str, Any]] = []
            for index, upload in enumerate(uploads):
                payload = await upload.read()
                raw_items.append(
                    {
                        "filename": upload.filename or f"{image_key}_{index:02d}.jpg",
                        "content_type": upload.content_type or "",
                        "bytes": payload,
                    }
                )
                tensors.append(_load_image_tensor(
                    payload, image_key=image_key, image_size=service.args.image_size,
                    masker=service.dino_masker if image_key == "cam_high" else None,
                    pad=service.pad_input_images,
                ))
            return tensors, raw_items

        try:
            cam_high_tensors, cam_high_raw = await _read_upload_group(cam_high, image_key="cam_high")
            wrist_left_tensors, wrist_left_raw = await _read_upload_group(wrist_left or [], image_key="wrist_left")
            wrist_right_tensors, wrist_right_raw = await _read_upload_group(wrist_right or [], image_key="wrist_right")
            resolved_task = task or service.args.default_task
            sample = {
                service.head_image_key: _stack_temporal_images(cam_high_tensors, torch=torch),
                "observation.state": torch.tensor(state_arr, dtype=torch.float32),
                "task": resolved_task,
            }
            for image_key, tensors in (
                (service.left_image_key, wrist_left_tensors),
                (service.right_image_key, wrist_right_tensors),
            ):
                if image_key is not None:
                    sample[image_key] = _stack_temporal_images(tensors, torch=torch)
            if clip_running_status_value is not None:
                sample["clip_running_status"] = torch.tensor(clip_running_status_value, dtype=torch.float32)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": f"bad input: {exc}"})

        chunk_override: int | None = None
        if chunk_max_steps is not None and str(chunk_max_steps).strip() != "":
            try:
                v = int(str(chunk_max_steps).strip())
                chunk_override = v if v > 0 else None
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"chunk_max_steps must be a positive int, got {chunk_max_steps!r}"},
                )

        try:
            parsed_flow_noise_seed = _parse_optional_seed(flow_noise_seed)
            save_request_enabled = _parse_bool(save_request, default=True)
            parsed_observation_stamp_ns = _parse_optional_int(
                observation_stamp_ns, name="observation_stamp_ns"
            )
            parsed_client_request_started_unix_ns = _parse_optional_int(
                client_request_started_unix_ns,
                name="client_request_started_unix_ns",
            )
            parsed_client_received_monotonic_ns = _parse_optional_int(
                client_received_monotonic_ns,
                name="client_received_monotonic_ns",
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        saved_request_dir: Path | None = None
        try:
            if save_request_enabled:
                saved_request_dir = _save_raw_request_payload(
                    service.args,
                    task=resolved_task,
                    state_list=state_list,
                    clip_running_status=clip_running_status_value,
                    include_action_chunk=include_action_chunk,
                    chunk_max_steps=chunk_override,
                    flow_noise_seed=parsed_flow_noise_seed,
                    client_request_id=client_request_id,
                    observation_stamp_ns=parsed_observation_stamp_ns,
                    client_request_started_unix_ns=parsed_client_request_started_unix_ns,
                    client_received_monotonic_ns=parsed_client_received_monotonic_ns,
                    raw_uploads={
                        "cam_high": cam_high_raw,
                        "wrist_left": wrist_left_raw,
                        "wrist_right": wrist_right_raw,
                    },
                )
            if saved_request_dir is not None:
                print(f"[RequestDump] saved request to: {saved_request_dir}", flush=True)
        except Exception as exc:
            print(f"[RequestDump] failed to save request: {exc}", flush=True)

        try:
            out = service.predict(
                sample,
                chunk_max_steps_override=chunk_override,
                flow_noise_seed=parsed_flow_noise_seed,
            )
            if saved_request_dir is not None:
                try:
                    _save_prediction_artifacts(saved_request_dir, out)
                except Exception as exc:
                    print(f"[RequestDump] failed to save output artifacts: {exc}", flush=True)
            # Allow overriding whether to include action_chunk in the response.
            if include_action_chunk is not None:
                want = str(include_action_chunk).strip().lower() in {"1", "true", "yes", "y"}
                if not want:
                    out.pop("action_chunk", None)
            if saved_request_dir is not None:
                out["saved_request_dir"] = str(saved_request_dir)
            return out
        except Exception as exc:
            if saved_request_dir is not None:
                try:
                    _write_json_atomic(
                        saved_request_dir / "error.json",
                        {
                            "error": f"inference failed: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                except Exception as dump_exc:
                    print(f"[RequestDump] failed to save error info: {dump_exc}", flush=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"inference failed: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

    return app


def parse_args() -> ServerArgs:
    p = argparse.ArgumentParser(description="PI05 remote inference server (HTTP)")
    p.add_argument("--host", type=str, default=ServerArgs.host)
    p.add_argument("--port", type=int, default=ServerArgs.port)

    p.add_argument("--model-dir", type=Path, default=ServerArgs.model_dir)
    p.add_argument("--dataset-root", type=Path, default=ServerArgs.dataset_root)
    p.add_argument("--dataset-repo-id", type=str, default=ServerArgs.dataset_repo_id)

    p.add_argument("--device", type=str, default=ServerArgs.device)
    p.add_argument("--dtype", type=str, default=ServerArgs.dtype, choices=["float32", "float16", "bfloat16"])

    p.add_argument("--tokenizer-path", type=Path, default=None)
    p.add_argument("--default-task", type=str, default=ServerArgs.default_task)

    p.add_argument("--dino-model", type=Path, default=None,
                   help="Local Grounding DINO checkpoint; enables head-image rectangular blackout masking.")
    p.add_argument("--dino-prompt", default=ServerArgs.dino_prompt)
    p.add_argument("--dino-threshold", type=float, default=ServerArgs.dino_threshold)
    p.add_argument("--dino-max-boxes", type=int, default=ServerArgs.dino_max_boxes)
    p.add_argument("--dino-device", default=None)
    p.add_argument("--dino-short-edge", type=int, default=None,
                   help="Override detector resize shortest edge; must match dataset conversion.")
    p.add_argument("--image-size", type=int, default=ServerArgs.image_size)
    p.add_argument("--num-inference-steps", type=int, default=ServerArgs.num_inference_steps)
    p.add_argument("--return-action-chunk", action="store_true", default=ServerArgs.return_action_chunk)
    p.add_argument("--no-return-action-chunk", action="store_false", dest="return_action_chunk")
    p.add_argument(
        "--chunk-max-steps",
        type=int,
        default=0,
        help="If >0, return only the first N steps in action_chunk / clip_running_status_chunk (0 = full horizon).",
    )
    p.add_argument(
        "--save-request-dir",
        type=Path,
        default=ServerArgs.save_request_dir,
        help="If set, save each incoming multipart request under this directory.",
    )

    ns = p.parse_args()
    return ServerArgs(**vars(ns))


def main() -> None:
    args = parse_args()

    # Import uvicorn only when running main
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: uvicorn. Install with: pip install uvicorn") from exc

    service = PI05RemoteService(args)
    app = build_app(service)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
