"""Shared action post-processing utilities for remote inference clients/servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChunkPostprocessStats:
    raw_steps_before: int
    raw_steps_after: int
    truncated_to: int | None
    bridge_steps_inserted: int


def parse_skip_dims(value: str | None, *, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Parse comma-separated action indices to skip during smoothing."""
    if value is None or not str(value).strip():
        return default
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(out)


def gaussian_filter_action_chunk(
    actions: list[np.ndarray],
    *,
    window: int = 9,
    sigma: float = 2.0,
    initial: np.ndarray | None = None,
    skip_dims: tuple[int, ...] = (),
) -> list[np.ndarray]:
    """Apply 1D Gaussian smoothing along the chunk time axis for each action dim.

    Gripper / discrete dims listed in ``skip_dims`` are left unchanged.
    """
    if len(actions) < 2:
        return actions

    w = int(window)
    if w <= 1:
        return actions
    if w % 2 == 0:
        w += 1

    s = float(sigma)
    if not np.isfinite(s) or s <= 0.0:
        return actions

    arr = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in actions], axis=0)
    t, a_dim = arr.shape
    skip = {int(d) for d in skip_dims if 0 <= int(d) < a_dim}

    m = w // 2
    k = np.arange(-m, m + 1, dtype=np.float32)
    kernel = np.exp(-(k * k) / (2.0 * s * s)).astype(np.float32)
    kernel /= float(kernel.sum())

    out = arr.copy()
    init_vec = None
    if initial is not None:
        init_vec = np.asarray(initial, dtype=np.float32).reshape(-1)

    for d in range(a_dim):
        if d in skip:
            continue

        left = arr[0, d]
        if init_vec is not None and init_vec.shape[0] > d:
            left = float(init_vec[d])
        right = arr[-1, d]

        seq = np.concatenate(
            [
                np.full((m,), left, dtype=np.float32),
                arr[:, d],
                np.full((m,), right, dtype=np.float32),
            ],
            axis=0,
        )
        out[:, d] = np.convolve(seq, kernel, mode="valid").astype(np.float32)

    for d in skip:
        out[:, d] = arr[:, d]

    return [out[i].astype(np.float32, copy=True) for i in range(t)]


def _action_chunk_from_result(result: dict[str, Any]) -> np.ndarray:
    action_chunk = result.get("action_chunk")
    if action_chunk is not None:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            return chunk.reshape(1, -1)
        return chunk
    action = np.asarray(result["action"], dtype=np.float32).reshape(-1)
    return action.reshape(1, -1)


def apply_gaussian_filter_to_predict_result(
    result: dict[str, Any],
    *,
    window: int,
    sigma: float,
    skip_dims: tuple[int, ...],
    initial: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Smooth ``action`` / ``action_chunk`` inside a PI05 predict result dict."""
    chunk = _action_chunk_from_result(result)
    if chunk.shape[0] < 2:
        last_action = chunk[0].astype(np.float32, copy=True) if chunk.shape[0] == 1 else None
        return result, last_action

    filtered = gaussian_filter_action_chunk(
        [chunk[i] for i in range(chunk.shape[0])],
        window=window,
        sigma=sigma,
        initial=initial,
        skip_dims=skip_dims,
    )
    filtered_arr = np.stack(filtered, axis=0).astype(np.float32)

    updated = dict(result)
    updated["action"] = filtered_arr[0].tolist()
    if "action_chunk" in result:
        updated["action_chunk"] = filtered_arr.tolist()
    return updated, filtered_arr[-1]


def gaussian_filter_enabled(window: int, sigma: float) -> bool:
    return int(window) > 1 and float(sigma) > 0.0 and np.isfinite(float(sigma))


def linear_bridge_between_vectors(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Insert ``n`` strictly intermediate linear steps between vectors ``a`` and ``b``."""
    n = int(n)
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"bridge vector shape mismatch: {a.shape} vs {b.shape}")
    if n <= 0:
        return np.zeros((0, a.shape[0]), dtype=np.float32)
    out = np.empty((n, a.shape[0]), dtype=np.float32)
    for k in range(1, n + 1):
        alpha = k / float(n + 1)
        out[k - 1] = a + alpha * (b - a)
    return out


def linear_bridge_between_scalars(ca: float, cb: float, n: int) -> np.ndarray:
    """Insert ``n`` strictly intermediate linear steps between scalars ``ca`` and ``cb``."""
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    out = np.empty((n,), dtype=np.float64)
    for k in range(1, n + 1):
        alpha = k / float(n + 1)
        out[k - 1] = float(ca) + alpha * (float(cb) - float(ca))
    return out


def _clip_chunk_from_result(result: dict[str, Any], length: int) -> list[float]:
    clip_chunk = result.get("clip_running_status_chunk") or []
    if clip_chunk:
        return [float(v) for v in clip_chunk[:length]]
    clip_scalar = result.get("clip_running_status")
    if clip_scalar is not None:
        return [float(clip_scalar)] * length
    return []


def _write_chunk_to_result(
    result: dict[str, Any],
    chunk: np.ndarray,
    clip_list: list[float],
) -> dict[str, Any]:
    updated = dict(result)
    updated["action"] = chunk[0].astype(np.float32).tolist()
    if "action_chunk" in result:
        updated["action_chunk"] = chunk.astype(np.float32).tolist()
    if clip_list:
        updated["clip_running_status"] = float(clip_list[0])
        updated["clip_running_status_chunk"] = clip_list
    updated["chunk_steps_returned"] = int(chunk.shape[0])
    return updated


def apply_chunk_postprocess_to_predict_result(
    result: dict[str, Any],
    *,
    filter_window: int,
    filter_sigma: float,
    filter_skip_dims: tuple[int, ...],
    chunk_execute_max_raw_steps: int,
    cross_chunk_bridge_steps: int,
    last_executed_action: np.ndarray | None = None,
    last_executed_clip: float | None = None,
) -> tuple[dict[str, Any], ChunkPostprocessStats]:
    """Apply piper-aligned chunk postprocess: gaussian smooth -> truncate -> cross-chunk bridge."""
    raw_steps_before = int(_action_chunk_from_result(result).shape[0])
    working = dict(result)

    if gaussian_filter_enabled(filter_window, filter_sigma):
        working, _ = apply_gaussian_filter_to_predict_result(
            working,
            window=int(filter_window),
            sigma=float(filter_sigma),
            skip_dims=filter_skip_dims,
            initial=last_executed_action,
        )

    chunk = _action_chunk_from_result(working)
    clip_list = _clip_chunk_from_result(working, int(chunk.shape[0]))

    truncated_to: int | None = None
    max_raw = max(0, int(chunk_execute_max_raw_steps))
    if max_raw > 0 and chunk.shape[0] > max_raw:
        chunk = chunk[:max_raw].copy()
        clip_list = clip_list[:max_raw]
        truncated_to = max_raw

    bridge_steps_inserted = 0
    n_bridge = max(0, int(cross_chunk_bridge_steps))
    if n_bridge > 0 and last_executed_action is not None and chunk.shape[0] > 0:
        bridge_from = np.asarray(last_executed_action, dtype=np.float32).reshape(-1)
        bridge_mat = linear_bridge_between_vectors(bridge_from, chunk[0], n_bridge)
        chunk = np.concatenate([bridge_mat, chunk], axis=0)
        if clip_list:
            clip_a = float(last_executed_clip) if last_executed_clip is not None else float(clip_list[0])
            clip_b = float(clip_list[0])
            bridge_clip = linear_bridge_between_scalars(clip_a, clip_b, n_bridge).astype(np.float64)
            clip_list = bridge_clip.tolist() + clip_list
        bridge_steps_inserted = n_bridge

    working = _write_chunk_to_result(working, chunk, clip_list)
    stats = ChunkPostprocessStats(
        raw_steps_before=raw_steps_before,
        raw_steps_after=int(chunk.shape[0]),
        truncated_to=truncated_to,
        bridge_steps_inserted=bridge_steps_inserted,
    )
    return working, stats
