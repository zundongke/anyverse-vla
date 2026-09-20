#!/usr/bin/env python3
"""Convert Revo2 RRD/CSV episodes to LeRobot with top_head and DINO masks.

The action/state convention intentionally remains absolute here.  The UMI-style
relative conversion is applied by the PI05 data processor at load time.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from rerun.dataframe import load_recording
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.dino_masking import DinoMasker


POSE_ENTITY = {
    "left": "/ego/synced/left_wrist/pose",
    "right": "/ego/synced/right_wrist/pose",
}
POSE_COMPONENTS = (
    "position/x", "position/y", "position/z",
    "orientation/w", "orientation/x", "orientation/y", "orientation/z",
)
HAND_FIELDS = (
    "thumb_proximal_joint", "thumb_metacarpal_joint", "index_proximal_joint",
    "middle_proximal_joint", "ring_proximal_joint", "pinky_proximal_joint",
)
RGB_ENTITY = "/ego/synced/camera/head/rgb"


def _ns(value) -> int:
    return int(value.value) if hasattr(value, "value") else int(value)


def read_scalar_series(recording, entity: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for batch in recording.view(index="log_time", contents=entity).select():
        data = batch.to_pydict()
        key = next(k for k in data if k.endswith(":Scalars:scalars"))
        for timestamp, scalar in zip(data["timestamp"], data[key]):
            if scalar:
                values[_ns(timestamp)] = float(scalar[0])
    return values


def read_rgb(recording) -> dict[int, bytes]:
    frames: dict[int, bytes] = {}
    for batch in recording.view(index="log_time", contents=RGB_ENTITY).select():
        data = batch.to_pydict()
        key = next(k for k in data if k.endswith(":EncodedImage:blob"))
        for timestamp, payload in zip(data["timestamp"], data[key]):
            if payload and payload[0]:
                frames[_ns(timestamp)] = bytes(payload[0])
    return frames


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.isfinite(quat).all() or (norm < 1e-8).any():
        raise ValueError("Invalid wrist quaternion")
    quat = quat / norm
    w, x, y, z = quat.T
    rot = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)
    return np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=-1).astype(np.float32)


def pose_vector(rows: np.ndarray) -> np.ndarray:
    return np.concatenate([rows[:, :3].astype(np.float32), quat_to_rot6d(rows[:, 3:7])], axis=-1)


def resolve_rrd(csv_path: Path, rrd_source: Path | None = None) -> Path:
    directory = rrd_source or csv_path.parent
    exact = directory / f"{csv_path.stem}.rrd"
    if exact.is_file():
        return exact
    original = directory / f"{csv_path.stem.split('_revo2_pose_targets')[0]}.rrd"
    if original.is_file():
        return original
    raise FileNotFoundError(f"No original RRD for {csv_path}: checked {exact} and {original}")


def load_episode(csv_path: Path, rrd_source: Path | None = None):
    recording = load_recording(resolve_rrd(csv_path, rrd_source))
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_side = {"left": {}, "right": {}}
    for row in rows:
        side, timestamp = row["side"], int(row["timestamp_ns"])
        if timestamp in by_side[side]:
            raise ValueError(f"Duplicate {side} timestamp in {csv_path}: {timestamp}")
        by_side[side][timestamp] = row
    timestamps = sorted(by_side["left"])
    if len(timestamps) < 2:
        raise ValueError(f"Episode needs at least two paired observations: {csv_path}")
    if timestamps != sorted(by_side["right"]):
        raise ValueError(f"left/right timestamps differ: {csv_path}")

    pose_by_side = {}
    for side, prefix in POSE_ENTITY.items():
        components = [read_scalar_series(recording, f"{prefix}/{component}") for component in POSE_COMPONENTS]
        missing = [t for t in timestamps if any(t not in c for c in components)]
        if missing:
            raise ValueError(f"{csv_path.name}: missing {len(missing)} pose timestamps")
        pose_by_side[side] = np.stack([
            [components[j][t] for j in range(len(components))] for t in timestamps
        ]).astype(np.float64)

    pose = np.concatenate([pose_vector(pose_by_side["left"]), pose_vector(pose_by_side["right"])], axis=-1)
    hand = np.zeros((len(timestamps), 12), dtype=np.float32)
    for i, t in enumerate(timestamps):
        for side_i, side in enumerate(("left", "right")):
            hand[i, side_i * 6:(side_i + 1) * 6] = [float(by_side[side][t][f]) for f in HAND_FIELDS]
    rgb = read_rgb(recording)
    if not rgb:
        raise ValueError(f"{csv_path.name}: no {RGB_ENTITY} stream")
    rgb_times = np.asarray(sorted(rgb), dtype=np.int64)
    rgb_for_frame = []
    max_error = 0
    for t in timestamps[:-1]:
        pos = int(np.searchsorted(rgb_times, t))
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(rgb_times)]
        idx = min(candidates, key=lambda i: abs(int(rgb_times[i]) - t))
        max_error = max(max_error, abs(int(rgb_times[idx]) - t))
        rgb_for_frame.append(rgb[int(rgb_times[idx])])
    if max_error > 50_000_000:
        raise ValueError(f"{csv_path.name}: RGB/action alignment error {max_error / 1e6:.1f}ms")
    state = pose[:-1].astype(np.float32)
    action = np.concatenate([pose[1:], hand[1:]], axis=-1).astype(np.float32)
    time_s = (np.asarray(timestamps[:-1], dtype=np.float64) - timestamps[0]) / 1e9
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"Non-finite state/action in {csv_path}")
    return state, action, time_s.astype(np.float32), rgb_for_frame



def pad_image(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), (size, size), color=(0, 0, 0))


_MASKER = None
_WORKER_ARGS = None


def initialize_mask_worker(options: dict) -> None:
    global _MASKER, _WORKER_ARGS
    _WORKER_ARGS = argparse.Namespace(**options)
    torch.set_num_threads(_WORKER_ARGS.cpu_threads)
    if _WORKER_ARGS.mask_backend == 'sam2':
        from lerobot.utils.sam_box_masking import SamBoxMasker
        _MASKER = SamBoxMasker(_WORKER_ARGS.sam2_config, _WORKER_ARGS.sam2_checkpoint,
                              _WORKER_ARGS.device)
        return
    _MASKER = DinoMasker(
        _WORKER_ARGS.dino_model, _WORKER_ARGS.prompt, _WORKER_ARGS.threshold,
        _WORKER_ARGS.device, _WORKER_ARGS.max_boxes, short_edge=_WORKER_ARGS.dino_short_edge,
    )


def cache_signature(csv_path: Path, args, *, detection_only=False) -> str:
    rrd = resolve_rrd(csv_path, args.rrd_source)
    model_file = args.dino_model / 'model.safetensors'
    data = {
        'version': 1, 'csv': str(csv_path), 'rrd': str(rrd),
        'csv_size': csv_path.stat().st_size, 'csv_mtime': csv_path.stat().st_mtime_ns,
        'rrd_size': rrd.stat().st_size, 'rrd_mtime': rrd.stat().st_mtime_ns,
        'model': str(args.dino_model), 'model_size': model_file.stat().st_size,
        'model_mtime': model_file.stat().st_mtime_ns,
        'prompt': args.prompt, 'threshold': args.threshold, 'max_boxes': args.max_boxes,
        'image_size': args.image_size, 'dino_short_edge': args.dino_short_edge,
    }
    if getattr(args, 'mask_backend', 'box') == 'sam2' and not detection_only:
        checkpoint = args.sam2_checkpoint
        data.update(mask_backend='sam2_rgb_v1', sam2_config=args.sam2_config,
                    sam2_checkpoint=str(checkpoint), sam2_size=checkpoint.stat().st_size,
                    sam2_mtime=checkpoint.stat().st_mtime_ns)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def mask_episode(csv_path: Path) -> dict:
    args = _WORKER_ARGS
    cache = args.cache_dir / f'{csv_path.stem}.npz'
    signature = cache_signature(csv_path, args)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['signature']) == signature:
                return {'episode': csv_path.stem, 'frames': len(saved['state']), 'cached': True}
    state, action, times, payloads = load_episode(csv_path, args.rrd_source)
    detection_boxes = None
    if args.mask_backend == 'sam2':
        with np.load(args.detection_cache_dir / f'{csv_path.stem}.npz', allow_pickle=False) as saved:
            if str(saved['signature']) != cache_signature(csv_path, args, detection_only=True):
                raise ValueError(f'DINO detection cache signature mismatch: {csv_path}')
            detection_boxes = saved['boxes'].copy()
        if len(detection_boxes) != len(payloads):
            raise ValueError(f'DINO detection frame count mismatch: {csv_path}')
    images, boxes = [], []
    for start in range(0, len(payloads), args.batch_size):
        batch = [Image.open(BytesIO(p)).convert('RGB') for p in payloads[start:start + args.batch_size]]
        if detection_boxes is None:
            masked, detected = _MASKER.process(batch)
        else:
            masked, detected = _MASKER.process(batch, detection_boxes[start:start + len(batch)])
        images.extend(np.asarray(pad_image(im, args.image_size)) for im in masked)
        boxes.append(detected)
        completed = min(start + args.batch_size, len(payloads))
        if completed % 16 == 0 or completed == len(payloads):
            print(json.dumps({"episode": csv_path.stem, "masked_frames": completed, "total_frames": len(payloads)}), flush=True)
    tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(
        tmp, signature=signature, state=state, action=action, times=times,
        images=np.stack(images), boxes=np.concatenate(boxes),
    )
    tmp.replace(cache)
    return {'episode': csv_path.stem, 'frames': len(state), 'cached': False}


class RunningStats:
    def __init__(self, dims):
        self.count = 0
        self.minimum = np.full(dims, np.inf)
        self.maximum = np.full(dims, -np.inf)
        self.total = np.zeros(dims)
        self.squares = np.zeros(dims)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, len(self.total))
        self.count += len(values)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.total += values.sum(axis=0)
        self.squares += (values * values).sum(axis=0)

    def result(self):
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.squares / self.count - mean * mean, 0))
        return {'min': self.minimum.tolist(), 'max': self.maximum.tolist(),
                'mean': mean.tolist(), 'std': std.tolist(), 'count': [self.count]}


def add_relative_stats(state, action, state_stats, action_stats, args):
    from lerobot.policies.pi05.revo2_relative import convert_revo2_relative
    current = np.arange(len(state))[:, None]
    history = np.arange(-(args.history_frames - 1) * args.history_stride, 1, args.history_stride)
    observations = state[np.clip(current + history, 0, len(state) - 1)]
    targets = action[np.clip(current + np.arange(args.action_horizon), 0, len(action) - 1)]
    relative_state, relative_action = convert_revo2_relative(
        torch.from_numpy(observations), torch.from_numpy(targets),
    )
    state_stats.add(relative_state.numpy())
    action_stats.add(relative_action.numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='CSV directory, or root with original_rrd/retargeted_csv')
    parser.add_argument('--rrd-source', type=Path, default=None, help='Original RRD directory when separate from CSV')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dino-model', type=Path, required=True)
    parser.add_argument('--task', default='pico_demo')
    parser.add_argument('--prompt', default='robot arm . robot gripper .')
    parser.add_argument('--threshold', type=float, default=.30)
    parser.add_argument('--max-boxes', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--dino-short-edge', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--history-frames', type=int, default=6)
    parser.add_argument('--history-stride', type=int, default=5)
    parser.add_argument('--action-horizon', type=int, default=50)
    parser.add_argument('--limit-episodes', type=int, default=None)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--mask-backend', choices=['box', 'sam2'], default='box')
    parser.add_argument('--sam2-config', default='configs/sam2/sam2_hiera_t.yaml')
    parser.add_argument('--sam2-checkpoint', type=Path)
    parser.add_argument('--detection-cache-dir', type=Path)
    args = parser.parse_args()
    if args.mask_backend == 'sam2' and (args.sam2_checkpoint is None or args.detection_cache_dir is None):
        parser.error('sam2 requires --sam2-checkpoint and --detection-cache-dir')
    for name in ['batch_size', 'workers', 'cpu_threads', 'history_frames', 'history_stride', 'action_horizon']:
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if (args.source / 'retargeted_csv').is_dir():
        args.rrd_source = args.rrd_source or args.source / 'original_rrd'
        args.source = args.source / 'retargeted_csv'
    csv_paths = sorted(args.source.glob('*.csv'))
    if args.limit_episodes is not None:
        csv_paths = csv_paths[:args.limit_episodes]
    if not csv_paths:
        raise ValueError(f'No episode CSV files in {args.source}')
    args.cache_dir = args.cache_dir or args.output.parent / f'{args.output.name}.cache'
    torch.set_num_threads(args.cpu_threads)
    if args.validate_only:
        episodes = []
        for csv_path in tqdm(csv_paths, desc='Validating source episodes'):
            state, action, times, payloads = load_episode(csv_path, args.rrd_source)
            episodes.append({'episode': csv_path.stem, 'frames': len(state),
                             'state_dim': state.shape[-1], 'action_dim': action.shape[-1]})
        print(json.dumps({'episodes': len(episodes), 'frames': sum(e['frames'] for e in episodes),
                          'task': args.task, 'details': episodes}))
        return
    if args.output.exists():
        raise FileExistsError(f'{args.output}: choose a new output directory; cached DINO results can be reused')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        initialize_mask_worker(vars(args))
        results = [mask_episode(p) for p in tqdm(csv_paths, desc='DINO mask episodes')]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_mask_worker, initargs=(vars(args),)) as pool:
            results = []
            futures = {pool.submit(mask_episode, path): path for path in csv_paths}
            from concurrent.futures import as_completed
            for future in tqdm(as_completed(futures), total=len(futures), desc='DINO mask episodes'):
                result = future.result()
                results.append(result)
                print(json.dumps({'completed': len(results), 'total': len(csv_paths), **result}), flush=True)
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (18,), 'names': [f'state_{i}' for i in range(18)]},
        'action': {'dtype': 'float32', 'shape': (30,), 'names': [f'action_{i}' for i in range(30)]},
        'observation.images.top_head': {'dtype': 'image', 'shape': (3, args.image_size, args.image_size),
                                        'names': ['channels', 'height', 'width']},
        'observation.boxes': {'dtype': 'float32', 'shape': (args.max_boxes, 4),
                              'names': ['box', ['x1_norm', 'y1_norm', 'x2_norm', 'y2_norm']]},
    }
    dataset = LeRobotDataset.create(repo_id=args.output.name, root=args.output, fps=15,
                                    robot_type='pico_revo2', features=features, use_videos=False,
                                    image_writer_processes=0, image_writer_threads=4)
    state_stats, action_stats = RunningStats(18), RunningStats(30)
    for csv_path in tqdm(csv_paths, desc='Writing LeRobot episodes'):
        with np.load(args.cache_dir / f'{csv_path.stem}.npz', allow_pickle=False) as saved:
            state, action = saved['state'], saved['action']
            images, boxes = saved['images'], saved['boxes']
            add_relative_stats(state, action, state_stats, action_stats, args)
            for i in range(len(state)):
                dataset.add_frame({'observation.state': state[i], 'action': action[i],
                                   'observation.images.top_head': Image.fromarray(images[i]),
                                   'observation.boxes': boxes[i], 'task': args.task})
            dataset.save_episode()
    dataset.finalize()
    dataset.stop_image_writer()
    stats_path = args.output / 'meta/stats.json'
    original_stats = stats_path.read_text()
    (args.output / 'meta/absolute_stats.json').write_text(original_stats)
    stats = json.loads(original_stats)
    stats['observation.state'], stats['action'] = state_stats.result(), action_stats.result()
    stats_path.write_text(json.dumps(stats, indent=2))
    manifest = {
        'episodes': len(csv_paths), 'frames': sum(r['frames'] for r in results), 'task': args.task,
        'source_csv': str(args.source), 'source_rrd': str(args.rrd_source), 'output': str(args.output),
        'stored_pose_frame': 'absolute', 'normalization_pose_frame': 'relative_pose',
        'history_frames': args.history_frames, 'history_stride': args.history_stride,
        'action_horizon': args.action_horizon,
        'dino_model': str(args.dino_model), 'dino_prompt': args.prompt, 'dino_threshold': args.threshold,
        'dino_max_boxes': args.max_boxes, 'dino_short_edge': args.dino_short_edge,
        'image_size': args.image_size, 'source_episodes': [r['episode'] for r in sorted(results, key=lambda r: r['episode'])],
        'mask_backend': args.mask_backend,
        'sam2_config': args.sam2_config if args.mask_backend == 'sam2' else None,
        'sam2_checkpoint': str(args.sam2_checkpoint) if args.mask_backend == 'sam2' else None,
        'detection_cache_dir': str(args.detection_cache_dir) if args.mask_backend == 'sam2' else None,
    }
    (args.output / 'meta/conversion_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
