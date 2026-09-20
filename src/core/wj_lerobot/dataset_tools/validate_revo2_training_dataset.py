#!/usr/bin/env python3
"""Validate converted Revo2 data through the real training preprocessor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--expected-episodes', type=int, default=100)
    parser.add_argument('--expected-frames', type=int, default=3968)
    parser.add_argument('--tokenizer', type=Path, default=Path('/wj-dataset/lerobot/vla_pretrain_model/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.dataset / 'meta/conversion_manifest.json').read_text())
    assert manifest['episodes'] == args.expected_episodes
    assert manifest['frames'] == args.expected_frames
    assert manifest['normalization_pose_frame'] == 'relative_pose'
    history = list(range(-(manifest['history_frames'] - 1) * manifest['history_stride'], 1, manifest['history_stride']))
    dataset = LeRobotDataset(args.dataset.name, root=args.dataset, delta_timestamps={
        'observation.state': [t / 15 for t in history],
        'observation.images.top_head': [t / 15 for t in history],
        'action': [t / 15 for t in range(manifest['action_horizon'])],
    })
    assert dataset.meta.total_episodes == args.expected_episodes
    assert len(dataset) == args.expected_frames
    config = PI05Config(
        device='cpu', action_space='revo2_eef_pose', action_target_mode='relative_pose',
        enable_task_aux_loss=False, enable_split_action_heads=True, gripper_action_indices=list(range(18, 30)),
        input_features={
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(18,)),
            'observation.images.top_head': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(30,))},
        normalization_mapping={FeatureType.STATE: NormalizationMode.MIN_MAX,
                               FeatureType.ACTION: NormalizationMode.MIN_MAX,
                               FeatureType.VISUAL: NormalizationMode.IDENTITY},
        paligemma_tokenizer_path=str(args.tokenizer),
    )
    pre, _ = make_pi05_pre_post_processors(config, dataset_stats=dataset.meta.stats, task_pool=[manifest['task']])
    indices = sorted(set([0, min(10, len(dataset) - 1), len(dataset) - 1, *np.linspace(0, len(dataset) - 1, 12).astype(int)]))
    for index in indices:
        sample = dataset[int(index)]
        assert sample['task'] == manifest['task']
        batch = pre(default_collate([sample]))
        for key in ['observation.state', 'action']:
            value = batch[key]
            assert torch.isfinite(value).all(), (index, key)
            assert float(value.abs().max()) <= 1.0002, (index, key, value.min(), value.max())
        assert batch['observation.state'].shape == (1, manifest['history_frames'], 18)
        assert batch['action'].shape == (1, manifest['action_horizon'], 30)
        assert batch['observation.images.top_head'].shape == (1, manifest['history_frames'], 3, 224, 224)
        assert torch.isfinite(batch['observation.language.tokens']).all()
    print(json.dumps({'episodes': dataset.meta.total_episodes, 'frames': len(dataset),
                      'validated_samples': len(indices), 'task': manifest['task'],
                      'relative_normalization_and_tokenizer': 'passed'}))


if __name__ == '__main__':
    main()
