from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / 'src/core/wj_lerobot/dataset_tools/convert_revo2_rrd_to_lerobot_dino.py'
spec = importlib.util.spec_from_file_location('revo2_conversion', MODULE_PATH)
conversion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion)


def test_separate_original_rrd_and_retargeted_csv(tmp_path):
    original = tmp_path / 'original_rrd'
    original.mkdir()
    expected = original / 'episode_000246.rrd'
    expected.touch()
    csv = tmp_path / 'episode_000246_revo2_pose_targets.csv'
    assert conversion.resolve_rrd(csv, original) == expected


def test_relative_statistics_cover_history_and_future_targets():
    identity = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    state = np.tile(np.concatenate((identity, identity)), (4, 1))
    state[:, 0] = np.arange(4) * .1
    state[:, 9] = np.arange(4) * .1 + 10
    targets = state.copy()
    targets[:, [0, 9]] += .1
    hands = np.tile(np.arange(12, dtype=np.float32), (4, 1))
    action = np.concatenate((targets, hands), axis=-1)
    state_copy, action_copy = state.copy(), action.copy()
    state_stats, action_stats = conversion.RunningStats(18), conversion.RunningStats(30)
    args = SimpleNamespace(history_frames=2, history_stride=1, action_horizon=3)
    conversion.add_relative_stats(state, action, state_stats, action_stats, args)
    historical, future = state_stats.result(), action_stats.result()
    np.testing.assert_allclose(np.array(historical['min'])[[0, 9]], [-.1, -.1], atol=2e-6)
    np.testing.assert_allclose(np.array(historical['max'])[[0, 9]], [0, 0], atol=2e-6)
    np.testing.assert_allclose(np.array(future['min'])[[0, 9]], [.1, .1], atol=2e-6)
    np.testing.assert_allclose(np.array(future['max'])[[0, 9]], [.3, .3], atol=2e-6)
    np.testing.assert_array_equal(future['min'][18:], np.arange(12))
    np.testing.assert_array_equal(future['max'][18:], np.arange(12))
    assert historical['count'] == [8]
    assert future['count'] == [12]
    np.testing.assert_array_equal(state, state_copy)
    np.testing.assert_array_equal(action, action_copy)
