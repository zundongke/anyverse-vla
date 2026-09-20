from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[2] / "src/open_source/lerobot/src/lerobot/policies/pi05/revo2_relative.py"
spec = importlib.util.spec_from_file_location("revo2_relative", MODULE_PATH)
relative = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relative)


def poses(batch, time):
    # A 90-degree rotation about Z ensures translations are tested in local axes.
    pose = torch.tensor([2., 3., 4., 0., 1., 0., -1., 0., 0.], dtype=torch.float64)
    result = torch.cat((pose, pose)).repeat(batch, time, 1)
    result[..., 9:12] += 10
    result[:, :, 0] += torch.arange(time)
    return result


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("chunk", [False, True])
def test_relative_absolute_roundtrip(history, chunk):
    state = poses(2, 3)
    if not history:
        state = state[:, -1]
    action = torch.cat((poses(2, 4), torch.arange(12, dtype=torch.float64).repeat(2, 4, 1)), dim=-1)
    if not chunk:
        action = action[:, 0]
    state_copy, action_copy = state.clone(), action.clone()
    state_rel, action_rel = relative.convert_revo2_relative(state, action)
    restored = relative.restore_revo2_absolute(state, action_rel)
    torch.testing.assert_close(restored, action)
    torch.testing.assert_close(action_rel[..., 18:], action[..., 18:])
    torch.testing.assert_close(state, state_copy)
    torch.testing.assert_close(action, action_copy)
    last = state_rel[:, -1] if history else state_rel
    identity = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.], dtype=state.dtype).repeat(2)
    torch.testing.assert_close(last, identity.repeat(2, 1))


def test_observation_only_and_common_base_translation():
    state = poses(1, 3)
    state_rel, missing_action = relative.convert_revo2_relative(state)
    assert missing_action is None
    torch.testing.assert_close(state_rel[0, 0, :3], torch.tensor([0., 2., 0.], dtype=state.dtype))
    # Repeated future targets must restore identically, rather than accumulate.
    action = torch.cat((poses(1, 1).repeat(1, 3, 1), torch.zeros(1, 3, 12, dtype=state.dtype)), dim=-1)
    _, action_rel = relative.convert_revo2_relative(state, action)
    torch.testing.assert_close(relative.restore_revo2_absolute(state, action_rel), action)


def test_reject_mismatched_batches_and_dimensions():
    state = poses(2, 3)
    with pytest.raises(ValueError, match="batch"):
        relative.convert_revo2_relative(state, torch.zeros(1, 4, 30))
    with pytest.raises(ValueError, match="batch"):
        relative.restore_revo2_absolute(state, torch.zeros(1, 4, 30))
    with pytest.raises(ValueError, match="30"):
        relative.restore_revo2_absolute(state, torch.zeros(2, 4, 18))
