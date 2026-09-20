"""UMI-style relative pose conversion for Revo2 LeRobot samples."""

from __future__ import annotations

import torch
from torch import Tensor


def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    eps = torch.finfo(rotation_6d.dtype).eps * 16.0
    basis_x = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(eps)
    second_orthogonal = second - (basis_x * second).sum(dim=-1, keepdim=True) * basis_x
    basis_y = second_orthogonal / torch.linalg.vector_norm(second_orthogonal, dim=-1, keepdim=True).clamp_min(eps)
    basis_z = torch.linalg.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-1)


def _matrix_to_rotation_6d(rotation: Tensor) -> Tensor:
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def _pose_to_matrix(pose: Tensor) -> Tensor:
    if pose.shape[-1] != 9:
        raise ValueError(f"Expected [xyz, rotation_6d] pose with 9 dims, got {tuple(pose.shape)}")
    matrix = torch.eye(4, dtype=pose.dtype, device=pose.device).expand(*pose.shape[:-1], 4, 4).clone()
    matrix[..., :3, :3] = _rotation_6d_to_matrix(pose[..., 3:9])
    matrix[..., :3, 3] = pose[..., :3]
    return matrix


def _matrix_to_pose(matrix: Tensor) -> Tensor:
    return torch.cat((matrix[..., :3, 3], _matrix_to_rotation_6d(matrix[..., :3, :3])), dim=-1)


def _relative_pose(pose: Tensor, base: Tensor) -> Tensor:
    return _matrix_to_pose(torch.linalg.solve(base, _pose_to_matrix(pose)))


def convert_revo2_relative(state: Tensor, action: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    """Use the final observation frame as the common base for state and action."""
    if state.ndim not in (2, 3) or (action is not None and action.ndim not in (2, 3)):
        raise ValueError(f"Expected state/action rank 2 or 3, got {state.ndim}/{getattr(action, 'ndim', None)}")
    if state.shape[-1] != 18:
        raise ValueError(f"Revo2 state must be 18D, got {state.shape[-1]}")
    if action is not None and action.shape[-1] != 30:
        raise ValueError(f"Revo2 action must be 30D, got {action.shape[-1]}")

    if action is not None and state.shape[0] != action.shape[0]:
        raise ValueError("state and action batch sizes must match")

    state_was_2d = state.ndim == 2
    action_was_2d = action is not None and action.ndim == 2
    state_seq = state[:, None, :] if state_was_2d else state
    action_seq = action[:, None, :] if action_was_2d else action
    state_out = state_seq.clone()
    action_out = action_seq.clone() if action_seq is not None else None

    for side in range(2):
        pose_slice = slice(side * 9, (side + 1) * 9)
        base = _pose_to_matrix(state_seq[:, -1, pose_slice])
        state_out[..., pose_slice] = _relative_pose(state_seq[..., pose_slice], base[:, None])
        if action_out is not None:
            action_out[..., pose_slice] = _relative_pose(action_seq[..., pose_slice], base[:, None])

    if state_was_2d:
        state_out = state_out[:, 0]
    if action_was_2d:
        action_out = action_out[:, 0]
    return state_out, action_out



def restore_revo2_absolute(state: Tensor, action: Tensor) -> Tensor:
    """Restore physical actions from ORIGINAL observations using a common base.

    State: [B,18] or [B,observations,18]. Action: [B,30] or [B,horizon,30].
    Hand targets pass through; inputs are not modified. Call after unnormalization.
    """
    if state.ndim not in (2, 3) or state.shape[-1] != 18:
        raise ValueError("state must have shape [B,18] or [B,observations,18]")
    if action.ndim not in (2, 3) or action.shape[-1] != 30:
        raise ValueError("action must have shape [B,30] or [B,horizon,30]")
    if state.shape[0] != action.shape[0]:
        raise ValueError("state and action batch sizes must match")
    current = state[:, -1] if state.ndim == 3 else state
    current = current.to(device=action.device, dtype=action.dtype)
    result = action.clone()
    for side in range(2):
        pose_slice = slice(side * 9, (side + 1) * 9)
        base = _pose_to_matrix(current[..., pose_slice])
        if action.ndim == 3:
            base = base[:, None]
        result[..., pose_slice] = _matrix_to_pose(base @ _pose_to_matrix(action[..., pose_slice]))
    return result
