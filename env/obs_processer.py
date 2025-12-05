import torch
from isaaclab.utils.math import quat_apply

def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

def compute_privilege_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    #progress: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
            #progress
        ),
        dim=-1,
    )

    return obs

def compute_default_obs(
    angular_velocity: torch.Tensor,
    gravity_oritation: torch.Tensor,
    dof_position: torch.Tensor,
    dof_velocity: torch.Tensor,
    previous_action: torch.Tensor
):
    obs = torch.cat(
        (
            angular_velocity,
            gravity_oritation,
            dof_position,
            dof_velocity,
            previous_action
        ),
        dim=-1
    )

    return obs