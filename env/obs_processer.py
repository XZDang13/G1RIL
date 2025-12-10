import torch
from isaaclab.utils.math import quat_apply

def get_noise(values: torch.Tensor, noise_scale:float):
    noise = torch.rand_like(values) * noise_scale
    return noise

def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

class ObservationManager:
    @staticmethod
    def compute_privilege_obs(
        dof_positions: torch.Tensor,
        dof_velocities: torch.Tensor,
        root_positions: torch.Tensor,
        root_rotations: torch.Tensor,
        root_linear_velocities: torch.Tensor,
        root_angular_velocities: torch.Tensor,
        key_body_positions: torch.Tensor
    ) -> torch.Tensor:
        
        root_height = root_positions[:, 2:3]
        tangent_and_normal = quaternion_to_tangent_and_normal(root_rotations)
        relative_key_body_pos = (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0],
                                                                                         -1)

        obs = torch.cat(
            (
                dof_positions,
                dof_velocities,
                root_height,  # root body height
                tangent_and_normal,
                root_linear_velocities,
                root_angular_velocities,
                relative_key_body_pos,
                
            ),
            dim=-1,
        )

        
        dof_positions_noise = get_noise(dof_positions, 0.1)
        dof_velocities_noise = get_noise(dof_velocities, 0.15)
        root_height_noise = get_noise(root_height, 0.05)
        tangent_and_normal_noise = get_noise(tangent_and_normal, 0.1)
        root_linear_velocities_noise = get_noise(root_linear_velocities, 0.01)
        root_angular_velocities_noise = get_noise(root_angular_velocities, 0.01)
        relative_key_body_pos_noise = get_noise(relative_key_body_pos, 0.05)

        noise = torch.cat(
            (
                dof_positions_noise,
                dof_velocities_noise,
                root_height_noise,
                tangent_and_normal_noise,
                root_linear_velocities_noise,
                root_angular_velocities_noise,
                relative_key_body_pos_noise
            ),
            dim=-1,
        )
        

        return obs, noise

    @staticmethod
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

        angular_velocity_noise = get_noise(angular_velocity, 0.1)
        gravity_oritation_noise = get_noise(gravity_oritation, 0.05)
        dof_position_noise = get_noise(dof_position, 0.1)
        dof_velocity_noise = get_noise(dof_velocity, 0.15)
        previous_action_noise = get_noise(previous_action, 0.05)

        noise = torch.cat(
            (
                angular_velocity_noise,
                gravity_oritation_noise,
                dof_position_noise,
                dof_velocity_noise,
                previous_action_noise
            ),
            dim=-1,
        )


        return obs, noise
    

    