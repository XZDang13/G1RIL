import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .motion_dataset import MotionLoader
from .amp_env_cfg import G1WalkEnvCfg
from .amp_env_cfg import G1DanceEnvCfg

class G1AMPEnv(DirectRLEnv):
    cfg:G1WalkEnvCfg | G1DanceEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, 23, device=self.device)

        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = 0.5 * (dof_upper_limits - dof_lower_limits)

        key_body_names = [
            "torso_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_hip_yaw_link",
            "right_hip_yaw_link",
            "left_rubber_hand",
            "right_rubber_hand",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

        self.motion_loader = MotionLoader(motion_file=self.cfg.expert_motion_file, device=self.device)

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self.motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self.motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self.motion_loader.get_body_index(key_body_names)

        self.motion_buffer = torch.zeros(
            (
                self.num_envs,
                self.cfg.motion_buffer_size,
                self.cfg.motion_space
            ),
            device=self.device
        )


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target_positions = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target_positions)

    def _get_observations(self):
        self.previous_actions = self.actions.clone()

        progress = (self.episode_length_buf.squeeze(-1).float() / (self.max_episode_length - 1)).unsqueeze(-1)

        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
            progress
        )

        for i in reversed(range(self.cfg.motion_buffer_size - 1)):
            self.motion_buffer[:, i + 1] = self.motion_buffer[:, i]
        # build AMP observation
        self.motion_buffer[:, 0] = obs.clone()
        motion_obs = self.motion_buffer.view(-1, self.cfg.motion_observation_space)

        return {"policy": obs, "motion": motion_obs}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)
     
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_state_w[:, 2]
        terminate = base_height < self.cfg.termination_height
        #terminate = base_height < 0
        return terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_samples = len(env_ids)
        if self.cfg.training:
            times = self.motion_loader.sample_times(num_samples)
        else:
            times = np.zeros(num_samples, dtype=np.float32)
        steps = torch.from_numpy(times * 30).to(self.device).long()
        self.episode_length_buf[env_ids] = steps
        
        self.actions[env_ids] = 0.0

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self.motion_loader.sample(num_samples=num_samples, times=times)

        motion_reference_body_index = self.motion_loader.get_body_index([self.cfg.reference_body])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        
        root_state[:, 0:3] = body_positions[:, motion_reference_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_reference_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_reference_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_reference_body_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)

        motion_observations = self.collect_expert_motion(num_samples, times)
        self.motion_buffer[env_ids] = motion_observations.view(num_samples, self.cfg.motion_buffer_size, -1)

        self.target_positions = self.robot.data.joint_pos.clone()
    
    def collect_expert_motion(self, num_samples: int, current_times: np.ndarray | None = None):
        if current_times is None:
            current_times = self.motion_loader.sample_times(num_samples)
            
        times = (
            np.expand_dims(current_times, axis=-1)
            - self.motion_loader.dt * np.arange(0, self.cfg.motion_buffer_size)
        ).flatten()

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self.motion_loader.sample(num_samples=num_samples, times=times)

        progress = (
            torch.as_tensor(times, device=dof_positions.device, dtype=dof_positions.dtype).unsqueeze(-1)
            / self.motion_loader.duration
        )

        motion_obs = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
            progress
        )

        return motion_obs.view(-1, self.cfg.motion_observation_space)

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    progress: torch.Tensor,
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
            progress
        ),
        dim=-1,
    )

    return obs