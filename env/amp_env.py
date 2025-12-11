import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .motion_dataset import MotionLoader
from .amp_env_cfg import G1WalkEnvCfg
from .amp_env_cfg import G1DanceEnvCfg
from .obs_processer import ObservationManager

class G1AMPEnv(DirectRLEnv):
    cfg:G1WalkEnvCfg | G1DanceEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, 23, device=self.device)

        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = 0.5 * (dof_upper_limits - dof_lower_limits)

        #print(self.action_offset)
        #print(self.action_scale)

        #self.action_scale = .5

        self.action_queue = torch.zeros(
            (self.cfg.scene.num_envs, self.cfg.ctrl_delay_step_range[1] + 1, self.cfg.action_space),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._action_delay = torch.randint(
            self.cfg.ctrl_delay_step_range[0],
            self.cfg.ctrl_delay_step_range[1] + 1,
            (self.cfg.scene.num_envs,),
            device=self.device,
            requires_grad=False,
        )

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
        if self.cfg.ctrl_delay_step_range[1] > 0:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            self.actions = self.action_queue[torch.arange(self.num_envs), self._action_delay].clone()
        else:
            self.actions = actions.clone()

        self.target_pos = self.action_scale * self.actions + self.action_offset
        self.tau = self.pd_control()

    def _apply_action(self):
        self.robot.set_joint_effort_target(self.tau)

    def pd_control(self):
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        tau = (
            self.robot.data.joint_stiffness * (self.target_pos - joint_pos) - self.robot.data.joint_damping * joint_vel
        )

        return tau

    def _get_observations(self):
        self.previous_actions = self.actions.clone()

        #progress = (self.episode_length_buf.squeeze(-1).float() / (self.max_episode_length - 1)).unsqueeze(-1)

        default_obs, default_obs_noise = ObservationManager.compute_default_obs(
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.previous_actions
        )

        privilege_obs, privilege_obs_noise = ObservationManager.compute_privilege_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        for i in reversed(range(self.cfg.motion_buffer_size - 1)):
            self.motion_buffer[:, i + 1] = self.motion_buffer[:, i]
        # build AMP observation
        self.motion_buffer[:, 0] = privilege_obs.clone()
        motion_obs = self.motion_buffer.view(-1, self.cfg.motion_observation_space)

        if self.cfg.add_default_obs_noise:
            default_obs += default_obs_noise

        if self.cfg.add_privilege_obs_noise:
            privilege_obs += privilege_obs_noise
            
        return {"default": default_obs, "privilege": privilege_obs, "motion": motion_obs}
    
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
        
        steps = torch.from_numpy(times * 60).to(self.device).long()
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

        #progress = (
        #    torch.as_tensor(times, device=dof_positions.device, dtype=dof_positions.dtype).unsqueeze(-1)
        #    / self.motion_loader.duration
        #)

        motion_obs, _ = ObservationManager.compute_privilege_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        return motion_obs.view(-1, self.cfg.motion_observation_space)
