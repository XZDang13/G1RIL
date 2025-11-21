import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .motion_dataset import MotionDataset
from .amp_env_cfg import G1WalkEnvCfg

class G1WalkEnv(DirectRLEnv):
    cfg:G1WalkEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, 23, device=self.device)
        self.previous_motion_obs = torch.zeros(self.num_envs, 3+4+23, device=self.device)

        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

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

        self.expert_motion_data = MotionDataset(self.cfg.expert_motion_file)


    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self):
        self.previous_actions = self.actions.clone()

        root_pos = self.robot.data.root_state_w[:, :3] - self.scene.env_origins
        root_quat = self.robot.data.root_state_w[:, 3:7]
        joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos # (num_envs, 23)
        joint_vel = self.robot.data.joint_vel               # (num_envs, 23)


        policy_obs = torch.cat([
            root_pos,
            root_quat,
            joint_pos,
            joint_vel
        ], dim=-1)

        current_motion_obs = torch.cat([
            root_pos,
            root_quat,
            joint_pos
        ], dim=-1)

        motion_obs = torch.cat(
            [current_motion_obs, self.previous_motion_obs], dim=-1
        )

        self.previous_motion_obs = current_motion_obs.clone()

        return {"policy": policy_obs, "motion": motion_obs}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)
     
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_state_w[:, 2]
        terminate = base_height < 0.5
        #terminate = base_height < 0
        return terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self.actions[env_ids] = 0.0

        samples = self.expert_motion_data.sample(num_samples=len(env_ids))

        root_state = self.robot.data.root_state_w[env_ids].clone()
        
        root_pos = samples["root_pos"].to(self.device) + self.scene.env_origins[env_ids]
        root_rot = samples["root_rot"].to(self.device)
        root_state[:, :3] = root_pos
        root_state[:, 3:7] = root_rot

        joint_pos = samples["joint_pos"].to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.previous_motion_obs[env_ids] = torch.cat([
            root_pos,
            root_rot,
            joint_pos
        ], dim=-1)

        self.target_joint_pos = joint_pos

    def collect_expert_motion(self, num_samples: int):
        current_times = self.expert_motion_data.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self.expert_motion_data.dt * np.arange(0, 2)
        ).flatten()

        samples = self.expert_motion_data.sample(num_samples, times=times)

        root_pos = samples["root_pos"]
        root_rot = samples["root_rot"]
        joint_pos = samples["joint_pos"]

        motion_obs = torch.cat([
            root_pos,
            root_rot,
            joint_pos
        ], dim=-1)

        return motion_obs.view(-1, 2*(3+4+23))

