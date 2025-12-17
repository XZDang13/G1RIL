import time
import numpy as np
import torch

import mujoco
import mujoco.viewer

from .motion_dataset import MotionLoader

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q (torch.Tensor): Quaternion [w, x, y, z]
        v (torch.Tensor): Vector to rotate

    Returns:
        torch.Tensor: Rotated vector
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (torch.dot(q_vec, v)) * 2.0
    return a - b + c

class MujocoEnv:
    def __init__(self, simulation_dt, decimation, render=False):
        self.mj_model = mujoco.MjModel.from_xml_path("env/assets/scene.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = simulation_dt
        self.mj_viewer = None
        self.render = render
        if self.render:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.motion_loader = MotionLoader("env/motion_data/walk.npz", device="cpu")

        self.gravity_vector = torch.tensor([0.0, 0.0, -1.0]).float()
        self.previous_action = torch.zeros(23).float()

        self.mujoco2isaac = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
        self.isaac2mujoco = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

        self.action_offset = torch.as_tensor(
            [
                0.1746,  0.1746,  0.0000,  1.2217, -1.2217, -0.2094, -0.2094,  0.0000,
                0.0000,  0.3316, -0.3316,  1.3963,  1.3963,  0.0000,  0.0000, -0.1745,
                -0.1745,  0.5236,  0.5236,  0.0000,  0.0000,  0.0000,  0.0000
            ]
        ).float()

        self.action_scale = torch.as_tensor(
            [
                2.4347, 2.4347, 2.3562, 1.5708, 1.5708, 2.5918, 2.5918, 2.4818, 2.4818,
                1.7279, 1.7279, 1.3352, 1.3352, 2.3562, 2.3562, 0.6283, 0.6283, 1.4137,
                1.4137, 0.2356, 0.2356, 1.7750, 1.7750
            ]
        ).float()

        self.effort_limit = torch.as_tensor(
            [ 88.,  88.,  88., 139., 139.,  25.,  25.,  88.,  88.,  25.,  25., 139.,
             139.,  25.,  25.,  50.,  50.,  25.,  25.,  50.,  50.,  25.,  25.]
        )[self.isaac2mujoco]

        self.kp = torch.as_tensor(
            [40.1792, 40.1792, 40.1792, 99.0984, 99.0984, 14.2506, 14.2506, 40.1792,
             40.1792, 14.2506, 14.2506, 99.0984, 99.0984, 14.2506, 14.2506, 28.5012,
             28.5012, 14.2506, 14.2506, 28.5012, 28.5012, 14.2506, 14.2506]
        )[self.isaac2mujoco]

        self.kd = torch.as_tensor(
            [2.5579, 2.5579, 2.5579, 6.3088, 6.3088, 0.9072, 0.9072, 2.5579, 2.5579,
             0.9072, 0.9072, 6.3088, 6.3088, 0.9072, 0.9072, 1.8144, 1.8144, 0.9072,
             0.9072, 1.8144, 1.8144, 0.9072, 0.9072]
        )[self.isaac2mujoco]

        self.simulation_dt = simulation_dt
        self.decimation = decimation
        self.policy_dt = simulation_dt * decimation

        self.n_steps = 0

    def get_projected_gravity(self):
        base_quat = torch.from_numpy(self.mj_data.qpos[3:7]).float()
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vector).float()

        return projected_gravity
    
    def get_base_ang_vel(self):
        base_ang_vel = torch.from_numpy(self.mj_data.qvel[3:6]).float()
        return base_ang_vel
    
    def get_joint_pos(self):
        joint_pos = torch.from_numpy(self.mj_data.qpos[7:]).float()[self.mujoco2isaac]
        return joint_pos
    
    def get_joint_vel(self):
        joint_vel = torch.from_numpy(self.mj_data.qvel[6:]).float()[self.mujoco2isaac]

        return joint_vel

    def get_obs(self):
        
        projected_gravity = self.get_projected_gravity()
        base_ang_vel = self.get_base_ang_vel()
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()

        return torch.cat([
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            self.previous_action,
        ])

    def reset(self):
        self.previous_action[:] = 0.0
        times = np.zeros(1)

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self.motion_loader.sample(num_samples=1, times=times)

        dof_positions = dof_positions.squeeze(0).numpy()
        dof_velocities = dof_velocities.squeeze(0).numpy()
        body_positions = body_positions.squeeze(0).numpy()
        body_rotations = body_rotations.squeeze(0).numpy()
        body_linear_velocities = body_linear_velocities.squeeze(0).numpy()
        body_angular_velocities = body_angular_velocities.squeeze(0).numpy()

        root_pos = body_positions[0]
        root_pos[2] += 0.05
        root_quat = body_rotations[0]
        root_linear_vel = body_linear_velocities[0]
        root_ang_vel = body_angular_velocities[0]

        self.mj_data.qpos[0] = 0.0
        self.mj_data.qpos[1] = 0.0
        self.mj_data.qpos[2] = root_pos[2]
        self.mj_data.qpos[3:7] = root_quat
        self.mj_data.qpos[7:] = dof_positions

        self.mj_data.qvel[:3] = root_linear_vel
        self.mj_data.qvel[3:6] = root_ang_vel
        self.mj_data.qvel[6:] = dof_velocities

        obs = self.get_obs()

        return obs
    
    def _apply_actions(self, actions: torch.Tensor):
        target_pos = self.action_offset + self.action_scale * actions

        target_pos = target_pos[self.isaac2mujoco]
        joint_pos = torch.from_numpy(self.mj_data.qpos[7:]).float()
        joint_vel = torch.from_numpy(self.mj_data.qvel[6:]).float()

        # PD control
        tau = self.kp * (target_pos - joint_pos) - self.kd * joint_vel

        # Apply EMA filtering and torque limits
        tau_clipped = torch.clip(tau, -self.effort_limit, self.effort_limit)

        self.mj_data.ctrl[:] = tau_clipped.numpy()

    def step(self, actions):
        step_start_time = time.perf_counter()
        self.previous_action = actions.clone()
        for _ in range(self.decimation):
            self._apply_actions(actions)
            mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None and self.mj_viewer.is_running():
            self.mj_viewer.sync()
        else:
            # viewer was closed manually -> stop touching it
            self.mj_viewer = None

        obs = self.get_obs()

        time_until_next_step = self.policy_dt - (time.perf_counter() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.n_steps += 1

        return obs
    
    def close(self):
        if self.mj_viewer is not None:
            try:
                if self.mj_viewer.is_running():
                    self.mj_viewer.close()
            finally:
                self.mj_viewer = None

