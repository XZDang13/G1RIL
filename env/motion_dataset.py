import torch
import numpy as np
from typing import Optional

def quat_xyzw_to_wxyz(q):
    # q shape: (..., 4) in xyzw order
    # return same shape in wxyz order
    return torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)

mujoco_to_isaac_idx = [
    0,   # left_hip_pitch_joint
    6,   # right_hip_pitch_joint
    12,  # waist_yaw_joint
    1,   # left_hip_roll_joint
    7,   # right_hip_roll_joint
    13,  # left_shoulder_pitch_joint
    18,  # right_shoulder_pitch_joint
    2,   # left_hip_yaw_joint
    8,   # right_hip_yaw_joint
    14,  # left_shoulder_roll_joint
    19,  # right_shoulder_roll_joint
    3,   # left_knee_joint
    9,   # right_knee_joint
    15,  # left_shoulder_yaw_joint
    20,  # right_shoulder_yaw_joint
    4,   # left_ankle_pitch_joint
    10,  # right_ankle_pitch_joint
    16,  # left_elbow_joint
    21,  # right_elbow_joint
    5,   # left_ankle_roll_joint
    11,  # right_ankle_roll_joint
    17,  # left_wrist_roll_joint
    22,  # right_wrist_roll_joint
]


class MotionDataset:
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.fps = data["fps"]
        self.root_pos = torch.from_numpy(data["root_pos"]).float()
        root_rot = torch.from_numpy(data["root_rot"]).float()
        joint_pos = torch.from_numpy(data["dof_pos"]).float()

        self.root_rot = quat_xyzw_to_wxyz(root_rot)
        self.joint_pos = joint_pos[:, mujoco_to_isaac_idx]

        self.num_frames = self.joint_pos.shape[0]

        self.dt = 1.0 / self.fps
        self.duration = self.dt * (self.num_frames - 1)
    
    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b
    
    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q
    
    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend
    
    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert (
            duration <= self.duration
        ), f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)
    
    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32)

        root_pos = self._interpolate(
            a=self.root_pos,
            start=index_0,
            end=index_1,
            blend=blend,
        )
        root_rot = self._slerp(
            q0=self.root_rot,
            start=index_0,
            end=index_1,
            blend=blend,
        )
        joint_pos = self._interpolate(
            a=self.joint_pos,
            start=index_0,
            end=index_1,   
            blend=blend,
        )

        sampled_states = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "joint_pos": joint_pos,
        }

        return sampled_states