import time

import mujoco
import mujoco.viewer

import torch
import numpy as np

from RLAlg.normalizer import Normalizer
from RLAlg.nn.steps import StochasticContinuousPolicyStep

from model import Actor
from env.motion_dataset import MotionLoader

simulation_dt = 1/600
simulation_duration = 60

isaac_joints_order = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']
mujoco_joints_order = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']

action_offset = torch.as_tensor(
    [
        0.1746,  0.1746,  0.0000,  1.2217, -1.2217, -0.2094, -0.2094,  0.0000,
        0.0000,  0.3316, -0.3316,  1.3963,  1.3963,  0.0000,  0.0000, -0.1745,
        -0.1745,  0.5236,  0.5236,  0.0000,  0.0000,  0.0000,  0.0000
    ]
)
action_scale = torch.as_tensor(
    [
        2.4347, 2.4347, 2.3562, 1.5708, 1.5708, 2.5918, 2.5918, 2.4818, 2.4818,
        1.7279, 1.7279, 1.3352, 1.3352, 2.3562, 2.3562, 0.6283, 0.6283, 1.4137,
        1.4137, 0.2356, 0.2356, 1.7750, 1.7750
    ]
)

kps = np.array(
    [
        200., 150., 150., 200.,  20.,  20., 200., 150., 150., 200.,  20.,
        20., 200.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,
        40.
    ]
)

kds = np.array(
    [
        5.,  5.,  5.,  5.,  2.,  2.,  5.,  5.,  5.,  5.,  2.,  2.,  5.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10.
    ]
)

motion_loader = MotionLoader("env/motion_data/walk.npz", device="cpu")
times = np.zeros(1)
(
    dof_positions,
    dof_velocities,
    body_positions,
    body_rotations,
    body_linear_velocities,
    body_angular_velocities,
) = motion_loader.sample(num_samples=1, times=times)

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

mujoco2isaac = []
for joint in isaac_joints_order:
    idx = mujoco_joints_order.index(joint)
    mujoco2isaac.append(idx)

isaac2mujoco = []
for joint in mujoco_joints_order:
    idx = isaac_joints_order.index(joint)
    isaac2mujoco.append(idx)

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

@torch.no_grad()
def get_action(obs_batch:torch.Tensor, determine:bool=False):
    obs_batch = obs_normalizer(obs_batch)
    actor_step:StochasticContinuousPolicyStep = actor(obs_batch)
    action = actor_step.action
    if determine:
        action = actor_step.mean
    
    return action

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

device = torch.device("cuda:0")

obs_normalizer = Normalizer((75,)).to(device)
actor = Actor(75, 23).to(device)

normalizer_weights, actor_weights, _ = torch.load("student_weight.pth")
obs_normalizer.load_state_dict(normalizer_weights)
actor.load_state_dict(actor_weights)
obs_normalizer.eval()
actor.eval()

mj_model = mujoco.MjModel.from_xml_path("env/assets/scene.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = simulation_dt

previous_action = np.zeros(23)

counter = 0

mujoco.mj_resetData(mj_model, mj_data)

mj_data.qpos[2] = root_pos[2]
mj_data.qpos[3:7] = root_quat
mj_data.qpos[7:] = dof_positions

mj_data.qvel[:3] = root_linear_vel
mj_data.qvel[3:6] = root_ang_vel
mj_data.qvel[6:] = dof_velocities

mujoco.mj_step(mj_model, mj_data)

target_joints = dof_positions

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    # Close the viewer automatically after simulation_duration wall-seconds.

    start = time.time()
    

    while viewer.is_running() and time.time() - start < simulation_duration:
        step_start = time.time()
        
        if counter % 10 == 0:
            qj = mj_data.qpos[7:]
            dqj = mj_data.qvel[6:]
            quat = mj_data.qpos[3:7]
            ang_vel = mj_data.qvel[3:6]

            qj = qj[mujoco2isaac]
            dqj = dqj[mujoco2isaac]
            gravity_orientation = get_gravity_orientation(quat)

            obs = np.concatenate([ang_vel, gravity_orientation, qj, dqj, previous_action])
            
            obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            action = get_action(obs, True).cpu()

            target_joints = action_scale * action + action_offset

            target_joints = target_joints[:, isaac2mujoco].squeeze(0).numpy()
            #target_joints = dof_positions

            previous_action = action.squeeze(0).numpy()

        tau = pd_control(target_joints, mj_data.qpos[7:], kps, np.zeros_like(kds), mj_data.qvel[6:], kds)
        #print(tau)
        mj_data.ctrl[:] = tau
        #mj_data.ctrl[:] = np.ones(23) * 10
        #mj_data.qpos[-1] = 1
        mujoco.mj_step(mj_model, mj_data)

        counter += 1
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)