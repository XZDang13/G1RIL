import numpy as np
import time
import torch
import struct

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from config import G1Config
from model import Actor
from RLAlg.normalizer import Normalizer
from RLAlg.nn.steps import StochasticContinuousPolicyStep

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

class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def create_damping_cmd(cmd: LowCmdHG):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0


def create_zero_cmd(cmd: LowCmdHG):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


def init_cmd_hg(cmd: LowCmdHG, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

class Controller:
    def __init__(self):
        self.config = G1Config()
        self.remote_controller = RemoteController()

        self.device = torch.device("cuda:0")

        self.obs_normalizer = Normalizer((75,)).to(self.device)
        self.actor = Actor(75, 23).to(self.device)

        normalizer_weights, actor_weights, _ = torch.load("student_weight.pth")
        self.obs_normalizer.load_state_dict(normalizer_weights)
        self.actor.load_state_dict(actor_weights)
        self.obs_normalizer.eval()
        self.actor.eval()

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

        self.replays = torch.load("actions_replay.pt")

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        self.counter = 0
        self.last_actions = np.zeros(len(self.config.policy_joints_order), dtype=np.float32)

        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        obs_batch = self.obs_normalizer(obs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action.cpu()

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        init_pos = {

        }

        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            init_pos[joint_name] = self.low_state.motor_state[idx].q
        

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for idx, joint_name in enumerate(self.config.joints_settings.keys()):
                target_pos = self.config.init_state[joint_name]
                self.low_cmd.motor_cmd[idx].q = init_pos[joint_name] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = self.config.pd_params[joint_name][0]
                self.low_cmd.motor_cmd[idx].kd = self.config.pd_params[joint_name][1]
                self.low_cmd.motor_cmd[idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for idx, joint_name in enumerate(self.config.joints_settings.keys()):
                self.low_cmd.motor_cmd[idx].q = self.config.init_state[joint_name]
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = self.config.pd_params[joint_name][0]
                self.low_cmd.motor_cmd[idx].kd = self.config.pd_params[joint_name][1]
                self.low_cmd.motor_cmd[idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def get_target_pos(self):
        joint_states = {}

        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            joint_states[joint_name] = (self.low_state.motor_state[idx].q, self.low_state.motor_state[idx].dq)

        dof_pos = np.zeros(len(self.config.policy_joints_order), dtype=np.float32)
        dof_vel = np.zeros(len(self.config.policy_joints_order), dtype=np.float32)
        for idx, joint_name in enumerate(self.config.policy_joints_order):
            dof_pos[idx] = joint_states[joint_name][0]
            dof_vel[idx] = joint_states[joint_name][1]

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        gravity_orientation = get_gravity_orientation(quat)

        gravity_orientation = torch.from_numpy(gravity_orientation).float()
        ang_vel = torch.from_numpy(ang_vel).float()
        dof_pos = torch.from_numpy(dof_pos).float()
        dof_vel = torch.from_numpy(dof_vel).float()
        last_action = torch.from_numpy(self.last_actions).float()

        obs = torch.cat([
            ang_vel,
            gravity_orientation,
            dof_pos,
            dof_vel,
            last_action,
        ]).unsqueeze(0).to(self.device)

        actions = self.get_action(obs, True).squeeze(0)
        
        self.last_actions = actions.numpy().copy()

        target_pos = (self.action_offset + self.action_scale * actions).numpy()

        cmd = {}

        for idx, joint_name in enumerate(self.config.policy_joints_order):
            cmd[joint_name] = target_pos[idx]

        return cmd

    def run(self):
        cmd = self.get_target_pos()
    
        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            if joint_name in cmd:
                self.low_cmd.motor_cmd[idx].q = cmd[joint_name]
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = self.config.pd_params[joint_name][0]
                self.low_cmd.motor_cmd[idx].kd = self.config.pd_params[joint_name][1]
                self.low_cmd.motor_cmd[idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

        self.counter += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()


    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller()

    controller.wait_for_low_state()
    controller.zero_torque_state()

    controller.move_to_default_pos()

    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    
    print("Exit")
