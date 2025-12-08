import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tqdm import trange
import gymnasium
import torch
import numpy as np

from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.normalizer import Normalizer

from env.amp_env_cfg import G1WalkEnvCfg, G1DanceEnvCfg
from model import Actor

class Evaluator:
    def __init__(self):
        self.cfg = G1WalkEnvCfg()

        self.env_name = "G1AMP-v0"
        
        self.cfg.scene.num_envs = 4
        self.cfg.training = False
        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        obs_dim = self.cfg.observation_space
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.obs_normalizer = Normalizer((obs_dim,)).to(self.device)
        self.actor = Actor(obs_dim, action_dim).to(self.device)

        normalizer_weights, actor_weights, _ = torch.load("student_weight.pth")
        self.obs_normalizer.load_state_dict(normalizer_weights)
        self.actor.load_state_dict(actor_weights)
        self.obs_normalizer.eval()
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        obs_batch = self.obs_normalizer(obs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action

    
    def rollout(self, obs, info):
        length = 0

        for i in range(5000):
            privilege_obs = obs["default"]
            action = self.get_action(privilege_obs, True)
            #print(action)
            #action = torch.zeros_like(action)  # for testing, use zero action
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)

            length += 1

            obs = next_obs

        return obs, info

    def eval(self):
        obs, info = self.env.reset()
        print(obs)
        obs, info = self.rollout(obs, info)

        self.env.close()

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.eval()
    simulation_app.close()