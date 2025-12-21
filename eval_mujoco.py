import torch
from env.mujoco_env import MujocoEnv
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.normalizer import Normalizer

from model import Actor

device = torch.device("cuda:0")

obs_normalizer = Normalizer((75,)).to(device)
actor = Actor(75, 23).to(device)

normalizer_weights, actor_weights, _ = torch.load("student_weight.pth")
obs_normalizer.load_state_dict(normalizer_weights)
actor.load_state_dict(actor_weights)
obs_normalizer.eval()
actor.eval()

@torch.no_grad()
def get_action(obs_batch:torch.Tensor, determine:bool=False):
    obs_batch = obs_normalizer(obs_batch)
    actor_step:StochasticContinuousPolicyStep = actor(obs_batch)
    action = actor_step.action
    if determine:
        action = actor_step.mean
    
    return action.cpu()

env = MujocoEnv(1/3000, 50, True)

#'''
obs = env.reset()
for _ in range(1000):
    action = get_action(obs.to(device), True)
    #action = torch.zeros_like(action)
    obs = env.step(action)
'''

obs = env.reset()

print("joint pos:")
print(env.get_joint_pos()[env.isaac2mujoco])

actions = get_action(obs.to(device), True)
next_obs = env.step(actions)

print("obs:")
print(obs)
print("-------")
print("actions:")
print(actions)
print("-------")
print("next obs:")
print(next_obs)
print("-------")
print("ori:")
print(env.get_projected_gravity())
print("-------")
print("ang vel:")
print(env.get_base_ang_vel())
print("-------")
print("joint pos:")
print(env.get_joint_pos())
print("-------")
print("joint vel:")
print(env.get_joint_vel())
print("-------")
print("pre action:")
print(env.previous_action)
print("--------")
#'''
env.close()
