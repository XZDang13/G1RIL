from env.mujoco_env import MujocoEnv

env = MujocoEnv(1/200)

obs = env.reset()