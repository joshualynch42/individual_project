from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import tensorflow as tf
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)
model = PPO.load("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent_after", env=env)
# model.learn(total_timesteps=1000000)

# model.save("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent_before")
# print('saved model')

# start_time = time.time()
# total_timesteps = 0
# save_every = 1000
# save_counter = 0
# obs = env.reset()
# while total_timesteps < 22000:
#     action, _states = model.predict(obs)
#     for i in range(4):
#         obs, rewards, dones, info = env.step(action)
#         model.train()
#         total_timesteps += 8
#         save_counter += 8
#     if save_counter > save_every:
#         save_counter -= save_every
#         model.save("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent_after")
#         print('saved model | Total steps: ', total_timesteps)
#
# end_time = time.time()
# print('time taken is ', (end_time - start_time))
# model.save("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent_after")

dones = [False]
obs = env.reset()
while not dones[0]:
    action, _states = model.predict(obs)
    for i in range(4):
        if dones[0] == True:
            break
        else:
            obs, rewards, dones, info = env.step(action)
            env.render()
