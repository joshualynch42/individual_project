from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from sim_utils import *

## PONG AGENT AND ENV ###
env_p = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
env_p = VecFrameStack(env_p, n_stack=4)
model = PPO.load("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent", env=env_p)

## SIM AGENT AND ENV ##
rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 64,
'epsilon_decay': 0, # for alphabet
'discount': 0,
'min_replay_memory_size': 200,
'min_epsilon': 0,
'epsilon': 0,
'update_target_every': 1,
'episodes': 10
}
current_letter = 'DOWN'
goal_letter = 'DOWN'
env_r = discrete_arrow_env_pong('key_images', current_letter, goal_letter)
agent = Dueling_Per_DDQNAgent(env_r, rl_params)
agent.load_model("D:/Josh/github/individual_project/simulation/sim_agents/arrow_Dueling Double Per.h5")

## INTEGRATION ##
# pong action to desired key
action_to_key_arr = ['DOWN', 'DOWN', 'RIGHT', 'LEFT', 'RIGHT', 'LEFT']
key_to_action_dict = {'DOWN': [1], 'RIGHT': [4], 'LEFT': [5]}
actions_per_predict = 1

dones = [False]
obs = env_p.reset()

while not dones[0]:
# for i in range(500):
    action_p, _states = model.predict(obs, deterministic=True) # pong agent predicts action
    goal_letter = action_to_key_arr[action_p[0]] # action to goal key
    current_state = env_r.reset('key_images', current_letter, goal_letter)
    done = False
    steps = 0
    while not done:
        steps += 1
        action_r = agent.act(current_state) # predict robot action
        new_state, reward, done, _ = env_r.step(action_r, steps) # execute action
        current_state = new_state # update state
    current_letter = coords_to_letter(env_r.current_coords)
    action_p = key_to_action_dict[current_letter]
    for i in range(actions_per_predict):
        obs, rewards, dones, info = env_p.step(action_p)
        env_p.render()
