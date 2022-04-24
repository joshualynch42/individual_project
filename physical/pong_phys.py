import sys
sys.path.insert(1, 'D:/Users/Josh/github/individual_project/simulation')
from dueling_ddqn_per import *
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from phys_utils import *
import random
import time

## PONG AGENT AND ENV ###
env_p = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
env_p = VecFrameStack(env_p, n_stack=4)
model = PPO.load("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent_after", env=env_p)

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
environment = 'arrow' # alphabet, arrow

if environment == 'arrow':
    current_letter = 'DOWN'
    goal_letter = 'DOWN'
    action_to_key_arr = ['DOWN', 'DOWN', 'RIGHT', 'LEFT', 'RIGHT', 'LEFT']
    key_to_action_dict = {'DOWN': [1], 'RIGHT': [4], 'LEFT': [5]}
    env_r = phys_discrete_arrow_env_pong('key_images', current_letter, goal_letter)
    agent_dir = "D:/Users/Josh/github/individual_project/physical/phys_agents/arrow_no_sim_phys_Dueling Double Per.h5"
elif environment == 'alphabet':
    current_letter = 'D'
    goal_letter = 'D'
    action_to_key_arr = ['D', 'C', 'F', 'S', 'R', 'E']
    key_to_action_dict = {'D': [0], 'C': [1], 'F': [2], 'S': [3], 'R': [4], 'E': [5]}
    env_r = phys_discrete_alphabet_env_pong('key_images', current_letter, goal_letter)
    agent_dir = "D:/Users/Josh/github/individual_project/physical/phys_agents/alphabet_no_sim_phys_Dueling Double Per_new.h5"
else:
    print('incorrect environment argument')
    exit()

agent = Dueling_Per_DDQNAgent(env_r, rl_params)
agent.load_model(agent_dir)

actions_per_predict = 4

dones = [False]
obs = env_p.reset()

steps_time_arr = []
ep_start_time = time.time()
while not dones[0]:
# for i in range(500):
    step_start_time = time.time()
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
    print('goal letter is ', goal_letter)
    print('current letter is ', current_letter)
    if current_letter in key_to_action_dict:
        action_p = key_to_action_dict[current_letter]
        print('actual')
    else:
        action_p = [random.randint(0, len(key_to_action_dict)-1)]
        print('random')
    for i in range(actions_per_predict):
        if dones[0] == True:
            break
        else:
            obs, rewards, dones, info = env_p.step(action_p)
            env_p.render()
    step_end_time = time.time()
    steps_time_arr.append(step_end_time-step_start_time)

eps_end_time = time.time()

print('The average step time was ', round(sum(steps_time_arr) / len(steps_time_arr), 3))
print('The total episode time was ', round(eps_end_time - ep_start_time, 3))
