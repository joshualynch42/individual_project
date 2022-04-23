import numpy as np
import gym
from sim_utils import *
from pong_agent import *
import time
from tqdm import tqdm
from gym_wrapper import *

rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 32,
'epsilon_decay': 0.9999,
'discount': 0.99,
'min_replay_memory_size': 10000,
'min_epsilon': 0.02,
'epsilon': 1,
'update_target_every': 10,
'episodes': 1000,
'action_space_size': 6
}

env = make_env('PongNoFrameskip-v4')

show_every = 5

avg_rew_size = 50
save_model_bool = True
environment = 'pong'

agent = PongAgent(rl_params)

reward_arr = []
episodes = rl_params['episodes']
for episode in range(1, episodes):
    start_time = time.time()
    done = False
    reward_tot = 0
    steps = 0

    # if episode % show_every == 0:
    #     env = make_env('PongNoFrameskip-v4', render_mode='human')
    # else:
    #     env = make_env('PongNoFrameskip-v4')

    current_state  = env.reset()
    # current_state  = process_image(current_state)
    # current_state = np.dstack((current_state, current_state, current_state, current_state))
    while not done:
        steps += 1
        action = agent.act(current_state)
        new_state, reward, done, info = env.step(action)
        reward_tot += reward # episode rewards

        # new_state = process_image(new_state)
        # new_state = np.dstack((new_state, current_state[:, :, 0], current_state[:, :, 1], current_state[:, :, 2]))

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state

    reward_arr.append(reward_tot)
    end_time = time.time()
    print('episode: {}, reward: {}, epsilon: {}, steps: {}, done: {}, time: {}, avg reward: {}'.
    format(episode, reward_tot, round(agent.epsilon, 3), steps, done,
            round(end_time-start_time, 3), round(sum(reward_arr[-50:])/len(reward_arr[-50:]), 3)))

if save_model_bool == True:
    model_dir = "D:\Josh\github\individual_project\simulation\sim_agents\{}_{}.h5".format(environment, agent.label)
    agent.save_model(model_dir)

avg_reward_arr = [] # calcualting moving average
for i in range(len(reward_arr) - avg_rew_size +1):
    this_window = reward_arr[i : i + avg_rew_size]
    window_average = sum(this_window) / avg_rew_size
    avg_reward_arr.append(window_average)

x = np.linspace(avg_rew_size, episodes, episodes-avg_rew_size+1)
plt.plot(x, avg_reward_arr, label=agent.label)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
