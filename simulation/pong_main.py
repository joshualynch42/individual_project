import numpy as np
import gym
from sim_utils import *
from pong_agent import *
# gym initialization

rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 64,
'epsilon_decay': 0.995,
'discount': 0.95,
'min_replay_memory_size': 200,
'min_epsilon': 0.001,
'epsilon': 1,
'update_target_every': 1,
'episodes': 50
}

act_arr = [0, 2, 5]

env = gym.make('ALE/Pong-v5', render_mode='human')
state = env.reset()
agent = PongAgent(rl_params)

for episode in range(1):
    done = False
    steps = 0
    current_state = env.reset()
    current_state = process_image(current_state)
    print(np.shape(current_state))
    while not done:
        steps += 1
        action = agent.act(current_state)
        actual_act = act_arr[action] # required as env's actions are 0,2 and 5
        new_state, reward, done, info = env.step(actual_act)
        new_state = process_image(new_state)
        agent.update_replay_memory((current_state, action, reward, new_state, done))

        agent.train(done)
        current_state = new_state
