import numpy as np
import gym
import h5py
import time

env = gym.make('MountainCar-v0')
skip_actions = 20
discrete_os_size = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

hf = h5py.File('D:\Josh\github\individual_project\simulation\sim_agents\mount_cart.h5', 'r')
n1 = hf.get('q_table')
q_table = np.array(n1)

for episode in range(1, 3):
    discrete_state = get_discrete_state(env.reset())
    done = False
    skip_counter = 0

    print('Episode is ', episode)

    while not done:
        # Agent acting #
        if skip_counter % skip_actions == 0:
            skip_counter = 0
            action = np.argmax(q_table[discrete_state])
            time.sleep(3)

        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        env.render()

        discrete_state = new_discrete_state
        skip_counter += 1
