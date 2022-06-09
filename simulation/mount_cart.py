import numpy as np
import gym
import h5py

env = gym.make('MountainCar-v0')

learning_rate = 0.1
discount = 0.95
episodes = 25000
show_every = 3000
epsilon = 1
start_ep_decay = 1
end_ep_decay = episodes / 2
epsilon_decay = epsilon / (end_ep_decay - start_ep_decay)
skip_actions = 20

discrete_os_size = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size

q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

for episode in range(1, episodes+1):
    discrete_state = get_discrete_state(env.reset())
    done = False
    skip_counter = 0

    if episode % show_every == 0:
        print('Episode is ', episode)

    while not done:
        # Agent acting #
        if skip_counter % skip_actions == 0:
            skip_counter = 0
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % show_every == 0:
            env.render()

        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
        skip_counter += 1

    # Decaying is being done every episode if episode number is within decaying range
    if end_ep_decay >= episode >= start_ep_decay:
        epsilon -= epsilon_decay

hf = h5py.File('D:\Josh\github\individual_project\simulation\sim_agents\mount_cart.h5', 'w')
hf.create_dataset('q_table', data=q_table)
hf.close()
