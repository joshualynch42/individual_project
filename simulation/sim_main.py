from sim_utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

# starting_letter = np.array('F')
# goal_letter = np.array('N')
# env = discrete_alphabet_env(starting_letter, goal_letter)

env = discrete_arrow_env()
agent = DQNAgent(env, epsilon_decay=0.999)

# agent.load_model('ep1_mb32_rms150_mrm100')
episodes = 1000
reward_arr = []
success_arr = np.array([0]*episodes)

for episode in range(episodes):
    reward_tot = 0
    done = False
    success = False
    steps = 1
    current_state = env.reset()
    while not done:
        action = agent.act(current_state)
        new_state, reward, done, info = env.step(action, steps)
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        reward_tot += reward

        if reward == 1:
            success = True
            success_arr[episode] = 1
        agent.train(done)
        current_state = new_state
        steps += 1

    reward_arr.append(reward_tot)
    print('episode: {}, reward: {}, epsilon: {}, steps: {}, start letter: {}, '
                                'goal_letter: {}, end_letter: {}, success: {}'.
    format(episode, reward_tot, agent.epsilon, steps, env.starting_letter,
                env.goal_letter, coords_to_letter(env.current_coords), success))

agent.save_model(episodes)


colormap = np.array(['r', 'g'])
x = np.linspace(1, episodes, episodes)
plt.scatter(x, reward_arr, c=colormap[success_arr])
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
