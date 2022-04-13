from sim_utils import *
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

# agent.load_model('ep1_mb32_rms150_mrm100')

rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 64,
'epsilon_decay': 0.9995, # for alphabet
#'epsilon_decay': 0.999, # for arrows
'discount': 0.95,
'min_replay_memory_size': 200,
'min_epsilon': 0.001,
'epsilon': 1,
'update_target_every': 1,
'episodes': 6500
}

key_image_loc = 'key_images'
# key_image_loc = 'alex_key_images'

avg_rew_size = 50

env = discrete_alphabet_env(key_image_loc)
agent = Dueling_Per_DDQNAgent(env, rl_params)

reward_arr = []
episodes = rl_params['episodes']
success_arr = np.array([0]*episodes)
for episode in range(episodes):
    reward_tot = 0
    done = False
    success = False
    steps = 0
    current_state = env.reset(key_image_loc)
    hindsight_buffer = her()
    while not done and steps < env.max_ep_len:
        steps += 1
        starting_let = coords_to_letter(env.current_coords)
        action = agent.act(current_state)
        new_state, reward, done, info = env.step(action, steps)
        done = False if steps == env.max_ep_len else done
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        hindsight_buffer.update_her_buffer((current_state, action, reward, new_state, done))
        reward_tot += reward

        # print('start letter: {}, action: {}, current_letter: {}'. format(starting_let, env.action_array[action], coords_to_letter(env.current_coords, file_na)))

        if reward == 1:
            success = True
            success_arr[episode] = 1

        agent.train(done)
        current_state = new_state

    if reward_tot == 0 and done == True and steps != env.max_ep_len: # HER
        # Assume final key was goal key for HER
        # find out current key
        final_letter = coords_to_letter(env.current_coords)
        # turn current key into one one_hot
        one_hot = create_one_hot(final_letter)
        # iterate through all steps in episode
        for step in range(steps):
            # change goals and rewards from old transition to new goal
            transition = hindsight_buffer.update_transition(step, one_hot, steps)
            # add new values to replay memory
            agent.update_replay_memory(transition)

    reward_arr.append(reward_tot)

    print('episode: {}, reward: {}, epsilon: {}, steps: {}, start letter: {}, '
                                'goal_letter: {}, end_letter: {}, success: {}, done: {}'.
    format(episode, reward_tot, round(agent.epsilon, 3), steps, env.starting_letter,
                env.goal_letter, coords_to_letter(env.current_coords), success, done))

# print('beta is ', agent.experience_replay.beta)

# calculating moving average of reward array
avg_reward_arr = [] # calcualting moving average
for i in range(len(reward_arr) - avg_rew_size +1):
    this_window = reward_arr[i : i + avg_rew_size]
    window_average = sum(this_window) / avg_rew_size
    avg_reward_arr.append(window_average)

x = np.linspace(avg_rew_size, episodes, episodes-avg_rew_size+1)
plt.plot(x, avg_reward_arr, label=agent.label)

agent.save_model(episodes)

# colormap = np.array(['r', 'g'])
# x = np.linspace(1, episodes, episodes)
# plt.scatter(x, reward_arr, c=colormap[success_arr])
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
