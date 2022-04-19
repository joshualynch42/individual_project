from sim_utils import *
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

plt.rcParams.update({'font.size': 18})

# agent.load_model('ep1_mb32_rms150_mrm100')

rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 64,
<<<<<<< Updated upstream
<<<<<<< Updated upstream
'epsilon_decay': 0.9995, # for alphabet
# 'epsilon_decay': 0.995, # for arrows
=======
# 'epsilon_decay': 0.9995, # for alphabet
'epsilon_decay': 0.995, # for arrows
>>>>>>> Stashed changes
=======
# 'epsilon_decay': 0.9995, # for alphabet
'epsilon_decay': 0.995, # for arrows
>>>>>>> Stashed changes
'discount': 0.95,
'min_replay_memory_size': 200,
'min_epsilon': 0.001,
'epsilon': 1,
'update_target_every': 1,
<<<<<<< Updated upstream
<<<<<<< Updated upstream
'episodes': 4000
# 'episodes': 500
=======
'episodes': 500
>>>>>>> Stashed changes
=======
'episodes': 500
>>>>>>> Stashed changes
}

key_image_loc = 'key_images'
# key_image_loc = 'alex_key_images'

<<<<<<< Updated upstream
avg_rew_size = 200
# avg_rew_size = 50

# TIMING ##
total_time_dict = {}
time_per_ep_dict = {}
time_per_step_dict = {}
time_to_converge_dict = {}
acc_300_dict = {}
accuracy_point = 3000

for iii in range(4):
    start_time = time.time()
    converge_time = time.time()
    converge_bool = False
    total_steps = 0
    env = discrete_alphabet_env(key_image_loc)
    if iii == 0:
        agent = Dueling_Per_DDQNAgent(env, rl_params)
    elif iii == 1:
        agent = Dueling_DDQNAgent(env, rl_params)
    elif iii == 2:
        agent = Double_DQNAgent(env, rl_params)
    else:
        agent = DQNAgent(env, rl_params)

    HER = True
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

        if reward_tot == 0 and done == True and steps != env.max_ep_len and HER == True: # HER
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

        total_steps += steps
        reward_arr.append(reward_tot)

        ### CHECK CONVERGE ### 95%
        if len(reward_arr) > 20 and converge_bool == False:
            mean = sum(reward_arr[episode-20: episode]) / 20
            if mean >= 0.95:
                converge_bool = True
                converge_time = time.time()

        print('episode: {}, reward: {}, epsilon: {}, steps: {}, start letter: {}, '
                                    'goal_letter: {}, end_letter: {}, success: {}, done: {}'.
        format(episode, reward_tot, round(agent.epsilon, 3), steps, env.starting_letter,
                    env.goal_letter, coords_to_letter(env.current_coords), success, done))

    ### TIMINGS ###
    end_time = time.time()
    time_elapsed = end_time - start_time
    total_time_dict[agent.label] = time_elapsed
    time_per_ep_dict[agent.label] = time_elapsed/episodes
    time_per_step_dict[agent.label] = time_elapsed/total_steps
    time_to_converge_dict[agent.label] = converge_time - start_time
    acc_300_dict[agent.label] = sum(reward_arr[accuracy_point:-1]) / (episodes - accuracy_point)

    # calculating moving average of reward array
    avg_reward_arr = [] # calcualting moving average
    for i in range(len(reward_arr) - avg_rew_size +1):
        this_window = reward_arr[i : i + avg_rew_size]
        window_average = sum(this_window) / avg_rew_size
        avg_reward_arr.append(window_average)

    x = np.linspace(avg_rew_size, episodes, episodes-avg_rew_size+1)
    plt.plot(x, avg_reward_arr, label=agent.label)

# agent.save_model(episodes)

### PRINT TIMINGS ###
print('Total time ', total_time_dict)
print('Time per episode ', time_per_ep_dict)
print('Time per step ', time_per_step_dict)
print('Time to converge ', time_to_converge_dict)
print('Accuracy ', acc_300_dict)
=======
avg_rew_size = 50

env = discrete_arrow_env(key_image_loc)
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
>>>>>>> Stashed changes

# colormap = np.array(['r', 'g'])
# x = np.linspace(1, episodes, episodes)
# plt.scatter(x, reward_arr, c=colormap[success_arr])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

# Total time  {'Dueling Double Per': 285.6249439716339, 'Dueling Double': 236.0352807044983, 'Double': 287.8598966598511, 'DQN': 257.68923568725586}
# Time per episode  {'Dueling Double Per': 0.5712498879432678, 'Dueling Double': 0.47207056140899656, 'Double': 0.5757197933197021, 'DQN': 0.5153784713745118}
# Time per step  {'Dueling Double Per': 0.1381833304168524, 'Dueling Double': 0.13426352713566456, 'Double': 0.1313229455565014, 'DQN': 0.13120633181632171}
# Time to converge  {'Dueling Double Per': 285.62394404411316, 'Dueling Double': 236.03425359725952, 'Double': 287.8588993549347, 'DQN': 257.6876378059387}
# Accuracy  {'Dueling Double Per': 0.995, 'Dueling Double': 0.99, 'Double': 0.995, 'DQN': 0.99}

# alphabet_Total time  {'Dueling Double Per': 5427.999376535416, 'Dueling Double': 5110.121971130371, 'Double': 5484.142999410629, 'DQN': 25183.189202070236}
# Time per episode  {'Dueling Double Per': 1.356999844133854, 'Dueling Double': 1.2775304927825928, 'Double': 1.3710357498526573, 'DQN': 6.295797300517559}
# Time per step  {'Dueling Double Per': 0.15533867660290804, 'Dueling Double': 0.14241066719979853, 'Double': 0.14241198159938273, 'DQN': 0.674537665454284}
# Time to converge  {'Dueling Double Per': 5427.998868703842, 'Dueling Double': 5110.121495962143, 'Double': 5484.142081737518, 'DQN': 25183.187163591385}
# Accuracy  {'Dueling Double Per': 0.992, 'Dueling Double': 0.99, 'Double': 0.991, 'DQN': 0.986}
