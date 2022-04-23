import sys
sys.path.insert(1, 'D:/Users/Josh/github/individual_project/simulation')
from dueling_ddqn_per import *
from phys_utils import *
import time
from transfer_dueling_ddqn_per import *

rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 64,
'epsilon_decay': 0.99, # for alphabet
'discount': 0.95,
'min_replay_memory_size': 200,
'min_epsilon': 0.001,
'epsilon': 0,
'update_target_every': 1,
'episodes': 200
}

### CHANGED DELTA Z TO 35.5 FROM 36

# TIMING ##
total_time_dict = {}
time_per_ep_dict = {}
time_per_step_dict = {}
time_to_converge_dict = {}
acc_300_dict = {}
accuracy_point = 1

# SAVE MODEL??????
train_type = 'test_transfer_learning'
environment = 'alphabet'


### TIMING ###
start_time = time.time()
converge_time = time.time()
converge_bool = False
total_steps = 0

if environment == 'arrow':
    env = phys_discrete_arrow_env()
    rl_params['epsilon_decay'] = 0.995

else:
    env = phys_discrete_alphabet_env()
    rl_params['epsilon_decay'] = 0.9995

model_dir = "D:/Users/Josh/github/individual_project/simulation/sim_agents/{}_Dueling Double Per.h5".format(environment)

if train_type == 'zero_shot':
    rl_params['epsilon'] = 0
    rl_params['episodes'] = 100
    avg_rew_size = int(rl_params['episodes'] / 5)
    agent = Dueling_Per_DDQNAgent(env, rl_params)
    agent.load_model(model_dir)
    HER = False
elif train_type == 'transfer_learning':
    rl_params['epsilon'] = 0.5
    rl_params['episodes'] = 1000
    avg_rew_size = int(rl_params['episodes'] / 10)
    agent = Dueling_Per_DDQNAgent(env, rl_params)
    agent.load_model(model_dir)
    HER = True
elif train_type == 'no_sim':
    rl_params['epsilon'] = 1
    rl_params['episodes'] = 2000
    avg_rew_size = int(rl_params['episodes'] / 10)
    agent = Dueling_Per_DDQNAgent(env, rl_params)
    HER = True
elif train_type == 'test_transfer_learning':
    rl_params['epsilon'] = 0
    rl_params['episodes'] = 20
    avg_rew_size = int(rl_params['episodes'] / 5)
    agent = Dueling_Per_DDQNAgent(env, rl_params)
    model_dir = "D:/Users/Josh/github/individual_project/physical/phys_agents/{}_transfer_learning_phys_Dueling Double Per_new.h5".format(environment)
    # agent.save_model(model_dir)
    agent.load_model(model_dir)
    HER = False
elif train_type == 'test_no_sim':
    rl_params['epsilon'] = 0
    rl_params['episodes'] = 100
    avg_rew_size = int(rl_params['episodes'] / 5)
    agent = Dueling_Per_DDQNAgent(env, rl_params)
    model_dir = "D:/Users/Josh/github/individual_project/physical/phys_agents/{}_no_sim_phys_Dueling Double Per_new.h5".format(environment)
    agent.load_model(model_dir)
    HER = False
else:
    print('No valid train type selected')
    exit()

reward_arr = []
episodes = rl_params['episodes']
failed_goals = []
failed_ends = []
failed_starts = []

for episode in range(episodes):
    reward_tot = 0
    done = False
    success = False
    steps = 0
    current_state = env.reset()
    hindsight_buffer = her()
    while not done and steps < env.max_ep_len:
        steps += 1
        starting_let = coords_to_letter(env.current_coords)
        action = agent.act(current_state)
        new_state, reward, done, info = env.step(action, steps)
        done = False if steps == env.max_ep_len else done
        reward_tot += reward
        # print('new_letter: {}'.format(coords_to_letter(env.current_coords)))

        if train_type == 'transfer_learning' or train_type == 'no_sim':
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            hindsight_buffer.update_her_buffer((current_state, action, reward, new_state, done))
            agent.train(done)
        current_state = new_state

    if reward_tot == 0:
        failed_goals.append(env.goal_letter)
        failed_ends.append(coords_to_letter(env.current_coords))
        failed_starts.append(starting_let)

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

    reward_arr.append(reward_tot)
    total_steps += steps

    ### CHECK CONVERGE ### 95%
    if len(reward_arr) > 20 and converge_bool == False:
        mean = sum(reward_arr[episode-20: episode]) / 20
        if mean >= 0.95:
            converge_bool = True
            converge_time = time.time()

    print('episode: {}, reward: {}, epsilon: {}, steps: {}, start letter: {}, '
                                'goal_letter: {}, end_letter: {}, done: {}'.
    format(episode, reward_tot, round(agent.epsilon, 3), steps, env.starting_letter,
                env.goal_letter, coords_to_letter(env.current_coords), done))

### TIMINGS ###
end_time = time.time()
time_elapsed = end_time - start_time
total_time_dict[agent.label] = time_elapsed
time_per_ep_dict[agent.label] = time_elapsed/episodes
time_per_step_dict[agent.label] = time_elapsed/total_steps
time_to_converge_dict[agent.label] = converge_time - start_time
acc_300_dict[agent.label] = sum(reward_arr[accuracy_point-1:-1]) / (episodes - accuracy_point)

# calculating moving average of reward array
avg_reward_arr = [] # calcualting moving average
for i in range(len(reward_arr) - avg_rew_size +1):
    this_window = reward_arr[i : i + avg_rew_size]
    window_average = sum(this_window) / avg_rew_size
    avg_reward_arr.append(window_average)


# model_dir = "D:/Users/Josh/github/individual_project/physical/phys_agents/{}_{}_phys_Dueling Double Per_new.h5".format(environment, train_type)
# agent.save_model(model_dir)

x = np.linspace(avg_rew_size, episodes, episodes-avg_rew_size+1)
plt.plot(x, avg_reward_arr, label=agent.label)

### PRINT TIMINGS ###
print('Total time ', total_time_dict)
print('Time per episode ', time_per_ep_dict)
print('Time per step ', time_per_step_dict)
print('Time to converge ', time_to_converge_dict)
print('Accuracy ', acc_300_dict)

print('Failed goals ', failed_goals)
print('failed ends ', failed_ends)
print('failed starts ', failed_starts)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

### Initial transfer learning
# Total time  {'Dueling Double Per': 6176.979035139084}
# Time per episode  {'Dueling Double Per': 30.88489517569542}
# Time per step  {'Dueling Double Per': 3.441213947152693}
# Time to converge  {'Dueling Double Per': 0.0}
# Accuracy  {'Dueling Double Per': 0.62}

### arrow zero-shot ###
# Total time  {'Dueling Double Per': 688.1160800457001}
# Time per episode  {'Dueling Double Per': 6.881160800457001}
# Time per step  {'Dueling Double Per': 3.1564957800261473}
# Time to converge  {'Dueling Double Per': 157.23517990112305}

### arrow no sim ### training
# Total time  {'Dueling Double Per': 5399.491830587387}
# Time per episode  {'Dueling Double Per': 10.798983661174773}
# Time per step  {'Dueling Double Per': 3.20064720248215}
# Time to converge  {'Dueling Double Per': 2323.665095090866}
# Accuracy  {'Dueling Double Per': 0.8717434869739479}

### arrow no sim ### post training
# Total time  {'Dueling Double Per': 647.8709669113159}
# Time per episode  {'Dueling Double Per': 6.478709669113159}
# Time per step  {'Dueling Double Per': 3.3568443881415333}
# Time to converge  {'Dueling Double Per': 132.89744806289673}
# Accuracy  {'Dueling Double Per': 1.0}

### alphabet no sim first 1000 train ###
# Total time  {'Dueling Double Per': 26521.448010206223}
# Time per episode  {'Dueling Double Per': 26.52144801020622}
# Time per step  {'Dueling Double Per': 3.188823855982472}
# Time to converge  {'Dueling Double Per': 20024.82089948654}
# Accuracy  {'Dueling Double Per': 0.43743743743743746}

### alphabet transfer learning first 1000 ###
# Total time  {'Dueling Double Per': 19175.530752420425}
# Time per episode  {'Dueling Double Per': 19.175530752420425}
# Time per step  {'Dueling Double Per': 3.345931033400877}
# Time to converge  {'Dueling Double Per': 6765.873987197876}
# Accuracy  {'Dueling Double Per': 0.8158158158158159}

### alphabet no sim 2000 eps ###
# Total time  {'Dueling Double Per': 46297.913121938705}
# Time per episode  {'Dueling Double Per': 23.148956560969353}
# Time per step  {'Dueling Double Per': 3.3195607027990754}
# Time to converge  {'Dueling Double Per': 24581.400357484818}
# Accuracy  {'Dueling Double Per': 0.6988494247123562}

### alphabet no sim test ###
# Total time  {'Dueling Double Per': 1763.476628780365}
# Time per episode  {'Dueling Double Per': 17.63476628780365}
# Time per step  {'Dueling Double Per': 3.247654933297173}
# Time to converge  {'Dueling Double Per': 425.55912685394287}
# Accuracy  {'Dueling Double Per': 0.9595959595959596}
# Failed goals  ['B', 'B', 'V', 'K']
# failed ends  ['S', 'B', 'X', 'L']
# failed starts  ['S', 'B', 'X', 'L']

### alphabet transfer_learning test 20 eps ###
# Total time  {'Dueling Double Per': 289.1785092353821}
# Time per episode  {'Dueling Double Per': 14.458925461769104}
# Time per step  {'Dueling Double Per': 3.5265671857973424}
# Time to converge  {'Dueling Double Per': 0.0}
# Accuracy  {'Dueling Double Per': 1.0}
