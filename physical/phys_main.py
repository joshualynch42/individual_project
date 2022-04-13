import sys
sys.path.insert(1, 'D:/Josh\github/individual_project/simulation')
from dueling_ddqn_per import *
from phys_utils import *

def make_sensor(): # amcap: reset all settings; autoexposure off; saturdation max
    camera = CvPreprocVideoCamera(source=0,  # might need changing for webcam
                crop=[320-128-10, 240-128+10, 320+128-10, 240+128+10],
                size=[128, 128],
                threshold=[61, -5],
                exposure=-6)
    for _ in range(5): camera.read() # Hack - camera transient

    return AsyncProcessor(CameraStreamProcessor(camera=camera,
                display=CvVideoDisplay(name='sensor'),
                writer=CvImageOutputFileSeq()))

sensor = make_sensor()
robot = SyncRobot(Controller())
robot.linear_speed = 40
robot.coord_frame = [0, 0, 0, 0, 0, 0] # careful

rl_params = {
'replay_memory_size': 10000,
'minibatch_size': 64,
'epsilon_decay': 0.9995, # for alphabet
#'epsilon_decay': 0.999, # for arrows
'discount': 0.95,
'min_replay_memory_size': 200,
'min_epsilon': 0.001,
'epsilon': 0,
'update_target_every': 1,
'episodes': 500
}

avg_rew_size = 1

env = phys_discrete_alphabet_env()
agent = Dueling_Per_DDQNAgent(env, rl_params)
agent.load_model('alphabet_Dueling Double Per')

reward_arr = []
episodes = rl_params['episodes']
success_arr = np.array([0]*episodes)
for episode in range(episodes):
    reward_tot = 0
    done = False
    success = False
    steps = 0
    current_state = env.reset()
    while not done and steps < env.max_ep_len:
        steps += 1
        starting_let = coords_to_letter(env.current_coords)
        action = agent.act(current_state)
        # print('start letter: {}, action: {}'.format(starting_let, env.action_array[action]))
        new_state, reward, done, info = env.step(action, steps)
        done = False if steps == env.max_ep_len else done
        reward_tot += reward

        # print('new_letter: {}'.format(coords_to_letter(env.current_coords)))

        if reward == 1:
            success = True
            success_arr[episode] = 1

        current_state = new_state

    reward_arr.append(reward_tot)

    print('episode: {}, reward: {}, epsilon: {}, steps: {}, start letter: {}, '
                                'goal_letter: {}, end_letter: {}, success: {}, done: {}'.
    format(episode, reward_tot, round(agent.epsilon, 3), steps, env.starting_letter,
                env.goal_letter, coords_to_letter(env.current_coords), success, done))

# calculating moving average of reward array
avg_reward_arr = [] # calcualting moving average
for i in range(len(reward_arr) - avg_rew_size +1):
    this_window = reward_arr[i : i + avg_rew_size]
    window_average = sum(this_window) / avg_rew_size
    avg_reward_arr.append(window_average)

x = np.linspace(avg_rew_size, episodes, episodes-avg_rew_size+1)
plt.plot(x, avg_reward_arr, label=agent.label)

plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
