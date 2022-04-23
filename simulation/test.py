from test_utils import *
import gym
import time

env = gym.make('BreakoutDeterministic-v4') # render_mode = 'human'
frame = env.reset()

rl_params = {
'replay_memory_size': 10000,
'min_replay_memory_size': 2000,
'minibatch_size': 32,
'epsilon_decay': 0.999, # for alphabet
'discount': 0.95,
'min_epsilon': 0.001,
'epsilon': 1,
'update_target_every': 10,
'episodes': 200,
'action_space_size': 4,
'input_shape': (210, 160, 3)
}
agent = Double_DQNAgent(env, rl_params)

for episode in range(rl_params['episodes']):
    done = False
    start_time = time.time()
    current_state = env.reset()
    steps = 0
    tot_reward = 0
    while not done:
        steps += 1
        # Perform a random action, returns the new frame, reward and whether the game is over
        action = agent.act(current_state)
        new_state, reward, done, info = env.step(env.action_space.sample())
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)
        tot_reward += reward
        current_state = new_state

    end_time = time.time()
    print('episode: {}, epsilon: {}, steps: {}, reward: {}, time: {}'.
    format(episode, round(agent.epsilon, 3), steps, tot_reward, round(end_time - start_time, 3)))
