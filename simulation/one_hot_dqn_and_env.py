import pandas as pd
import numpy as np
from gym import Env, spaces, make
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input, BatchNormalization
from keras.optimizers import adam_v2
from collections import deque
import random, os
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
class one_hot_DQNAgent():
    def __init__(self, env, epsilon_decay):
        self.replay_memory_size = 10000
        self.discount = 0.95
        self.min_replay_memory_size = 200
        self.minibatch_size = 32
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.001
        self.action_space = env.action_space_size
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.epsilon = 1
        self.model = self.create_model(env)

    def create_model(self, env):

        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=env.img_shape[0]))
        model.add(Dropout(0.15))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # current_state, action, reward, new_state, done
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def act(self, current_state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.get_qs(current_state))
        else:
            action = np.random.randint(0, self.action_space)

        # Decay Epsilon
        if len(self.replay_memory) > self.min_replay_memory_size:
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)
        return action

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        ## current states = [dict{img, vec}, dict{img, vec}, dict{img, vec}......]
        ## seperating dicts into arrays
        current_states = np.array([transition[0] for transition in minibatch])  # Add divide by max (scale results)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])  # Add divide by max (scale results)
        future_qs_list = self.model.predict(new_current_states)

        x = []
        y = []  # array of new q values

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

    def render(self, reward_arr):
        pass

    def save_model(self, episodes):
        self.model.save("D:/Josh/github/individual_project/simulation/sim_agents/ep{}_mb{}_rms{}_mrm{}.h5".format(episodes,
                                                                                self.minibatch_size,self.replay_memory_size,
                                                                                self.min_replay_memory_size))

        print('Agent saved as ep{}_mb{}_rms{}_mrm{}.h5'.format(episodes, self.minibatch_size, self.replay_memory_size,
                                                                self.min_replay_memory_size))

    def load_model(self, model_name):
        self.model = keras.models.load_model("D:/Josh/github/individual_project/simulation/sim_agents/{}.h5".format(model_name))
        print('Agent {} has loaded'.format(model_name))

class one_hot_discrete_arrow_env(Env):
    def __init__(self):
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.goal_letter = random.choice(arrow_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        self.goal_coords = letter_to_coords(self.goal_letter)
        self.action_space_size = 5
        self.action_space = spaces.Discrete(self.action_space_size)
        self.action_array = np.array(['up', 'left', 'right', 'down', 'pressdown'])
        self.state = np.concatenate((create_one_hot(self.starting_letter), create_one_hot(self.goal_letter)))
        self.img_shape = np.shape(self.state)
        self.max_ep_len = 25

    def reset(self):
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.goal_letter = random.choice(arrow_arr)
        self.state = np.concatenate((create_one_hot(self.starting_letter), create_one_hot(self.goal_letter)))
        self.current_coords = letter_to_coords(self.starting_letter)
        self.goal_coords = letter_to_coords(self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary

        if action == 'up':  # all values become up key
            if self.current_coords[0] != 2 or self.current_coords[1] != 43:
                self.current_coords = np.array([2, 43, -37])

        elif action == 'right':
            if self.current_coords[1] == 43:  # up and down become right
                self.current_coords = np.array([3, 46, -37])
            elif self.current_coords[1] == 40:
                self.current_coords = np.array([3, 43, -37])

        elif action == 'left':
            if self.current_coords[1] == 43:  # up and down become right
                self.current_coords = np.array([3, 40, -37])
            elif self.current_coords[1] == 46:
                self.current_coords = np.array([3, 43, -37])

        elif action == 'down':
            if self.current_coords[0] == 2 and self.current_coords[1] == 43:
                self.current_coords = np.array([3, 43, -37])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        self.state = np.concatenate((create_one_hot(coords_to_letter(self.current_coords)), create_one_hot(self.goal_letter)))

    def step(self, action, steps):
        if steps > self.max_ep_len -1:
            done = True
            reward = 0
            info = []
        else:
            action = self.action_array[action]
            done = False
            info = []
            if action == 'pressdown':
                done = True
                if np.array_equal(self.goal_coords, self.current_coords, equal_nan=False):
                    reward = 1
                else:
                    reward = -1
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass
