import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input, Convolution2D
from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import Adam
from collections import deque
import random, os
import tensorflow as tf
import cv2
import gym

image_size = (84, 84)
image_size_net = (4, 84, 84)

def process_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state[20:300, 0:200]  # crop off score
    state = cv2.resize(state, dsize=image_size)
    return state / 255

def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1

class PongAgent():
    def __init__(self, rl_params):
        self.label = 'Dueling Double'
        self.replay_memory_size = rl_params['replay_memory_size']
        self.discount = rl_params['discount']
        self.min_replay_memory_size = rl_params['min_replay_memory_size']
        self.minibatch_size = rl_params['minibatch_size']
        self.epsilon_decay = rl_params['epsilon_decay']
        self.min_epsilon = rl_params['min_epsilon']
        self.action_space = rl_params['action_space_size']
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.epsilon = rl_params['epsilon']
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.update_target_every = rl_params['update_target_every'] # every 5 episodes

    def create_model(self):

        input_img = Input(shape=image_size_net)
        x = Sequential()(input_img)
        x = Conv2D(32, 8, activation='relu', strides=(4,4), data_format='channels_first')(x)
        x = Conv2D(64, 4, activation='relu', strides=(2,2), data_format='channels_first')(x)
        x = Conv2D(64, 3, activation='relu', strides=(1,1), data_format='channels_first')(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x) #
        x = Dense(self.action_space, activation="relu")(x) #

        model = Model(inputs=input_img, outputs=x)
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.0001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # current_state, action, reward, new_state, done
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def act(self, current_state):
        if np.random.random() > self.epsilon and len(current_state) > 3:
            action = np.argmax(self.get_qs(current_state))
        else:
            action = np.random.randint(0, self.action_space)

        # Decay Epsilon
        if len(self.replay_memory) >= self.min_replay_memory_size:
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)
        return action

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        current_states = np.array([transition[0] for transition in minibatch])  # Add divide by max (scale results)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])  # Add divide by max (scale results)
        future_qs_list = self.target_model.predict(new_current_states)

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

        if terminal_state:
            self.target_update_counter += 1
        # updating target models
        if self.target_update_counter > self.update_target_every:
            self.target_update_counter = 0
            self.target_model.set_weights(self.model.get_weights())

    def render(self, reward_arr):
        pass

    def save_model(self, model_dir):
        self.model.save(model_dir)

        print('Agent saved as {}'.format(model_dir))

    def load_model(self, model_dir):
        self.model = keras.models.load_model(model_dir)
        print('Agent {} has loaded'.format(model_dir))
