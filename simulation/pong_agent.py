import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input
from keras.optimizers import adam_v2
from collections import deque
import random, os
import tensorflow as tf
import cv2

def process_image(state):
    state = cv2.resize(state, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    return state / 255

class PongAgent():
    def __init__(self, rl_params):
        self.label = 'Pong'
        self.replay_memory_size = rl_params['replay_memory_size']
        self.discount = rl_params['discount']
        self.min_replay_memory_size = rl_params['min_replay_memory_size']
        self.minibatch_size = rl_params['minibatch_size']
        self.epsilon_decay = rl_params['epsilon_decay']
        self.min_epsilon = rl_params['min_epsilon']
        self.action_space = 3 # up, down, nothing
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.epsilon = rl_params['epsilon']
        self.model = self.create_model()

    def create_model(self):

        input_img = Input(shape=(64, 64, 1))
        x = Sequential()(input_img)
        x = Conv2D(32, 8, activation='relu', strides=(4,4))(x)
        x = Conv2D(64, 4, activation='relu', strides=(2,2))(x)
        x = Conv2D(64, 3, activation='relu', strides=(1,1))(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x) #64
        x = Dense(self.action_space, activation="linear")(x)

        model = Model(inputs=input_img, outputs=x)
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.0001), metrics=['accuracy'])

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
