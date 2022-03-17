# STATE IS IMAGE NOT COORDINATES
import pandas as pd
import numpy as np
from gym import Env, spaces, make
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input
from keras.optimizers import adam_v2
from collections import deque
import random, os
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #surpressing no gpu tensorflow warning

key_coords = pd.read_csv(r"D:\Josh\github\individual_project\simulation\simulation_data\key_coords.csv")
high_one_hot = [1] + [0]*(len(key_coords)-1)
low_one_hot = [0]*(len(key_coords)-1) + [1]
alphabet_arr = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
        'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE']
arrow_arr = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def translate_coord(coords):
    """
    Input: A 1x3 coordinate vector in the form [x, y, z]

    This function converts from imaginary to real coordinates and vice versa

    Returns: A translated 1x3 coordinate vector

    """
    x, y, z = coords[0], coords[1], coords[2]
    if x > 3:
        row = key_coords.loc[key_coords['X'] == x]
        row = row.loc[row['Y'] == y]
        if len(row) > 0:
            new_x = row['IM_X'].to_numpy()[0]
            new_y = row['IM_Y'].to_numpy()[0]
        else:
            print('Error: no matching coordinates found')
            exit()
    else:
        row = key_coords.loc[key_coords['IM_X'] == x]
        row = row.loc[row['IM_Y'] == y]
        if len(row) > 0:
            new_x = row['X'].to_numpy()[0]
            new_y = row['Y'].to_numpy()[0]
        else:
            print('Error: no matching coordinates found')
            exit()
    return [new_x, new_y, z]

def coords_to_letter(coords):
    x, y, z = coords[0], coords[1], coords[2]
    row = key_coords.loc[key_coords['X'] == x]
    row = row.loc[row['Y'] == y]
    if len(row) > 0:
        letter = row['Key'].to_numpy()[0]
    else:
        print('Error: no matching coordinates found')
        exit()

    return letter

def letter_to_coords(letter):
    row = key_coords.loc[key_coords['Key'] == letter]
    if len(row) > 0:
        x = row['X'].to_numpy()[0]
        y = row['Y'].to_numpy()[0]
        z = row['Z'].to_numpy()[0]
    else:
        print('Error: no matching coordinates found')
        exit()

    return [x, y, z]

def get_image(key):
    dir = "D:/Josh/github/individual_project/simulation/simulation_data/key_images/{}/".format(key)
    file_name = random.choice(os.listdir(dir))
    img = load_img(dir+file_name, color_mode='grayscale', target_size=(64,64))
    img = img_to_array(img)
    return img

def create_state(image, goal_letter):
    one_hot_arr = [0]*(len(key_coords))
    num = key_coords.index[key_coords['Key'] == goal_letter].tolist()[0]
    one_hot_arr[num] = 1
    state = {'img': image, 'vec': np.array(one_hot_arr)}
    return state  # current image and goal letter

#coords = [x, y] || [0,0] top left

class discrete_alphabet_env(Env):
    def __init__(self):
            self.current_step = 0
            self.starting_letter = random.choice(alphabet_arr)
            self.goal_letter = random.choice(alphabet_arr)
            self.current_coords = letter_to_coords(self.starting_letter)
            self.goal_coords = letter_to_coords(self.goal_letter)
            self.action_space_size = 7
            self.action_space = spaces.Discrete(self.action_space_size)
            self.action_array = np.array(['upleft', 'upright', 'left', 'right', 'downleft', 'downright', 'pressdown'])
            self.observation_space = spaces.Dict({'img' : spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
                                        'vec' : spaces.Box(low=np.array(low_one_hot), high=np.array(high_one_hot), dtype=np.uint8)})
            self.state = create_state(get_image(self.starting_letter), self.goal_letter)
            self.img_shape = np.shape(get_image(self.starting_letter))
            self.max_ep_len = 25

    def reset(self):
        self.current_step = 0
        self.starting_letter = random.choice(alphabet_arr)
        self.goal_letter = random.choice(alphabet_arr)
        self.state = create_state(get_image(self.starting_letter), self.goal_letter)
        self.current_coords = letter_to_coords(self.starting_letter)
        self.goal_coords = letter_to_coords(self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary

        if action == 'upleft':
            if self.current_coords[0] > 0 and self.current_coords[1] > 0:
                self.current_coords = self.current_coords + np.array([-1, -1, 0])

        elif action == 'upright':
            if self.current_coords[0] > 0:
                self.current_coords = self.current_coords + np.array([-1, 2, 0])

        elif action == 'left':
            if self.current_coords[1] > 2 and self.current_coords[0] < 3:
                self.current_coords = self.current_coords + np.array([0, -3, 0])

        elif action == 'right':
            if self.current_coords[1] < 25 and self.current_coords[0] < 2:
                self.current_coords = self.current_coords + np.array([0, 3, 0])
            elif self.current_coords[1] < 18 and self.current_coords[0] < 3:
                self.current_coords = self.current_coords + np.array([0, 3, 0])

        elif action == 'downleft':
            if self.current_coords[0] == 1 and self.current_coords[1] > 24:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([1, -2, 0])

        elif action == 'downright':
            if self.current_coords[1] > 19 and self.current_coords[0] == 1:
                pass
            elif self.current_coords[1] < 25 and self.current_coords[0] < 2:
                self.current_coords = self.current_coords + np.array([1, 1, 0])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        current_img = get_image(coords_to_letter(self.current_coords)) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

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

class discrete_arrow_env(Env):
    def __init__(self):
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.goal_letter = random.choice(arrow_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        self.goal_coords = letter_to_coords(self.goal_letter)
        self.action_space_size = 5
        self.action_space = spaces.Discrete(self.action_space_size)
        self.action_array = np.array(['up', 'left', 'right', 'down', 'pressdown'])
        self.observation_space = spaces.Dict({'img' : spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
                                    'vec' : spaces.Box(low=np.array(low_one_hot), high=np.array(high_one_hot), dtype=np.uint8)})
        self.state = create_state(get_image(self.starting_letter), self.goal_letter)
        self.img_shape = np.shape(get_image(self.starting_letter))
        self.max_ep_len = 25

    def reset(self):
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.goal_letter = random.choice(arrow_arr)
        self.state = create_state(get_image(self.starting_letter), self.goal_letter)
        self.current_coords = letter_to_coords(self.starting_letter)
        self.goal_coords = letter_to_coords(self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary

        if action == 'up':  # all values become up key
            if self.current_coords[0] == 3 and self.current_coords[1] == 43:
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
                self.current_coords = np.array([3, 46, -37])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        current_img = get_image(coords_to_letter(self.current_coords)) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

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

class DQNAgent():
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
        self.model = self.create_model(env, env.state['vec'])

    def create_model(self, env, goal):

        input_img = Input(shape=env.img_shape)
        x = Sequential()(input_img)
        x = Conv2D(8, (3,3), activation='relu', strides=(1,1))(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
        x = Conv2D(16, (3,3), activation='relu', strides=(1,1))(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
        x = Conv2D(32, (3,3), activation='relu', strides=(1,1))(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
        x = Conv2D(32, (3,3), activation='relu', strides=(1,1))(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)

        x = Flatten()(x)

        input_vec = Input(shape=(len(goal)))
        y = Dense(31, activation='relu')(input_vec)  # change one-hot to layer
        x = Concatenate(axis=1)([x, y])

        x = Dense(512, activation="relu")(x)
        x = Dense(self.action_space, activation="linear")(x)

        model = Model(inputs=(input_img, input_vec), outputs=x)

        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # current_state, action, reward, new_state, done
        self.replay_memory.append(transition)

    def get_qs(self, state):
        img = state['img']
        vec = state['vec']

        return self.model.predict((np.array(img).reshape(-1, *img.shape),
                                np.array(vec).reshape(-1, *vec.shape)))[0]

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
        img_arr = np.array([x['img'] for x in current_states])
        vec_arr = np.array([x['vec'] for x in current_states])
        current_qs_list = self.model.predict((img_arr, vec_arr))

        new_current_states = np.array([transition[3] for transition in minibatch])  # Add divide by max (scale results)
        fut_img_arr = np.array([x['img'] for x in new_current_states])
        fut_vec_arr = np.array([x['vec'] for x in new_current_states])
        future_qs_list = self.model.predict((fut_img_arr , fut_vec_arr))

        x_img = []  # array of current states
        x_vec = []
        y = []  # array of new q values

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            x_img.append(current_state['img'])
            x_vec.append(current_state['vec'])
            y.append(current_qs)

        self.model.fit((np.array(x_img), np.array(x_vec)), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

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
