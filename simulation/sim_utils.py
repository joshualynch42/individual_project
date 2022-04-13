from gym import Env, spaces, make
import random, os
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from double_dqn import *
from dqn import *
from dueling_ddqn import *
from dueling_ddqn_per import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #surpressing no gpu tensorflow warning

key_coords = pd.read_csv(r"D:\Josh\github\individual_project\simulation\simulation_data\key_coords.csv")
high_one_hot = [1] + [0]*(len(key_coords)-1)
low_one_hot = [0]*(len(key_coords)-1) + [1]
alphabet_arr = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
        'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE']
arrow_arr = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# key_image_loc = 'key_images'
# key_image_loc = 'alex_key_images'


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

def get_image(key, key_image_loc):
    dir = "D:/Josh/github/individual_project/simulation/simulation_data/{}/{}/".format(key_image_loc, key)
    file_name = random.choice(os.listdir(dir))
    img = load_img(dir+file_name, color_mode='grayscale', target_size=(64,64))
    img = img_to_array(img).astype('float32') / 255
    return img

def create_state(image, goal_letter):
    one_hot_arr = [0]*(len(key_coords))
    num = key_coords.index[key_coords['Key'] == goal_letter].tolist()[0]
    one_hot_arr[num] = 1
    state = {'img': image, 'vec': np.array(one_hot_arr)}
    return state  # current image and goal letter

def create_one_hot(letter):
    one_hot_arr = [0]*(len(key_coords))
    num = key_coords.index[key_coords['Key'] == letter].tolist()[0]
    one_hot_arr[num] = 1
    return np.array(one_hot_arr)

#coords = [x, y] || [0,0] top left

class discrete_alphabet_env(Env):
    def __init__(self, key_image_loc):
        self.key_image_loc = key_image_loc
        self.current_step = 0
        self.starting_letter = random.choice(alphabet_arr)
        self.goal_letter = random.choice(alphabet_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        self.action_array = np.array(['upleft', 'upright', 'left', 'right', 'downleft', 'downright', 'pressdown'])
        self.state = create_state(get_image(self.starting_letter, self.key_image_loc), self.goal_letter)
        self.img_shape = np.shape(get_image(self.starting_letter, self.key_image_loc))
        self.max_ep_len = 40
        self.action_space_size = len(self.action_array)

    def reset(self, key_image_loc):
        self.key_image_loc = key_image_loc
        self.current_step = 0
        self.starting_letter = random.choice(alphabet_arr)
        self.goal_letter = random.choice(alphabet_arr)
        self.state = create_state(get_image(self.starting_letter, self.key_image_loc), self.goal_letter)
        self.current_coords = letter_to_coords(self.starting_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary

        if action == 'upleft':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 11, -37])
            elif self.current_coords[0] > 0 and self.current_coords[1] > 0:
                self.current_coords = self.current_coords + np.array([-1, -1, 0])

        elif action == 'upright':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 14, -37])
            elif self.current_coords[0] > 0:
                self.current_coords = self.current_coords + np.array([-1, 2, 0])

        elif action == 'left':
            if self.current_coords[0] < 3 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([0, -3, 0])

        elif action == 'right':
            if self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([0, 3, 0])
            elif self.current_coords[0] < 3 and self.current_coords[1] < 18:
                self.current_coords = self.current_coords + np.array([0, 3, 0])

        elif action == 'downleft':
            if self.current_coords[0] == 1 and self.current_coords[1] > 24:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([1, -2, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 5: # bottom row not z or x
                self.current_coords = np.array([3, 14, -37])

        elif action == 'downright':
            if self.current_coords[0] == 1 and self.current_coords[1] > 19:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([1, 1, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 2: # bottom row not z
                self.current_coords = np.array([3, 14, -37])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        current_img = get_image(coords_to_letter(self.current_coords), self.key_image_loc) #get new image of state
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
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
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
    def __init__(self, key_image_loc):
        self.key_image_loc = key_image_loc
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.goal_letter = random.choice(arrow_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        self.action_array = np.array(['up', 'left', 'right', 'down', 'pressdown'])
        self.state = create_state(get_image(self.starting_letter, self.key_image_loc), self.goal_letter)
        self.img_shape = np.shape(get_image(self.starting_letter, self.key_image_loc))
        self.max_ep_len = 25
        self.action_space_size = len(self.action_array)

    def reset(self, key_image_loc):
        self.key_image_loc = key_image_loc
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.goal_letter = random.choice(arrow_arr)
        self.state = create_state(get_image(self.starting_letter, self.key_image_loc), self.goal_letter)
        self.current_coords = letter_to_coords(self.starting_letter)

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
        current_img = get_image(coords_to_letter(self.current_coords), self.key_image_loc) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

    def step(self, action, steps):
        done = False
        info = []
        if steps > self.max_ep_len -1:
            reward = 0
        else:
            action = self.action_array[action]
            if action == 'pressdown':
                done = True
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass

class her():
    def __init__(self):
        self.her_buffer = []

    def update_her_buffer(self, transition):
        self.her_buffer.append(transition)

    def sample(self, index):
        return self.her_buffer[index]

    def update_transition(self, index, new_goal, max_steps):
        current_state, action, reward, new_state, done = self.sample(index)
        temp_current_state = current_state.copy()
        temp_new_state = new_state.copy()
        temp_current_state['vec'] = new_goal
        temp_new_state['vec'] = new_goal

        if index == max_steps-1:
            new_reward = 1
        else:
            new_reward = 0

        return (temp_current_state, action, new_reward, temp_new_state, done)
