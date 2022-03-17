import pandas as pd
import numpy as np
from gym import Env
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from collections import deque
import random, os
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #surpressing no gpu tensorflow warning

def get_image(key):
    dir = "D:/Josh/github/individual_project/simulation/simulation_data/key_images/{}/".format(key)
    file_name = random.choice(os.listdir(dir))
    img = load_img(dir+file_name)
    img = img_to_array(img)
    return img

model = Sequential()
model.add(Conv2D(256, (3,3), activation="relu", input_shape=[128,129,3]))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(self.action_space, activation="linear"))
model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.001), metrics=['accuracy'])
