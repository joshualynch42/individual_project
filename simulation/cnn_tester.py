import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input, Activation
from keras.optimizers import adam_v2
import random, os
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections

alphabet_arr = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
        'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE'
        , 'UP', 'DOWN', 'LEFT', 'RIGHT']

images = []
letters = []
counter = 0
for letter in alphabet_arr:
    dir = "D:/Josh/github/individual_project/simulation/simulation_data/key_images/{}/".format(letter)
    for file_name in os.listdir(dir):
        img = load_img(dir+file_name, color_mode='grayscale', target_size=(128,128))
        img = img_to_array(img)
        images.append(img)
        letters.append(counter)
    counter += 1

X_train, X_test, y_train, y_test = train_test_split(np.array(images), np.array(letters), test_size=0.2)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

plt.imshow(X_train[0])
plt.show()


# input_img = Input(shape=np.shape(X_train[0]))
# x = Sequential()(input_img)
# x = Conv2D(8, (3,3), activation='relu', strides=(1,1))(x)
# x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
# x = Conv2D(16, (3,3), activation='relu', strides=(1,1))(x)
# x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
# x = Conv2D(32, (3,3), activation='relu', strides=(1,1))(x)
# x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
# x = Conv2D(32, (3,3), activation='relu', strides=(1,1))(x)
# x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
#
# x = Flatten()(x)
#
# x = Dense(512, activation="relu")(x)
# x = Dense(1, activation="linear")(x)
# model = Model(inputs=input_img, outputs=x)

model = Sequential()
model.add(Conv2D(64, 5, padding='same', activation='relu', input_shape=np.shape(X_train[0])))

model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))

# model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(512, (5,5), padding='same', activation='relu'))
# model.add(Conv2D(512, (5,5), activation='relu'))
# model.add(MaxPooling2D((4,4)))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))

model.add(Dense(len(alphabet_arr), activation="softmax"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam_v2.Adam(learning_rate=0.001), metrics=["accuracy"])
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), verbose=1, batch_size=16)

# y_predict = model.predict(X_test)
# y_predict = np.rint(y_predict)
#
# print(y_predict)
# print(y_test)

# bool_arr = list((y_test == y_predict)[0])
#
# count = bool_arr.count(False)
#
# print('There are {} false values and {} true values'.format(count, (len(bool_arr)-count)))
