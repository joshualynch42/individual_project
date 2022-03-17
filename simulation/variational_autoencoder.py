import numpy as np
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, MaxPooling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import random, os, os.path
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def get_image(path, file):
    img = load_img(path + file)
    img = img_to_array(img)
    return img

img_array = []
label_array = []
counter = -1

path = "D:/Josh/github/individual_project/simulation/simulation_data/key_images/"
for key in os.listdir(path):
    new_path = path + key + "/"
    counter += 1
    for file in os.listdir(new_path):
        img = get_image(new_path, file)
        img_array.append(img)
        label_array.append(counter)

img_array = np.array(img_array)
train, y_train  = img_array, label_array

train = train.astype('float32')
# test = test.astype('float32')
train = train / 255 # normalise the images
# test = test / 255

img_width = train.shape[1]
img_height = train.shape[2]
print(train.shape)
num_channels = 3 #rgb
train = train.reshape(train.shape[0], img_height, img_width, num_channels)
# test = test.reshape(test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)

# #view of key_images
# plt.figure(1)
# plt.subplot(221)
# plt.imshow(train[4][:,:,0])
#
# plt.subplot(222)
# plt.imshow(train[8][:,:,0])
#
# plt.subplot(223)
# plt.imshow(train[12][:,:,0])
#
# plt.subplot(224)
# plt.imshow(train[16][:,:,0])
# plt.show()

latent_dim = 4

input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(64, 5, padding = 'same', activation='relu')(input_img)
x = Conv2D(128, 3, padding = 'same', activation='relu', strides=(2,2))(x)
x = Conv2D(128, 3, padding = 'same', activation='relu')(x)
x = Conv2D(128, 3, padding = 'same', activation='relu')(x)

conv_shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mu = Dense(latent_dim, name='latent_mu')(x)
z_sigma = Dense(latent_dim, name='laent_sigma')(x)

def sample_z(args):
    z_mu, z_sigma = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
    return z_mu + K.exp(z_sigma / 2) * eps

z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma])

encoder_output = [z_mu, z_sigma, z]

encoder = Model(input_img, encoder_output, name='encoder')
print(encoder.summary())

## Decoder ##
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2))(x)
x = Conv2DTranspose(num_channels, 5, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

z_decoded = decoder(z)

# defining custom loss
# VAE trained using 2 loss functions
# recontruction loss and kl divergence

class CustomLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# applying the custom loss to the input and decoded images
y = CustomLayer()([input_img, z_decoded])
# y is the input images after going through encoding
# this is the output for the vae

vae = Model(input_img, y, name='vae')

# compile
vae.compile(optimizer='adam', loss=None)
vae.summary()

vae.fit(train, None, epochs = 30, batch_size = 32, validation_split = 0.2)

encoder.save('D:/Josh/github/individual_project/simulation/vae_models/first_model.h5')

# mu, _, _ = encoder.predict(test)
# plt.figure(figsize=(10,10))
# plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
# plt.xlabel('dim 1') # 5 total dimensions
# plt.ylabel('dim 2')
# plt.colorbar()
# plt.show()
