from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Multiply, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from SpectralNorm import *
import sys
import numpy as np


def attention_block(input_tensor, input_channels=None, encoder_depth=1):

    if input_channels is None:
        input_channels = input_tensor.get_shape()[-1].value

    # Trunc Branch
    output_trunk = input_tensor

    # Soft Mask Branch
    output_soft_mask = MaxPooling2D(padding='same')(input_tensor)
    skip_connections = []
    for i in range(encoder_depth - 1):
        skip_connections.append(output_soft_mask)
        output_soft_mask = MaxPooling2D(padding='same')(output_soft_mask)

    skip_connections = list(reversed(skip_connections))

    for i in range(encoder_depth - 1):
        output_soft_mask = UpSampling2D()(output_soft_mask)
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    output_soft_mask = UpSampling2D()(output_soft_mask)

    # Output
    # Attention: (1 + output_soft_mask) * output_trunk
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])

    return output


def decode(layer_input, filters, f_size=4):
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
    u = BatchNormalization()(u)
    u = Activation('relu')(u)

    att = attention_block(u)
    u = Concatenate()([u, att])
    return u


def encode(layer_input, filters, f_size=4):
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    att = attention_block(d)
    d = Concatenate()([d, att])

    return d


def res_decode(layer_input, skip_input, filters, f_size=4):
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
    u = BatchNormalization()(u)
    u = Activation('relu')(u)

    u = Concatenate()([u, skip_input])

    att = attention_block(u)
    u = Concatenate()([u, att])
    return u


def res_encode(layer_input, skip_input, filters, f_size=4):

    u = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    u = BatchNormalization()(u)
    u = LeakyReLU(alpha=0.2)(u)

    u = Concatenate()([u, skip_input])

    att = attention_block(u)
    u = Concatenate()([u, att])
    return u


def make_image(layer_input, f_size=4):
    d = Conv2D(3, kernel_size=f_size, strides=1, padding='same')(layer_input)
    d = BatchNormalization()(d)
    d = Activation('tanh')(d)

    return d


class RESAT_GAN():

    def __init__(self):

        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.gf = 64
        self.df = 64
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        d0 = Input(shape=self.img_shape)
        #z = Input(shape=(self.latent_dim,))

        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)

        u1 = deconv2d(d6, d5, self.gf * 8)
        u2 = deconv2d(u1, d4, self.gf * 8)
        u3 = deconv2d(u2, d3, self.gf * 8)
        u4 = deconv2d(u3, d2, self.gf * 4)
        u5 = deconv2d(u4, d1, self.gf * 2)
        u6 = deconv2d(u5, d0, self.gf)

        output_img = Conv2D(self.channels, kernel_size=4, strides=1,
                            padding='same', activation='tanh')(u6)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4):

            d = ConvSN2D(filters, kernel_size=f_size, strides=2, kernel_initializer='glorot_uniform', padding='same')(layer_input)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)

            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)
        d5 = d_layer(d4, self.df * 16)

        validity = ConvSN2D(1, kernel_size=4, strides=1, kernel_initializer='glorot_uniform', padding='same',
                            activation='sigmoid')(d5)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        (X_train, _), (_, _) = mnist.load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = np.random.normal(0, 1, (batch_size, self.img_shape))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = RESAT_GAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
