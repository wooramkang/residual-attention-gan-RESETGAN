from __future__ import print_function, division

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

    return u


def encode(layer_input, filters, f_size=4):

    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    return d


def res_decode(layer_input, skip_input, filters, f_size=4):

    u0 = UpSampling2D(size=2)(layer_input)
    att = attention_block(skip_input)

    u = Concatenate()([u0, att])

    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
    u = BatchNormalization()(u)
    u = Activation('relu')(u)

    return u


def res_encode(layer_input, skip_input, filters, f_size=4):

    s0 = UpSampling2D(size=2)(skip_input)
    att = attention_block(s0)

    u = Concatenate()([layer_input, att])

    u = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(u)
    u = BatchNormalization()(u)
    u = LeakyReLU(alpha=0.2)(u)

    return u


def make_image(layer_input, f_size=4):
    img = Conv2D(3, kernel_size=f_size, strides=1, padding='same')(layer_input)
    img = BatchNormalization()(img)
    img = Activation('tanh')(img)

    return img


def dis_layer(layer_input, filters, f_size=4):
    d = ConvSN2D(filters, kernel_size=f_size, strides=2, kernel_initializer='glorot_uniform', padding='same')(
        layer_input)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    return d


class sentemb():

    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator(64)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        noise = Input(shape=(self.latent_dim,))
        sentence_emb = Input(shape=(512,))
        self.generator = self.build_generator(noise, sentence_emb, 64)

        img = self.generator([noise, sentence_emb])

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(noise, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.combined.summary()

    def build_generator(self, noise, sentence_emb, channel):

        noise0 = Dense(1024)(noise)
        sentence_emb0 = Dense(1024)(sentence_emb)

        in_vector = Concatenate()([noise0, sentence_emb0])
        in_vector = Activation('relu')(in_vector)

        z0 = Dense(channel * 32 * 8 * 8, activation="relu")(in_vector)
        z1 = Reshape((8, 8, channel * 32))(z0)
        z2 = UpSampling2D(size=2)(z1)

        res_de0 = res_decode(z1, z2, channel * 16)
        de0 = decode(res_de0, channel * 8)
        res_en0 = res_encode(de0, res_de0, channel * 16)

        res_de1 = res_decode(res_en0, de0, channel * 8)
        de1 = decode(res_de1, channel * 4)
        res_en1 = res_encode(de1, res_de1, channel * 8)

        res_de2 = res_decode(res_en1, de1, channel * 4)
        de2 = decode(res_de2, channel * 2)
        res_en2 = res_encode(de2, res_de2, channel * 4)

        '''
        res_de3 = res_decode(res_en2, de2, channel * 2)
        de3 = decode(res_de3, channel * 1)
        res_en3 = res_encode(de3, res_de3, channel * 2)
        '''

        de7 = decode(res_en2, channel * 1)
        output = make_image(de7)

        out_model = Model([noise, sentence_emb], output)
        out_model.summary()

        return out_model

    def build_discriminator(self, channel):

        img = Input(shape=self.img_shape)
        d0 = dis_layer(img, channel * 2)
        d1 = dis_layer(d0, channel * 4)
        d2 = dis_layer(d1, channel * 8)
        d3 = dis_layer(d2, channel * 16)
        d4 = dis_layer(d3, channel * 32)
        d5 = dis_layer(d4, channel * 32)
        d6 = dis_layer(d5, channel * 64)

        f = Flatten()(d6)
        f0 = Dense(1024)(f)
        f1 = LeakyReLU(alpha=0.2)(f0)

        text = Input(shape=(512,))
        text0 = Dense(1024)(text)
        text1 = LeakyReLU(alpha=0.2)(text0)

        output = Concatenate()([f1, text1])
        output = Dense(1, activation='sigmoid')(output)

        out_model = Model(img, output)
        out_model.summary()

        return out_model

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset

        # (X_train, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("result/without_sentemb_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    without = sentemb()
    #dcgan.train(epochs=4000, batch_size=32, save_interval=50)
