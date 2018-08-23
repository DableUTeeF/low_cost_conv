#
# Low-cost conv
#


import os
import numpy as np
import random

from tqdm import tqdm
from tqdm import trange

from keras.models import Model, Sequential
from keras.layers import (Reshape, Dense, Dropout, Flatten, Conv2D, UpSampling2D,
                          Lambda, Multiply, Add, BatchNormalization, Activation,
                          Input, Concatenate, MaxPooling2D, GlobalAveragePooling2D,
                          Permute)
from keras.engine.topology import Layer
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.utils import np_utils

from keras import backend as K

from keras import initializers, regularizers
from keras.initializers import RandomNormal

#############################################################################################

import tensorflow as tf


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    print(flops.total_float_ops)


# assume 'channels_first' -> (?,c,w,h) -> (?c,w*h) -> (?,m,w*h) -> (?,m,w,h)
class MyLayer(Layer):
    def __init__(self, c, w, h, m, **kwargs):
        self.c, self.w, self.h, self.m = c, w, h, m
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.c, self.m),
                                      initializer='uniform',
                                      trainable='True')
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        b = x.get_shape()
        x1 = Reshape((self.c, self.w * self.h))(x)  # K.reshape( x, (b[0], self.c, self.w*self.h) )
        x2 = K.permute_dimensions(x1, (0, 2, 1))
        x3 = K.dot(x2, self.kernel)
        x4 = K.permute_dimensions(x3, (0, 2, 1))
        x5 = Reshape((self.m, self.w, self.h))(x4)  # K.reshape( x4, (b[0], self.m, self.w, self.h) )
        return x5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.m, self.w, self.h)


def create_mycnn():
    # channels first!
    x = Input(shape=(1, 28, 28))

    y = Conv2D(8, (3, 3), padding='same', data_format='channels_first', use_bias=False)(x)
    y = BatchNormalization(axis=1)(y)
    y = MyLayer(8, 28, 28, 64)(y)
    y = Activation('relu')(y)
    y = MyLayer(64, 28, 28, 8)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = Conv2D(16, (3, 3), padding='same', data_format='channels_first', strides=(2, 2), use_bias=False)(y)
    y = BatchNormalization(axis=1)(y)
    y = MyLayer(16, 14, 14, 128)(y)
    y = Activation('relu')(y)
    y = MyLayer(128, 14, 14, 16)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = Conv2D(32, (3, 3), padding='same', data_format='channels_first', strides=(2, 2), use_bias=False)(y)
    y = BatchNormalization(axis=1)(y)
    y = MyLayer(32, 7, 7, 256)(y)
    y = Activation('relu')(y)
    y = MyLayer(256, 7, 7, 32)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling2D(data_format='channels_first')(y)
    z = Dense(10, activation='softmax')(y)

    m = Model(inputs=[x], outputs=[z])
    # m.summary()

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return m


##############################################################################################################
# The data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# m.fit( X_train, Y_train, batch_size=32, epochs=10, verbose=0 )

# score = m.evaluate( X_test, Y_test, batch_size=32 )
# print(score)

# get_flops(m)


def create_cnn():
    # channels first!
    x = Input(shape=(1, 28, 28))

    y = Conv2D(8, (3, 3), padding='same', data_format='channels_first')(x)
    y = BatchNormalization(axis=1)(y)
    y = Conv2D(64, (3, 3), padding='same', data_format='channels_first', use_bias=False)(y)
    y = Conv2D(8, (1, 1), padding='same', data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = Conv2D(16, (3, 3), padding='same', data_format='channels_first', strides=(2, 2))(y)
    y = BatchNormalization(axis=1)(y)
    y = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=False)(y)
    y = Conv2D(16, (1, 1), padding='same', data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = Conv2D(32, (3, 3), padding='same', data_format='channels_first', strides=(2, 2))(y)
    y = BatchNormalization(axis=1)(y)
    y = Conv2D(256, (3, 3), padding='same', data_format='channels_first', use_bias=False)(y)
    y = Conv2D(32, (1, 1), padding='same', data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling2D(data_format='channels_first')(y)
    z = Dense(10, activation='softmax')(y)

    m = Model(inputs=[x], outputs=[z])
    # m.summary()

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return m


m = create_mycnn()
get_flops(m)

m = create_cnn()
get_flops(m)

for i in range(3):
    m = create_mycnn()
    m.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=0)

    score = m.evaluate(X_test, Y_test, batch_size=32)
    print('mycnn ', score)

    m = create_cnn()
    m.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=0)

    score = m.evaluate(X_test, Y_test, batch_size=32)
    print('cnn ', score)

# m.fit( X_train, Y_train, batch_size=32, epochs=10, verbose=0 )

# score = m.evaluate( X_test, Y_test, batch_size=32 )
# print(score)

# get_flops(m)
