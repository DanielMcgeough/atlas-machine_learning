#!/usr/bin/env python3
"""hate this ish."""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception network as described
    in Going Deeper with Convolutions (2014).

    Args:
        None

    Returns:

    The Keras model.
    """
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    conv = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                           padding='same', activation='relu',
                           kernel_initializer=init)(input)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(conv)

    conv2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer=init)(max_pool)

    conv3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer=init)(conv2)

    max_pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(conv3)

    inception = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    inception1 = inception_block(inception, [128, 128, 192, 32, 96, 64])

    max_pool3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(inception1)

    inception2a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    inception2b = inception_block(inception2a, [160, 112, 224, 24, 64, 64])

    inception2c = inception_block(inception2b, [128, 128, 256, 24, 64, 64])

    inception2d = inception_block(inception2c, [112, 144, 288, 32, 64, 64])

    inception2e = inception_block(inception2d, [256, 160, 320, 32, 128, 128])

    max_pool4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(inception2e)

    inception3a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    inception3b = inception_block(inception3a, [384, 192, 384, 48, 128, 128])

    pool_avg = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(inception3b)

    dropout = K.layers.Dropout(rate=0.4)(pool_avg)

    output_layer = K.layers.Dense(units=1000, activation='softmax',
                                  kernel_initializer=init)(dropout)

    model = K.Model(inputs=input, outputs=output_layer)

    return model
