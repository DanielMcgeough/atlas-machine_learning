#!/usr/bin/env python3
"""Sometimes I hate this"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using Keras.

    Args:
        X: Input tensor of shape (m, 28, 28, 1).

    Returns:
        A compiled Keras model.
    """
    he_normal = K.initializers.VarianceScaling(scale=2.0, seed=0)

    conv1 = K.layers.Conv2D(6, (5, 5), padding='same', activation='relu',
                            kernel_initializer=he_normal)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(16, (5, 5), padding='valid', activation='relu',
                            kernel_initializer=he_normal)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flat = K.layers.Flatten()(pool2)

    fcl1 = K.layers.Dense(120, activation='relu',
                          kernel_initializer=he_normal)(flat)

    fcl2 = K.layers.Dense(84, activation='relu',
                          kernel_initializer=he_normal)(fcl1)

    output = K.layers.Dense(10, activation='softmax',
                            kernel_initializer=he_normal)(fcl2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
