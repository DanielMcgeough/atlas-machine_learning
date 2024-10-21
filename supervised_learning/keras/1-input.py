#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Building a keras model with the aid
    of the sequential class, meaning I need to build
    it from scratch with the Functional API.
    this is useful if the model is more complex
    or if the data doesn't flow in a sequential
    manner"""

    inputs = K.Input(shape=(nx,))

    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=tf.keras.regularizers.l2(lambtha))(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
