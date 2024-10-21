#!/usr/bin/env python3
"""Setting up an environment with EE"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """build a neural network with Keras"""

    model = K.Sequential()

    for i, layer_size in enumerate(layers):
        if i == 0:
            model.add(K.layers.Dense(
                layer_size,
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(
                layer_size,
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
