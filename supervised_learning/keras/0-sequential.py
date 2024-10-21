#!/usr/bin/env python3
"""Setting up an environment with EE"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """build a neural network with Keras"""

    model = K.Sequential()

    for i in range(1, len(layers)):
        model.add(K.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=tf.keras.regularizers.l2(lambtha)))
        model.add(K.layers.Dropout(rate=1 - keep_prob))
    
    return model
