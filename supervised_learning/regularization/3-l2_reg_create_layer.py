#!/usr/bin/env python3
"""even in a job I wont stop this i bet."""
import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer with L2 regularization.

    Args:
        prev: A tensor containing the output of the previous layer.
        n: The number of nodes in the new layer.
        activation: The activation function to use.
        lambtha: The L2 regularization parameter.

    Returns:
        The output of the new layer.
    """


    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.l2(lambtha)
    output = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=init)(prev)

    return output
