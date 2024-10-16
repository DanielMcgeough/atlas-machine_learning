#!/usr/bin/env python3
"""even in a job I wont stop this i bet."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a neural network layer with Dropout.

    Args:
    prev: A tensor containing the output of the previous layer.
    n: The number of nodes in the new layer.
    activation: The activation function to use.
    keep_prob: The probability that a node will be kept.
    training: A boolean indicating whether the model is in training mode.

    Returns:
    The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode="fan_avg"
                                                        )
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    output = dense_layer(prev)
    output = dropout_layer(output, training=training)
    
    return output
