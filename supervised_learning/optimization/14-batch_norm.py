#!/usr/bin/env python3
"""shorter than I remember"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow.   


    Args:
        prev: The activated output of the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function that should be used on the output of the layer.

    Returns:
        A tensor of the activated output for the layer.
    """

    # Create the dense layer with batch normalization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),

        use_bias=False
    )

    # Apply batch normalization
    bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-7, trainable=True)

    # Create trainable parameters for gamma and beta
    gamma = tf.Variable(tf.ones(shape=(n,)), name='gamma')
    beta = tf.Variable(tf.zeros(shape=(n,)), name='beta')

    # Apply the layer and batch normalization
    output = layer(prev)
    output = bn(output)
    output = gamma * output + beta

    return output
