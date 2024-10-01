#!/usr/bin/env python3
"""for tensorflow without Keras... explicitly"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """function to create layers in neural network"""
    v = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=v,
        name='layer'
    )
    return layer(prev)
