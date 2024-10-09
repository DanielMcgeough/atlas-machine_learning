#!/usr/bin/env python3
"""shorter than I remember"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha: The learning rate.
        beta2: The RMSProp weight (Discounting factor).
        epsilon: A small number to avoid division by zero.

    Returns:
        The optimizer.
    """

    # Create the RMSProp optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)

    return optimizer
