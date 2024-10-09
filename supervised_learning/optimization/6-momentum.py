#!/usr/bin/env python3
"""Getting ugly wid it"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up the gradient descent with
    momentum optimization algorithm in
    TensorFlow.

    Args:
        alpha: The learning rate.
        beta1: The momentum weight.

    Returns:
        The optimizer.
    """

    # Create the optimizer with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
