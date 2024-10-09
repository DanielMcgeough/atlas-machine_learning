#!/usr/bin/env python3
"""shorter than I remember"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Creates a learning rate decay operation in TensorFlow using inverse time decay.

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine
        the rate at which alpha will decay.
        decay_step: The number of passes of
        gradient descent that should occur
        before alpha is decayed further.

    Returns:
        The learning rate decay operation.
    """

    # Create a global step variable to track the number of training steps
    global_step = tf.Variable(0, trainable=False)

    # Create the learning rate decay schedule
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    # Create the learning rate decay operation
    learning_rate_decay_op = tf.assign_add(global_step, 1)
    learning_rate_op = tf.assign(alpha, learning_rate(global_step))

    return learning_rate_op
