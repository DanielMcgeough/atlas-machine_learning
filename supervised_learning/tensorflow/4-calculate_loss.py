#!/usr/bin/env python3
"""for tensorflow without Keras... explicitly"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the softmax
    cross-entropy loss of a prediction"""
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    average_loss = tf.reduce_mean(loss)

    return average_loss
