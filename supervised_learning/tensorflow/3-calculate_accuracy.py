#!/usr/bin/env python3
"""for tensorflow without Keras... explicitly"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """This will calculate the accuracy
    of a prediction"""
    predicted_indices = tf.argmax(y_pred, axis=1)
    correct_predictions = tf.equal(predicted_indices, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
