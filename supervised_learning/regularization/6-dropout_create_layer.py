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

  layer = tf.keras.layers.Dense(units=n, activation=activation)
  output = tf.keras.layers.Dropout(rate=1 - keep_prob, training=training)(layer(prev))

  return output
