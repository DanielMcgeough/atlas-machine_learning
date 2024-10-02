#!/usr/bin/env python3
"""for tensorflow without Keras... explicitly"""
import tensorflow.compat.v1 as tf

def evaluate(X, Y, save_path):
  """Evaluates the output of a neural network.

  Args:
    X: A numpy.ndarray containing the input data to evaluate.
    Y: A numpy.ndarray containing the one-hot labels for X.
    save_path: The location to load the model from.

  Returns:
    The network's prediction, accuracy, and loss, respectively.
  """

  # Load the saved model
  saver = tf.train.import_meta_graph(save_path + '.meta')

  with tf.Session() as sess:
    # Restore the saved variables
    saver.restore(sess, save_path)

    # Get the placeholders, tensors, and operations from the graph's collection
    x = tf.get_collection('placeholders')[0]
    y = tf.get_collection('placeholders')[1]
    y_pred = tf.get_collection('tensors')[0]
    loss = tf.get_collection('tensors')[1]
    accuracy = tf.get_collection('tensors')[2]

    # Evaluate the model
    prediction, loss_value, accuracy_value = sess.run([y_pred, loss, accuracy], feed_dict={x: X, y: Y})

  return prediction, accuracy_value, loss_value
