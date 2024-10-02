#!/usr/bin/env python3
"""for tensorflow without Keras... explicitly"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
  """
  Builds, trains, and saves a neural network classifier.

  Args:
    X_train: A numpy.ndarray containing the training input data.
    Y_train: A numpy.ndarray containing the training labels.
    X_valid: A numpy.ndarray containing the validation input data.
    Y_valid: A numpy.ndarray containing the validation labels.
    layer_sizes: A list containing the number of nodes in each layer of the network.
    activations: A list containing the activation functions for each layer of the network.
    alpha: The learning rate.
    iterations: The number of iterations to train over.
    save_path: The path where the model should be saved.

  Returns:
    The path where the model was saved.
  """

  # Create placeholders for input data and labels
  x, y = create_placeholders(layer_sizes[0], layer_sizes[-1])

  # Forward propagation
  y_pred = forward_prop(x, layer_sizes, activations)

  # Calculate loss and accuracy
  loss = calculate_loss(y, y_pred)
  accuracy = calculate_accuracy(y, y_pred)

  # Create training operation
  train_op = create_train_op(loss, alpha)

  # Add placeholders, tensors, and operation to the graph's collection
  tf.add_to_collection('placeholders', x)
  tf.add_to_collection('placeholders', y)
  tf.add_to_collection('tensors', y_pred)
  tf.add_to_collection('tensors', loss)
  tf.add_to_collection('tensors', accuracy)
  tf.add_to_collection('operations', train_op)

  # Create a saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Train the model
    for i in range(iterations):
      # Train the model
      _, train_cost, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})

      # ... (rest of your training loop here)

  # Save the model
  saver.save(sess, save_path)

  return save_path
