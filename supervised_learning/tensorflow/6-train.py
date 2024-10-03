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
  Builds, trains,
  and saves a neural network classifier.
  """

  """Create placeholders for input data and labels"""
  x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

  """Forward propagation"""
  y_pred = forward_prop(x, layer_sizes, activations)

  """Calculate loss and accuracy"""
  loss = calculate_loss(y, y_pred)
  accuracy = calculate_accuracy(y, y_pred)

  """Create training operation"""
  train_op = create_train_op(loss, alpha)

  """Add placeholders, tensors, and operation
  to the graph's collection"""
  tf.add_to_collection('x', x)
  tf.add_to_collection('y', y)
  tf.add_to_collection('y_pred', y_pred)
  tf.add_to_collection('loss', loss)
  tf.add_to_collection('accuracy', accuracy)
  tf.add_to_collection('train_op', train_op)

  """Create a saver"""
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as sess
    """initialize variables"""
    sess.run(init)

    """Train the model with a for loop"""
    for i in range(iterations + 1):
        # Calculate training cost and accuracy
        t_cost, t_accuracy = sess.run([loss, accuracy],
        feed_dict={x: X_train, y: y_train})

        # Calculate validation cost and accuracy
        v_cost, v_accuracy = sess.run([loss, accuracy],
        feed_dict={x: X_valid, y: y_valid})

        # Print progress at specified intervals
        if i % 100 == 0 or i == iterations:
            print(f"After {i} iterations:")
            print(f"\tTraining Cost: {t_cost}")
            print(f"\tTraining Accuracy: {t_accuracy}")
            print(f"\tValidation Cost: {v_cost}")
            print(f"\tValidation Accuracy: {v_accuracy}")

            # Perform training step if not at the last iteration
        if i < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: y_train})

        # Save the model and return the save path
        return saver.save(sess, save_path)
  return save_path
