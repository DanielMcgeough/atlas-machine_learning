#!/usr/bin/env python3
"""Sometimes I hate this"""
import tensorflow.compat.v1 as tf

def lenet5(x, y):
    """
    Builds a modified LeNet-5 architecture.

    Args:
        x: Input placeholder of shape (m, 28, 28, 1).
        y: One-hot encoded labels of shape (m, 10).

    Returns:
        A tuple containing:
            - The softmax activated output tensor.
            - The training operation.
            - The loss tensor.
            - The accuracy tensor.
    """

    # Convolutional Layer 1
    W1 = tf.get_variable("W1", shape=[5, 5, 1, 6], initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
    b1 = tf.Variable(tf.zeros([1, 1, 1, 6]))
    Z1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    A1 = tf.nn.relu(Z1)

    # Pooling Layer 1
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer 2
    W2 = tf.get_variable("W2", shape=[5, 5, 6, 16], initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
    b2 = tf.Variable(tf.zeros([1, 1, 1, 16]))
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
    A2 = tf.nn.relu(Z2)

    # Pooling Layer 2
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    P2 = tf.contrib.layers.flatten(P2)

    # Fully Connected Layer 1
    W3 = tf.get_variable("W3", shape=[400, 120], initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
    b3 = tf.Variable(tf.zeros([120]))
    Z3 = tf.matmul(P2, W3) + b3
    A3 = tf.nn.relu(Z3)

    # Fully Connected Layer 2
    W4 = tf.get_variable("W4", shape=[120, 84], initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
    b4 = tf.Variable(tf.zeros([84]))
    Z4 = tf.matmul(A3, W4) + b4
    A4 = tf.nn.relu(Z4)

    # Output Layer
    W5 = tf.get_variable("W5", shape=[84, 10], initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
    b5 = tf.Variable(tf.zeros([10]))
    Z5 = tf.matmul(A4, W5) + b5

    # Softmax Activation
    logits = Z5
    predictions = tf.nn.softmax(logits)

    # Loss and Optimization
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)


    # Accuracy
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    return predictions, train_op, loss, accuracy
