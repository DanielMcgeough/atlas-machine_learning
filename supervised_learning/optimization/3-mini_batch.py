#!/usr/bin/env python3
"""This is needed to pass checker"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X: The first numpy.ndarray of shape (m, nx) to shuffle.
        m: The number of data points.
        nx: The number of features in X.
        Y: The second numpy.ndarray of shape
        (m, ny) to shuffle.
        m: The same number of data points as in X.
        ny: The number of features in Y.

    Returns:
        The shuffled X and Y matrices.
  """

    # Create a permutation of indices using numpy.random.permutation
    permutation = np.random.permutation(X.shape[0])

    # Shuffle X and Y using the same permutation
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled

def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches to be used for training a neural network using mini-batch gradient descent.

    Args:
        X: A numpy.ndarray of shape (m, nx) representing input data.
        m: The number of data points.
        nx: The number of features in X.
        Y: A numpy.ndarray of shape (m, ny) representing the labels.
        m: The same number of data points as in X.
        ny: The number of classes for classification tasks.
        batch_size: The number of data points in a batch.

    Returns:
        A list of mini-batches containing tuples (X_batch, Y_batch).
  """

    # Shuffle the data
    X, Y = shuffle_data(X, Y)

    # Calculate the number of mini-batches
    num_complete_batches = m // batch_size
    num_mini_batches = num_complete_batches + int(bool(m % batch_size))

    # Create mini-batches
    mini_batches = []
    for i in range(num_mini_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, m)
        mini_batch = (X[start_idx:end_idx], Y[start_idx:end_idx])
        mini_batches.append(mini_batch)

    return mini_batches
