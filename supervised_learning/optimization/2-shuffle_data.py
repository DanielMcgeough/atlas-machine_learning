#!/usr/bin/env python3
"""This is needed to pass checker"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X: The first numpy.ndarray of shape (m, nx) to shuffle.
        m: The number of data points.
        nx: The number of features in X.
        Y: The
    second numpy.ndarray of shape (m, ny) to shuffle.
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
