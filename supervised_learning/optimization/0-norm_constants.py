#!/usr/bin/env python3
"""This is needed to pass checker"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix.

    Args:
        X: A numpy.ndarray of shape (m, nx) to normalize.

        m: The number of data points.
        nx: The number of features.

    Returns:
        The mean and standard deviation of each feature, respectively.
  """

    # Calculate the mean of each feature
    mean = np.mean(X, axis=0)

    # Calculate the standard deviation of each feature
    std_dev = np.std(X, axis=0)

    return mean, std_dev
