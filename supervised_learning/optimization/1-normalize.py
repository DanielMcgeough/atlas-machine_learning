#!/usr/bin/env python3
"""This is needed to pass checker"""
import numpy as np


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix.

    Args:
        X: A numpy.ndarray of shape (d, nx)
        to normalize.
        m: A numpy.ndarray of shape (nx,)
        that contains the mean of all features
        of X.
        s: A numpy.ndarray of shape (nx,)
        that contains the standard deviation
        of all features of X.

    Returns:
        The normalized X matrix.

  """

    # Normalize each feature by subtracting
    # the mean and dividing by the standard 
    # deviation
    X_normalized = (X - m) / s

    return X_normalized
