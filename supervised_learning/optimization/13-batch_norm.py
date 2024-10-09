#!/usr/bin/env python3
"""shorter than I remember"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch normalization.   


    Args:
        Z: A numpy.ndarray of shape (m, n) that should be normalized.
        m: The number of data points.
        n: The number of features in Z.
        gamma: A numpy.ndarray of shape (1, n) containing the scales used for batch normalization.
        beta: A numpy.ndarray of shape (1, n) containing the offsets used for batch normalization.
        epsilon: A small number used to avoid division by zero.

    Returns:
        The normalized Z matrix.

    """

    # Calculate the mean and variance of each feature
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    # Normalize the data
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_out = gamma * Z_norm + beta

    return Z_out
