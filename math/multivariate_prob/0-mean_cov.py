#!/usr/bin/env python3
"""Starting off with Multivariate
Gaussian Distributions"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance matrix of a dataset.

    Parameters:
        X: numpy.ndarray of shape (n, d) containing n d-dimensional data points

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
    """
    # Check if X is a 2D numpy array
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Get dimensions
    n, d = X.shape

    # Check if there are multiple data points
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data
    X_centered = X - mean

    # Calculate covariance matrix
    # Formula: cov = (1/(n-1)) * X_centered.T @ X_centered
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
