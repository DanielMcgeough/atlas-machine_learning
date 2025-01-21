#!/usr/bin/env python3
"""shebang shebang"""
import numpy as np


def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) on dataset X and returns transformed data.

    Parameters:
        X (numpy.ndarray): Array of shape
        (n, d) where n is number of data
        points and d is number of dimensions
        ndim (int): Target dimensionality for
        the transformed data

    Returns:
        numpy.ndarray: Transformed data
        matrix T of shape (n, ndim)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Perform SVD
    U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

    # Project the centered data onto the principal components
    # We can do this by multiplying X_centered with the first ndim right singular vectors
    T = np.dot(X_centered, Vh.T[:, :ndim])

    return T
