#!/usr/bin/env python3
"""shebang"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on dataset X using SVD.

    Parameters:
        X (numpy.ndarray): Array of shape (n, d) where n is number of data points
                          and d is number of dimensions. All dimensions should have
                          mean of 0 across data points.
        var (float): Fraction of variance to maintain (default: 0.95)

    Returns:
        numpy.ndarray: Weights matrix W of shape (d, nd) where nd is the new
                      dimensionality that maintains var fraction of variance
    """
    # Perform SVD
    # U: left singular vectors (n x n)
    # S: singular values (min(n,d))
    # Vh: right singular vectors (d x d)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    # Calculate explained variance ratio
    explained_variance = (s ** 2) / np.sum(s ** 2)

    # Calculate cumulative sum of variance ratios
    cumulative_variance = np.cumsum(explained_variance)

    # get number of components
    n_components = 1 + np.where(cumulative_variance >= var)[0][0] + 1
    W = Vh.T[:, :n_components]
    
    return W
