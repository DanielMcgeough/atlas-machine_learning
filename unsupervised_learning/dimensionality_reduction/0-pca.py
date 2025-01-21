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
    
    # Calculate variance explained by each component
    # Singular values are related to eigenvalues: λ = σ²/n
    eigenvalues = (S ** 2) / (X.shape[0] - 1)
    
    # Calculate cumulative sum of explained variance ratios
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(variance_ratio)
    
    # Find number of components needed to maintain desired variance
    n_components = np.argmax(cumulative_variance_ratio >= var) + 1
    
    # Return weight matrix W using selected right singular vectors
    W = Vh.T[:, :n_components]
    
    return W
