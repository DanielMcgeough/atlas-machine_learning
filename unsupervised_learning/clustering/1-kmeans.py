#!/usr/bin/env python3
"""another shebang for k-means"""
import numpy as np


import numpy as np

def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on dataset X.
    
    Parameters:
    X: numpy.ndarray of shape (n, d) containing the dataset
    k: positive integer containing the number of clusters
    iterations: positive integer for maximum number of iterations
    
    Returns:
    C: numpy.ndarray of shape (k, d) containing the centroid means
    clss: numpy.ndarray of shape (n,) containing the cluster indices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    try:
        n, d = X.shape
        
        # Initialize centroids using multivariate uniform distribution
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        C = np.random.uniform(low=X_min, high=X_max, size=(k, d))
        
        for i in range(iterations):
            # Store old centroids to check for convergence
            C_old = C.copy()
            
            # Calculate distances between each point and each centroid
            distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
            
            # Assign points to nearest centroid
            clss = np.argmin(distances, axis=0)
            
            # Update centroids
            for j in range(k):
                points = X[clss == j]
                if len(points) == 0:
                    # Reinitialize empty cluster
                    C[j] = np.random.uniform(low=X_min, high=X_max, size=d)
                else:
                    C[j] = np.mean(points, axis=0)
            
            # Check for convergence
            if np.all(C == C_old):
                break
                
        return C, clss
    
    except Exception:
        return None, None


def initialize(X, k):
    """function that initializes cluster centroids for K-means:
        X is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
        The cluster centroids should be initialized with a multivariate
        uniform distrubution along eah dimension in d:
            The minimum values for the distribution should be the ninimum
            values of X along each dimension in d
            The maximium values for the distribution should be the maximum
            values of X along each dimension in d
            You should use numpy.random.uniform exactly once
        Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure."""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    minimum = X.min(axis=0)
    maximum = X.max(axis=0)

    return np.random.uniform(low=minimum, high=maximum, size=(k, X.shape[1]))
