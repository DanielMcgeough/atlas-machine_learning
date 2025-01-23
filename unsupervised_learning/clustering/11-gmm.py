#!/usr/bin/env python3
"""Wooo for no comprehension."""
import sklearn.mixture
import numpy as np


def gmm(X, k):
    """
    Calculate a Gaussian Mixture Model from a dataset using scikit-learn.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Dataset of shape (n, d)
    k : int
        Number of clusters
    
    Returns:
    --------
    pi : numpy.ndarray
        Cluster priors of shape (k,)
    m : numpy.ndarray
        Centroid means of shape (k, d)
    S : numpy.ndarray
        Covariance matrices of shape (k, d, d)
    clss : numpy.ndarray
        Cluster indices for each data point of shape (n,)
    bic : float
        Bayesian Information Criterion value
    """
    # Validate input
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    
    n, d = X.shape
    
    # Validate k
    if not isinstance(k, int) or k < 1 or k > n:
        return None, None, None, None, None
    
    try:
        # Create and fit Gaussian Mixture Model
        gmm = sklearn.mixture.GaussianMixture(
            n_components=k, 
            covariance_type='full',
            n_init=10,  # Multiple initializations to avoid local optima
            random_state=42  # For reproducibility
        )
        gmm.fit(X)
        
        # Extract cluster priors (weights)
        pi = gmm.weights_
        
        # Extract cluster means
        m = gmm.means_
        
        # Extract covariance matrices
        S = gmm.covariances_
        
        # Predict cluster assignments
        clss = gmm.predict(X)
        
        # Get BIC
        bic = gmm.bic(X)
        
        return pi, m, S, clss, bic
    
    except Exception:
        return None, None, None, None, None
