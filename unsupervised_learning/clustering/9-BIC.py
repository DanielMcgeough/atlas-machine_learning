#!/usr/bin/env python3
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a Gaussian Mixture Model using Bayesian Information Criterion.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Dataset of shape (n, d)
    kmin : int, optional
        Minimum number of clusters to check (default 1)
    kmax : int or None, optional
        Maximum number of clusters to check (default None, which sets to max possible)
    iterations : int, optional
        Maximum iterations for EM algorithm (default 1000)
    tol : float, optional
        Tolerance for EM algorithm (default 1e-5)
    verbose : bool, optional
        Whether to print EM algorithm information (default False)
    
    Returns:
    --------
    best_k : int or None
        Best number of clusters based on BIC
    best_result : tuple or None
        Tuple of (pi, m, S) for the best clustering
    l : numpy.ndarray or None
        Log likelihoods for each cluster size
    b : numpy.ndarray or None
        BIC values for each cluster size
    """
    # Import the expectation maximization function
    try:
        expectation_maximization = __import__('8-EM').expectation_maximization
    except (ImportError, AttributeError):
        return None, None, None, None
    
    # Validate input
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    
    n, d = X.shape
    
    # Set kmax to maximum possible clusters if not specified
    if kmax is None:
        kmax = n
    
    # Validate parameters
    if (not isinstance(kmin, int) or kmin < 1 or 
        not isinstance(kmax, int) or kmax < kmin or 
        not isinstance(iterations, int) or iterations < 1 or 
        not isinstance(tol, float) or tol < 0 or 
        not isinstance(verbose, bool)):
        return None, None, None, None
    
    # Initialize arrays to store log likelihoods and BIC values
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    
    # Track best results
    best_k = None
    best_result = None
    best_bic = float('-inf')
    
    # Iterate through possible cluster numbers
    for k in range(kmin, kmax + 1):
        try:
            # Estimate GMM parameters using EM
            pi, m, S, log_l = expectation_maximization(X, k, iterations, tol, verbose)
            
            # Store log likelihood
            l[k - kmin] = log_l
            
            # Calculate number of parameters
            # Number of mix coefficients: k-1
            # Number of means: k * d
            # Number of covariance elements: k * d*(d+1)/2
            p = (k - 1) + k * d + k * d * (d + 1) // 2
            
            # Calculate BIC
            bic = p * np.log(n) - 2 * log_l
            b[k - kmin] = bic
            
            # Update best result if current BIC is better
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_result = (pi, m, S)
        
        except Exception:
            # If EM fails for any reason, continue to next k
            continue
    
    return best_k, best_result, l, b
