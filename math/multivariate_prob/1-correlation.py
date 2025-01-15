#!/usr/bin/env python3
"""Fills my brain up to capacity"""
import numpy as np

def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.
    
    Parameters:
        C: numpy.ndarray of shape (d, d) containing the covariance matrix
        
    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    # Check if C is a numpy array
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    
    # Check if C is 2D square matrix
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    
    # Get the diagonal elements (variances)
    variances = np.diag(C)
    
    # Calculate standard deviations
    std_devs = np.sqrt(variances)
    
    # Calculate correlation matrix using the formula:
    # correlation[i,j] = covariance[i,j] / (std_dev[i] * std_dev[j])
    correlation_matrix = C / np.outer(std_devs, std_devs)
    
    return correlation_matrix
