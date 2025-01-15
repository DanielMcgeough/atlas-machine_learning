#!/bin/usr/env python3
"""Doing my level best."""
import numpy as np


class MultiNormal:
    """
    Class that represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        Initialize the MultiNormal distribution with a dataset

        Parameters:
            data: numpy.ndarray of shape (d, n) containing n d-dimensional data points
        """
        # Check if data is a 2D numpy array
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Get dimensions (note: shape is (d, n) here)
        d, n = data.shape

        # Check if there are multiple data points
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean along n axis and keep dimensions
        self.mean = np.mean(data, axis=1, keepdims=True)  # shape (d, 1)

        # Center the data
        centered_data = data - self.mean

        # Calculate covariance matrix
        # For shape (d, n), formula is: (1/(n-1)) * centered_data @ centered_data.T
        self.cov = np.matmul(centered_data, centered_data.T) / (n - 1)  # shape (d, d)
