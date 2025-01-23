#!/usr/bin/env python3
"""shebang for my intro to K-means"""
import numpy as np


def initialize(X, k):
    """function that intializes cluster
    cetnroids for k-means. Centroids
    are randomly placed points that attract
    data points around them to create clusters
    that will help users identify patterns in
    the data."""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    minimum = X.min(axis=0)
    maximum = X.max(axis=0)

    return np.random.uniform(low=minimum, high=maximum, size=(k, X.shape[1]))
