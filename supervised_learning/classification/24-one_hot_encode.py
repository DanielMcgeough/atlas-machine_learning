#!/usr/bin/env python3
"""Module for converting a numeric label vector
into a one-hot matrix, whatever that means."""
import numpy as np


def one_hot_encode(Y, classes):
    """A one-hot matrix is a binary matrix
    where only one element in each row is 1,
    while all others are 0. It's a common
    representation for categorical data,
    where each unique category is assigned
    a corresponding column in the matrix."""    
    if not isinstance(Y, np.ndarray) or len(Y.shape) < 1:
        return None
    if not isinstance(classes, int) or classes < 1:
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except IndexError:
        return None
