#!/usr/bin/env python3
"""math of this type makes
more sense to me"""
import numpy as np


def definiteness(matrix):
    """write a function that calculates the
    definiteness of a matrix.
    it tells you about the curvature or shape
    associated with a matrix. They can be
    positive definite, negative definite,
    positive semi definite and negative semi
    definite and indefinite. it depends on
    the eigenvalues for these definite
    classifications.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None