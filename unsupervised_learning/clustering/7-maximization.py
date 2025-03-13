#!/usr/bin/env python3
"""module For maximization ugh!"""
import numpy as np


def maximization(X, g, verbose=False):
    """function that calculates the
    maximization step in the EM algorithm
    for a
    GMM
        X is a numpy.ndarray of shape
        (n, d) containing the data set
        g is a numpy.ndarray of shape
        (k, n) containing the posterior
        probabilities for each data point
        in each cluster
    Returns: pi, m, S, None, None, None
    on failure
        pi is a numpy.ndarray of shape
        (k,) contaaining the updated priors
        for each cluster
        m is a numpy.ndarray of shape
        (k, d) containing the updated centroid
        means for each cluster
        S is a numpy.ndarray of shape
        (k, d, d) containing the updated
        covariance matrices for each cluster"""
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(g, np.ndarray) or len(g.shape) != 2):
        return None, None, None

    n, d = X.shape
    k, n_2 = g.shape
    if n != n_2:
        return None, None, None

    # Check that each data point's posterior probabilities sum to 1
    if not np.allclose(g.sum(axis=0), np.ones(n)):
        return None, None, None

    # pi: shape (k,)
    pi = g.sum(axis=1) / n

    # Denominator For means/covariances:
    # shape (k, 1) so we can broadcast
    Nk = g.sum(axis=1)[:, np.newaxis]
    if np.any(Nk == 0):
        return None, None, None

    # m: shape (k, d)
    # Vectorized: g @ X yields a (k, d) result, then divide row-wise
    m = (g @ X) / Nk

    # S: shape (k, d, d), computed with at most 1 loop over k
    S = np.zeros((k, d, d))
    for i in range(k):  # Allowed single loop
        t = X - m[i]           # Center data around mean
        h = g[i] * t.T         # Multiply each column of t.T by g[i]
        S[i] = (h @ t) / np.sum(g[i])

    return pi, m, S
