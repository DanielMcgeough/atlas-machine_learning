#!/usr/bin/env python3
"""
Gaussian Mixture Model implementation using scikit-learn.
to close to sick so is the package trying to be down with the kids?
"""

import sklearn.mixture


def gmm(X, k):
    """
    Fit a Gaussian Mixture Model to input data.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int, number of clusters

    Returns:
        tuple containing:
            - numpy.ndarray of shape (k,) with cluster priors
            - numpy.ndarray of shape (k, d) with cluster means
            - numpy.ndarray of shape (k, d, d) with covariance matrices
            - numpy.ndarray of shape (n,) with cluster assignments
            - numpy.ndarray of BIC values for each k tested
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)

    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return gmm.weights_, gmm.means_, gmm.covariances_, clss, bic
