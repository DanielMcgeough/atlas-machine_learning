#!/usr/bin/env python3
"""Modules for HMM assignment 0"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov
    chain being in a particular state
    after a specified number of iterations.
    Args:
        P: A square 2D numpy.ndarray of shape
        (n, n) representing the transition
        matrix.
        P[i, j] is the probability of
        transitioning from state i to state j.
        s: A numpy.ndarray of shape (1, n)
        representing the probability of
        starting
        in each state.
        t: The number of iterations that the
        markov chain has been through.
        Defaults to 1.
    Returns:
        A numpy.ndarray of shape (1, n)
        representing the probability of
        being in
        a specific state after t iterations,
        or None on failure.
    """

    try:
        result = np.matmul(s, np.linalg.matrix_power(P, t))
        return result
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None
