#!/usr/bin/env python3
"""
Module that computes a policy's action
probabilities given input features and weights.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy's action probabilities
    using a linear transformation
    followed by a softmax activation.

    This function represents a common way
    to define a stochastic policy in
    reinforcement learning, where an input
    (e.g., state features) is
    transformed by a weight matrix to produce
    logits, which are then converted
    into a probability distribution over
    actions using the softmax function.

    Args:
        matrix (numpy.ndarray): The input to
        the policy. This can represent
            features of a single state
            (shape: `(n_features,)`) or
            features
            for a batch of states
            (shape: `(batch_size, n_features)`).
        weight (numpy.ndarray): The policy's
        weight matrix. Its shape should be
            `(n_features, n_actions)`, where
            `n_features` matches the number
            of features in `matrix`, and
            `n_actions` is the number of
            possible actions.

    Returns:
        numpy.ndarray: A numpy.ndarray
        representing the probability
        distribution
            over actions.
            - If `matrix` is `(n_features,)`,
            the output shape will be
              `(n_actions,)`.
            - If `matrix` is `(batch_size,
            n_features)`, the output shape will
            be`(batch_size, n_actions)`, with
              each row being a probability
              distribution for the
              corresponding state in the batch.
    """
    # Calculate the logits by performing a
    # matrix multiplication.
    # np.dot handles both 1D and 2D 'matrix'
    # inputs correctly.
    # If matrix is (n_features,), logits
    # will be (n_actions,).
    # If matrix is (batch_size, n_features),
    # logits will be (batch_size, n_actions).
    logits = np.dot(matrix, weight)

    # Apply softmax function for numerical
    # stability.
    # Subtracting the maximum value from
    # logits before exponentiation
    # prevents potential overflow issues
    # with very large numbers,
    # while not changing the final
    # probabilities.
    # np.max(logits, axis=-1, keepdims=True)
    # ensures correct behavior
    # for both single state and batch inputs.
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))

    # Normalize the exponentiated logits to
    # get probabilities.
    # Summing along the last axis (actions)
    # and keeping dimensions
    # ensures correct division for both
    # single state and batch inputs.
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return probabilities
