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

def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient
    for a given state and weight matrix.

    This function calculates the gradient
    of the log-policy with respect to
    the policy's weights for a sampled
    action. This gradient is a key component
    of policy gradient methods (e.g.,
    REINFORCE algorithm), where it's scaled
    by the observed return (G_t) and then
    used to update the policy weights.

    Args:
        state (numpy.ndarray): A numpy.ndarray
        representing the current
            observation (features) of the
            environment. Expected shape is
            `(n_features,)` for a single state,
            or `(1, n_features)` if
            passed as a single-sample batch.
        weight (numpy.ndarray): A numpy.ndarray
        representing the policy's
            weight matrix. Expected shape
            is `(n_features, n_actions)`.

    Returns:
        tuple: A tuple containing:
            - int: The action sampled from
            the policy's probability
            distribution.
            - numpy.ndarray: The gradient of
            the log-policy with respect to
              the weight matrix for the
              sampled action. Its shape is
              `(n_features, n_actions)`.
    """
    # 1. Get action probabilities from the
    # policy
    # The 'policy' function can handle both
    # (n_features,) and (1, n_features) inputs.
    # We flatten the result to ensure it's a
    # 1D array of probabilities for
    # np.random.choice.
    probabilities = policy(state, weight).flatten()

    # 2. Sample an action from the probability
    # distribution
    num_actions = probabilities.shape[0]
    action = np.random.choice(num_actions, p=probabilities)

    # 3. Compute the gradient of the
    # log-policy with respect to the weights
    # Create a one-hot vector for the
    # sampled action.
    one_hot_action = np.zeros(num_actions)
    one_hot_action[action] = 1

    # The core policy gradient identity is:
    # Ensure state is 1D for np.outer.
    # If state was (1, n_features), flatten
    # it to (n_features,).
    if state.ndim > 1:
        state_flat = state.flatten()
    else:
        state_flat = state

    # Calculate the gradient matrix using
    # the outer product.
    # The outer product of (n_features,)
    # and (n_actions,) results in
    # (n_features, n_actions).
    gradient = np.outer(state_flat, (one_hot_action - probabilities))

    return action, gradient
