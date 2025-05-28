#!/usr/bin/env python3

"""Yep we still don't know what we're doing here.
But we'll give it the old college try"""

import numpy as np


def policy(matrix, weight):
    """I guess all it is is a multiplication of the matrixes
    Nope it sure as hecky becky ain't."""
    logits = np.dot(matrix, weight)

    if logits.ndim == 1:
        logits = logits[np.newaxis, :]

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return softmax_probs


def policy_gradient(state, weight):
    """Gets the montecalro policy gradient
    Which I only kind of understand."""

    action_probabilities = policy(state, weight)
    action_probabilities = action_probabilities[0]

    action = np.random.choice(
        np.arange(len(action_probabilities)), p=action_probabilities
    )

    gradient = np.zeros_like(weight)

    for i in range(len(action_probabilities)):
        if i == action:
            # Aparantly this is softmax as much as the last one is.
            gradient[:, i] = state.flatten() * (1 - action_probabilities[i])
        else:
            # Non selected actions are negative I guess.
            gradient[:, i] = -state.flatten() * action_probabilities[i]

    return action, gradient
