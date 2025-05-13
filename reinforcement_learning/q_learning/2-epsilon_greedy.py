#!/usr/bin/env python3
"""Loads the FrozenLake environment from
Gymnasium."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next
    action.

    Args:
        Q (numpy.ndarray): A numpy.ndarray
        containing the q-table.
        state (int): The current state.
        epsilon (float): The epsilon value.

    Returns:
        int: The next action index.
    """
    p = np.random.uniform()
    if p < epsilon:
        # Explore: Choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: Choose the action with the highest Q-value
        action = np.argmax(Q[state, :])
    return action
