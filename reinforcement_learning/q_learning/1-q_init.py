#!/usr/bin/env python3
"""Loads the FrozenLake environment from
Gymnasium."""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env (gym.Env): The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: The Q-table as a numpy.ndarray of zeros.
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
