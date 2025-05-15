#!/usr/bin/env python3
"""Loads the FrozenLake environment from
Gymnasium."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv
    environment from gymnasium.

    Args:
        desc (list of lists, optional):
        A custom description of the map
            to load for the environment.
            Defaults to None.
        map_name (str, optional): The name
        of the pre-made map to load.
            Defaults to None.
        is_slippery (bool, optional):
        Boolean to determine if the ice
        is slippery.
            Defaults to False.

    Returns:
        gym.Env: The FrozenLake environment.
    """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi",  # Ensure we can render in a text-based environment
    )

    return gym.Env
