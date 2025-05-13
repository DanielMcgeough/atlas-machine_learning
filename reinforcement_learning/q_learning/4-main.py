#!/usr/bin/env python3
"""Implements the Q-learning algorithm for training an agent."""

import gymnasium as gym
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode with the trained agent (exploiting the Q-table).

    Args:
        env (gym.Env): The FrozenLakeEnv instance.
        Q (numpy.ndarray): The trained Q-table.
        max_steps (int, optional): The maximum number of steps in the episode.
            Defaults to 100.

    Returns:
        tuple: (total_reward, episode_states)
            - total_reward (float): The total reward for the episode.
            - episode_states (list): A list of strings, where each string
              represents the rendered board state at each step.
    """
    state = env.reset()[0]
    total_reward = 0
    episode_states = []
    done = False
    truncated = False

    for step in range(max_steps):
        # Exploit: Choose the action with the highest Q-value
        action = np.argmax(Q[state, :])
        new_state, reward, done, truncated, _ = env.step(action)

        # Render the current state of the environment as a string
        rendered_state = env.render()
        episode_states.append(rendered_state)
        print(rendered_state) # Print the current state

        total_reward += reward
        state = new_state

        if done or truncated:
            break

    # Print the final state of the environment.
    final_rendered_state = env.render()
    episode_states.append(final_rendered_state)
    print(final_rendered_state)
    return total_reward, episode_states
