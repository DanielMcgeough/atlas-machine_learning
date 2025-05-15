#!/usr/bin/env python3

"""This function plays trained an agent """

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
    episode_states.append(env.render())
    done = False
    truncated = False

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, truncated, _ = env.step(action)
        rendered_state = env.render()
        episode_states.append(rendered_state)
        total_reward += reward
        state = new_state
        if done or truncated:
            break

    return total_reward, episode_states
