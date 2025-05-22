#!/usr/bin/env python3

"""
Monte Carlo was the first task, it can be quick but TD updates in real time and
although more expensive, seems to require less episodes. It is also the name
of this project I mean come on!
"""

import numpy as np


def td_lambtha(
    env,
    V,
    policy,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99
):
    """
    Performs the TD(λ) algorithm for value
    estimation.

    Args:
        env (object): The environment instance
        (expected to be a Gymnasium Env).
        V (numpy.ndarray): A numpy.ndarray of
        shape (s,) containing the
            value estimate for each state.
        policy (function): A function that
        takes in a state and returns
            the next action to take.
        lambtha (float): The eligibility
        trace factor.
        episodes (int, optional): The total
        number of episodes to train over.
            Defaults to 5000.
        max_steps (int, optional): The maximum
        number of steps per episode.
            Defaults to 100.
        alpha (float, optional): The learning
        rate. Defaults to 0.1.
        gamma (float, optional): The discount
        rate. Defaults to 0.99.

    Returns:
        numpy.ndarray: V, the updated value
        estimate.
    """

    for episode in range(episodes):
        state, _ = env.reset()
        e_traces = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            td_error = reward + gamma * V[next_state] - V[state]

            e_traces[state] += 1

            V += alpha * td_error * e_traces

            e_traces *= gamma * lambtha

            state = next_state

            if done:
                break

    return V
