#!/usr/bin/env python3
"""
Performs the Monte Carlo algorithm for value estimation.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.

    Args:
        env (object): The environment instance (expected to be a Gymnasium Env).
        V (numpy.ndarray): A numpy.ndarray of shape (s,) containing the
            value estimate for each state.
        policy (function): A function that takes in a state and returns
            the next action to take.
        episodes (int, optional): The total number of episodes to train over.
            Defaults to 5000.
        max_steps (int, optional): The maximum number of steps per episode.
            Defaults to 100.
        alpha (float, optional): The learning rate. Defaults to 0.1.
        gamma (float, optional): The discount rate. Defaults to 0.99.

    Returns:
        numpy.ndarray: V, the updated value estimate.
    """
    for episode in range(episodes):
        # Reset environment for a new episode
        # env.reset() returns (observation, info), we only need the observation
        state = env.reset()[0]
        episode_history = []  # Stores (state, reward) for the episode
        done = False
        truncated = False

        for step in range(max_steps):
            action = policy(state)  # Get action from the policy
            new_state, reward, done, truncated, _ = env.step(action)

            # IMPORTANT: Adjust reward if agent falls into a hole
            # In FrozenLake, reward is 0 for holes, but we want -1 for learning
            if done and reward == 0:
                reward = -1

            # Store current state and the (potentially modified) reward
            # received *after* transitioning from it
            episode_history.append((state, reward))
            state = new_state  # Update current state

            if done or truncated:
                break

        # Monte Carlo update for each state visited in the episode (first-visit MC)
        G = 0  # Initialize return
        # Use a set to track states for which we've already processed the first visit
        visited_states = set()

        # Iterate through the episode history in reverse to calculate returns
        # and update V
        for t in reversed(range(len(episode_history))):
            current_state, current_reward = episode_history[t]
            G = current_reward + gamma * G  # Calculate discounted return

            # Update V only if this is the first time we're seeing this state
            # in this backward pass (i.e., first-visit Monte Carlo)
            if current_state not in visited_states:
                V[current_state] = V[current_state] + alpha * (G - V[current_state])
                visited_states.add(current_state)

    return V
