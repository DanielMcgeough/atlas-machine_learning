#!/usr/bin/env python3
"""Implements the Q-learning algorithm for
training an agent."""

import gymnasium as gym
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1.0,
          min_epsilon=0.1, epsilon_decay=0.001):
    """
    Performs Q-learning to train an agent
    in an environment.

    This function implements the Q-learning
    algorithm, which learns an optimal
    policy by iteratively updating the
    Q-table based on the agent's
    interactions with the environment.

    Args:
        env (gym.Env): The Gymnasium
        environment to train in.  It
        is assumed
            that the environment has a
            discrete state space and
            discrete
            action space.
        Q (numpy.ndarray): The Q-table, a
        2D numpy.ndarray where
            Q[state, action] represents the
            estimated value of taking
            a given action in a given state.
        episodes (int, optional): The number
        of training episodes.
            Defaults to 5000.
        max_steps (int, optional): The
        maximum number of steps allowed
            per episode.  If an episode
            does not reach a terminal state
            within this many steps, it is
            truncated. Defaults to 100.
        alpha (float, optional): The
        learning rate, controlling how much
            the Q-values are updated in each
            iteration.
            Defaults to 0.1.
        gamma (float, optional): The discount
        factor, controlling the
            importance of future rewards.
            Defaults to 0.99.
        epsilon (float, optional): The
        initial epsilon value for
            epsilon-greedy exploration.
            This is the probability of
            taking a random action. Defaults
            to 1.0.
        min_epsilon (float, optional):
        The minimum epsilon value.
            Epsilon decays over training,
            and this is the lowest value
            it will reach. Defaults to 0.1.
        epsilon_decay (float, optional):
        The amount by which epsilon is
            reduced after each episode.
            Defaults to 0.001.

    Returns:
        tuple: A tuple containing the
        updated Q-table and a list of
            total rewards per episode.
            - Q (numpy.ndarray): The updated
            Q-table after training.
            - total_rewards (list): A list of
            length `episodes`, where
              each element is the sum of
              rewards received in that
              episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]  # Get initial state from the reset
        episode_reward = 0

        for step in range(max_steps):
            # Epsilon-greedy action selection
            p = np.random.uniform()
            if p < epsilon:
                # Explore: Choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: Choose the action with the highest Q-value
                action = np.argmax(Q[state, :])

            # Take the action and observe the outcome
            new_state, reward, done, truncated, _ = env.step(action)

            # Adjust reward for falling in a hole (if applicable)
            if done and reward == 0:  # Assuming 0 reward for hole in FrozenLake
                reward = -1

            # Q-table update (the core Q-learning update rule)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
            )

            # Move to the next state
            state = new_state
            episode_reward += reward

            if done or truncated:
                break  # End episode if done or truncated

        # Update epsilon for the next episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)

    return Q, total_rewards
