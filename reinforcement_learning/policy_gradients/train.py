#!/usr/bin/env python3
"""
Implements a full training loop for a Monte-Carlo policy gradient agent.
"""

import numpy as np

# Import policy_gradient function from its module as specified.
# This assumes 'policy_gradient.py' exists in the same directory
# and contains both the 'policy' and 'policy_gradient' functions.
policy_gradient = __import__('policy_gradient').policy_gradient


def calculate_rewards_andr(rewards, gamma):
    """
    Calculates the discounted returns (G_t) for each step in an episode.

    Args:
        rewards (list): A list of rewards received at each step of an episode.
        gamma (float): The discount factor.

    Returns:
        list: A list of discounted returns, where DiscountedReturns[t] is
              the sum of future discounted rewards from time step t.
    """
    DiscountedReturns = []
    # Iterate backwards through rewards to efficiently calculate discounted returns
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        DiscountedReturns.insert(0, G)  # Insert at the beginning to maintain original order
    return DiscountedReturns


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Trains a policy using the Monte-Carlo policy gradient method.

    Args:
        env (object): The initial environment instance (expected to be a Gymnasium Env).
        nb_episodes (int): The number of episodes used for training.
        alpha (float, optional): The learning rate for updating the policy weights.
            Defaults to 0.000045.
        gamma (float, optional): The discount factor. Defaults to 0.98.
        show_result (bool, optional): If True, renders the environment every
            1000 episodes. Defaults to False.

    Returns:
        list: A list containing the total reward (score) obtained in each episode.
    """
    # Set a random seed for reproducibility.
    # This is crucial for consistent results in stochastic environments and algorithms.
    np.random.seed(0)

    # Determine the number of features from the environment's observation space shape.
    # For environments with Box observation spaces (e.g., CartPole),
    # env.observation_space.shape returns a tuple, and we need the first element.
    num_features = env.observation_space.shape[0]
    # Number of actions (assumed discrete for policy gradient output).
    num_actions = env.action_space.n

    # Initialize the policy's weight matrix.
    # Weight matrix shape: (num_features, num_actions).
    # Initialized with small random values to avoid large initial gradients.
    weight = np.random.rand(num_features, num_actions) * 0.01

    total_episode_scores = []

    # Get maximum steps per episode from environment specification.
    # Defaults to 200 if 'max_episode_steps' is not defined in env.spec.
    max_steps_in_episode = getattr(env.spec, 'max_episode_steps', 200)

    for episode in range(nb_episodes):
        # Lists to store the history of the current episode for Monte Carlo updates.
        episode_states = []    # Stores raw state observations (feature vectors)
        episode_actions = []   # Stores actions taken
        episode_rewards = []   # Stores rewards received
        episode_gradients = [] # Stores gradients computed at each step

        # Reset the environment for a new episode.
        # env.reset() returns (observation, info), we only need the observation.
        state, _ = env.reset()

        current_episode_score = 0.0

        # Render environment if show_result is True and it's every 1000th episode (and not the very first episode)
        if show_result and episode % 1000 == 0 and episode > 0:
            env.render()

        # Loop through steps within the current episode.
        for step in range(max_steps_in_episode):
            # The 'state' is already a feature vector (numpy array) for Box spaces,
            # so it can be passed directly to policy_gradient.
            action, gradient = policy_gradient(state, weight)

            # Take the chosen action in the environment.
            new_state, reward, done, truncated, _ = env.step(action)

            # Store the data for the current step.
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            current_episode_score += reward
            state = new_state  # Update the current state for the next iteration.

            # Check if the episode has terminated or been truncated.
            if done or truncated:
                break

        # Calculate discounted returns (G_t) for the collected episode rewards.
        # This is done after the episode completes, characteristic of Monte Carlo.
        discounted_returns = calculate_rewards_andr(episode_rewards, gamma)

        # Update policy weights using the REINFORCE algorithm.
        # The update is applied for each step in the episode.
        for t in range(len(episode_rewards)):
            G_t = discounted_returns[t]        # The return from time step t onwards.
            gradient_t = episode_gradients[t]  # The gradient computed at time step t.
            weight += alpha * G_t * gradient_t # Apply the gradient ascent update.

        # Print the current episode number and its total score.
        # Episode numbers are 1-based for user readability.
        print(f"Episode: {episode + 1} Score: {current_episode_score}")
        total_episode_scores.append(current_episode_score)

    return total_episode_scores
