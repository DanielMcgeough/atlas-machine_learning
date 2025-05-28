#!/usr/bin/env python3
"""
Implements a full training loop for a Monte-Carlo policy gradient agent.
"""

import numpy as np

# Import policy_gradient function from its module as specified.
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training loop for a Monte-Carlo policy gradient agent.

    Args:
        env (object): The initial environment instance (expected to be a Gymnasium Env).
        nb_episodes (int): The number of episodes used for training.
        alpha (float, optional): The learning rate. Defaults to 0.000045.
        gamma (float, optional): The discount factor. Defaults to 0.98.

    Returns:
        list: A list containing the score (sum of all rewards during one
              episode loop) for each episode.
    """
    # Initialize weights randomly
    # Number of states (features) for one-hot encoding
    num_states = env.observation_space.n
    # Number of actions
    num_actions = env.action_space.n
    # Weight matrix shape: (num_states, num_actions)
    weight = np.random.rand(num_states, num_actions)

    scores = []

    for episode in range(nb_episodes):
        state = env.reset()[0]  # Get initial state (integer index)
        done = False
        truncated = False
        episode_history = []  # Stores (state_one_hot, action, reward, gradient)
        current_episode_score = 0

        # Loop for steps within an episode
        # Use env.spec.max_episode_steps to get the maximum steps for the environment
        # If not available, a reasonable default like 100 or 500 can be used.
        # For FrozenLake, max_episode_steps is usually 200 for 8x8.
        max_steps_in_episode = getattr(env.spec, 'max_episode_steps', 200)

        for step in range(max_steps_in_episode):
            # Convert integer state to one-hot encoding
            state_one_hot = np.zeros(num_states)
            state_one_hot[state] = 1

            # Get action and gradient from the policy
            action, gradient = policy_gradient(state_one_hot, weight)

            # Take a step in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # Store the experience for later update
            episode_history.append((state_one_hot, action, reward, gradient))
            current_episode_score += reward
            state = new_state

            if done or truncated:
                break

        # Update weights using REINFORCE algorithm
        # Calculate returns (G_t) for each step in the episode
        returns = []
        G = 0
        # Iterate through the episode history in reverse to calculate discounted returns
        for t in reversed(range(len(episode_history))):
            # episode_history[t] is (state_one_hot, action, reward, gradient)
            # We only need the reward for return calculation
            _, _, reward_t, _ = episode_history[t]
            G = reward_t + gamma * G
            returns.insert(0, G) # Insert at the beginning to maintain original order of returns

        # Apply the weight update for each step using the calculated returns
        for t in range(len(episode_history)):
            # Get the gradient calculated at this specific step
            _, _, _, gradient_t = episode_history[t]
            G_t = returns[t] # Get the corresponding return for this step
            weight += alpha * gradient_t * G_t

        # Print current episode number and score in the specified format
        print(f"Episode: {episode + 1} Score: {current_episode_score}")
        scores.append(current_episode_score)

    return scores
