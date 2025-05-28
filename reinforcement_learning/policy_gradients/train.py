#!/usr/bin/env python3
"""runs training using monte carlo"""
import numpy as np

# Import policy_gradient function as specified by the prompt
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


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a policy using the Monte-Carlo policy gradient method.

    Args:
        env (object): The initial environment instance (expected to be a Gymnasium Env).
        nb_episodes (int): The number of episodes used for training.
        alpha (float, optional): The learning rate for updating the policy weights.
            Defaults to 0.000045.
        gamma (float, optional): The discount factor. Defaults to 0.98.

    Returns:
        list: A list containing the total reward (score) obtained in each episode.
    """
    # Determine the number of features from the environment's observation space shape
    # For continuous observation spaces (like Box), use .shape[0]
    num_features = env.observation_space.shape[0]
    # Number of actions (assumed discrete for policy gradient output)
    num_actions = env.action_space.n

    # Initialize the policy's weight matrix
    # Weight matrix shape: (num_features, num_actions)
    weight = np.random.rand(num_features, num_actions) * 0.01

    total_episode_scores = []

    # Get max steps for an episode from environment spec, default to 200 if not found
    max_steps_in_episode = getattr(env.spec, 'max_episode_steps', 200)

    for episode in range(nb_episodes):
        # Lists to store history for the current episode
        episode_states = [] # Stores raw state observations
        episode_actions = []
        episode_rewards = []
        episode_gradients = []

        state, _ = env.reset()  # Get initial state (numpy array for Box space)

        current_episode_score = 0.0

        for step in range(max_steps_in_episode):
            # Pass the raw state directly to policy_gradient as it's a feature vector
            action, gradient = policy_gradient(state, weight)

            # Take a step in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # Store the current step's data
            episode_states.append(state) # Store the raw state
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            current_episode_score += reward
            state = new_state  # Update state for next iteration

            if done or truncated:
                break

        # Calculate discounted returns for the collected episode rewards
        discounted_returns = calculate_rewards_andr(episode_rewards, gamma)

        # Update policy weights using the REINFORCE algorithm
        for t in range(len(episode_rewards)):
            G_t = discounted_returns[t]
            gradient_t = episode_gradients[t]
            weight += alpha * G_t * gradient_t

        # Print the episode number and score
        print(f"Episode: {episode + 1} Score: {current_episode_score}")
        total_episode_scores.append(current_episode_score)

    return total_episode_scores
