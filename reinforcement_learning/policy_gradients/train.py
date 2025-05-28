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
    # Determine the number of states and actions from the environment
    # For discrete observation spaces (like FrozenLake), use .n
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the policy's weight matrix (theta renamed to weight for consistency)
    # Using np.random.rand for random initialization as per common practice.
    # Scaled by 0.01 to keep initial values small.
    weight = np.random.rand(num_states, num_actions) * 0.01

    total_episode_scores = []  # Renamed from total_rewards for clarity as per prompt

    # Get max steps for an episode from environment spec, default to 200 if not found
    max_steps_in_episode = getattr(env.spec, 'max_episode_steps', 200)

    for episode in range(nb_episodes):
        # Lists to store history for the current episode
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_gradients = []  # Store the gradient computed at each step

        state, _ = env.reset()  # Reset environment and get initial state

        current_episode_score = 0.0 # Track score for current episode

        for step in range(max_steps_in_episode):
            # Convert integer state to one-hot encoded vector
            # This is crucial because policy_gradient expects a feature vector/matrix
            state_one_hot = np.zeros(num_states)
            state_one_hot[state] = 1

            # Get action and the corresponding gradient from the policy
            action, gradient = policy_gradient(state_one_hot, weight)
            next_state, reward, done, truncated, _ = env.step(action)

            # Adjust reward if agent falls into a hole (common for FrozenLake)
            if done and reward == 0:
                reward = -1

            # Store the current step's data
            episode_states.append(state_one_hot)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            current_episode_score += reward
            state = next_state  # Update state for next iteration

            if done or truncated:  # Check for both done and truncated
                break

        # Calculate discounted returns for the collected episode rewards
        discounted_returns = calculate_rewards_andr(episode_rewards, gamma)

        # Update policy weights using the REINFORCE algorithm
        # For each step in the episode:
        for t in range(len(episode_rewards)):
            G_t = discounted_returns[t]  # Return from time t onwards
            gradient_t = episode_gradients[t]  # Gradient computed at time t for the taken action
            weight += alpha * G_t * gradient_t

        # Print the episode number and score as per the specified format
        print(f"Episode: {episode + 1} Score: {current_episode_score}")
        total_episode_scores.append(current_episode_score)

    return total_episode_scores
