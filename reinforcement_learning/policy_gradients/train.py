#!/usr/bin/env python3
"""documentation"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """documentation"""
    initial_state = env.reset()
    if isinstance(initial_state, tuple):
        state = initial_state[0]
    else:
        state = initial_state

    n_states = len(state)
    n_actions = env.action_space.n

    weights = np.random.normal(size=(n_states, n_actions))

    all_scores = []

    for episode in range(nb_episodes):
        initial_state = env.reset()
        if isinstance(initial_state, tuple):
            state = initial_state[0]
        else:
            state = initial_state

        episode_rewards = []
        episode_gradients = []

        done = False
        while not done:

            action, gradient = policy_gradient(state, weights)

            step_result = env.step(action)

            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            else:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            state = next_state

        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(episode_rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean
                              (discounted_rewards)) / \
            (np.std(discounted_rewards) + 1e-10)

        for gradient, discounted_reward in zip(episode_gradients,
                                               discounted_rewards):
            weights += alpha * gradient * discounted_reward

        episode_score = sum(episode_rewards)
        all_scores.append(episode_score)

        print("Episode: {} Score: {}".format(episode, episode_score))

    return all_scores
