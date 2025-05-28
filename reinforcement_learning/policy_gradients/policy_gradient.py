#!/usr/bin/env python3
"""computes simple policy"""
import numpy as np


def softmax(x):
    """performs softmax function as numpy doesn't have a built in one"""
    logits = np.exp(x - np.max(x))
    return logits / np.sum(logits, axis=1, keepdims=True)


def policy(matrix, weight):
    """computes policy with a weight of a matrix. You just matmul and then
    apply softmax"""
    base = np.matmul(matrix, weight)

    if base.ndim == 1:
        base = base[np.newaxis, :]

    policy_softmax = softmax(base)

    return policy_softmax


def policy_action(policy_softmax):
    """function that gets the policy of the given policy output"""
    action_probabilites = policy_softmax.squeeze(
        0)  # since output of policy is 2d and we need 1d

    num_possible_actions = len(action_probabilites)
    # choose an action
    action = np.random.choice(
        np.arange(num_possible_actions), p=action_probabilites
    )
    return action, action_probabilites


def policy_gradient(state, weight):
    """computes Monte Carlo policy gradient, returns the action
    and the log of the gradient"""
    policy_softmax = policy(
        state,
        weight)
    # applying softmax to make sure it's a valid probability
    # distribution (positive and sums to 1)

    action, action_probs = policy_action(policy_softmax)

    dlogpi = np.outer(state, np.eye(len(action_probs))[action] - action_probs)

    return action, dlogpi


def calculate_rewards_andr(rewards, gamma):
    """calculates discounted rewards"""
    DiscountedReturns = []
    for t in range(len(rewards)):
        G = 0.0
        for k, r in enumerate(rewards[t:]):
            G += (gamma ** k) * r
        DiscountedReturns.append(G)
    return DiscountedReturns


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, max_steps=500):
    """trains a policy using monte carlo method"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    theta = np.random.randn(state_dim, action_dim) * 0.01
    total_rewards = []
    episode_reward = 0.0

    for episode in range(nb_episodes):
        states, actions, rewards,  = [], [], []
        episode_rewards, gradients = [], []
        state, _ = env.reset()
        for step in range(max_steps):
            action, gradient = policy_gradient(state, theta)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            gradients.append(gradient)

            # episode_reward += reward * (gamma ** step)
            state = next_state
            if done:
                break

        discounted_rewards = calculate_rewards_andr(rewards, gamma)
        # episode_rewards.append(episode_reward)

        # for t, (loss_state, loss_action, G_t) in enumerate(zip(states,
        # actions, discounted_rewards)):
        for t in range(len(rewards)):
            G_t = discounted_rewards[t]
            theta += alpha * G_t * gradients[t]

        print(f"Episode: {episode} Score: {sum(rewards)}")
        total_rewards.append(sum(rewards))

    return total_rewards
