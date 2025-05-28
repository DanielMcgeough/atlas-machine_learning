#!/usr/bin/env python3
"""runs training using monte carlo"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def calculate_rewards_andr(rewards, gamma):
    """calculates discounted rewards"""
    DiscountedReturns = []
    for t in range(len(rewards)):
        G = 0.0
        for k, r in enumerate(rewards[t:]):
            G += (gamma ** k) * r
        DiscountedReturns.append(G)
    return DiscountedReturns


def train(
        env,
        nb_episodes,
        alpha=0.000045,
        gamma=0.98,
        max_steps=500,
        show_result=False):
    """trains a policy using monte carlo method"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    theta = np.random.randn(state_dim, action_dim) * 0.01
    total_rewards = []
    episode_reward = 0.0

    for episode in range(nb_episodes):
        states, actions, rewards, = [], [], []
        episode_rewards, gradients = [], []
        state, _ = env.reset()
        for step in range(max_steps):
            if show_result and (episode % 1000 == 0) and episode > 0:
                env.render()
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
