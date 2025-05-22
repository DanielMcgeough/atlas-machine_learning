#!/usr/bin/env python3
"""uses monte carlo method"""
import numpy as np


def monte_carlo(
        env,
        V,
        policy,
        episodes=5000,
        max_steps=100,
        gamma=0.99,
        alpha=0.1):
    """
    Monte Carlo evaluation with First-Visit MC.

    Parameters:
    -----------
    env : gymnasium.Env
        The environment instance.
    policy : callable
        Function mapping state -> action.
    episodes : int
        Number of episodes to run.
    max_steps : int
        Max steps per episode.
    gamma : float
        Discount factor.

    Returns:
    --------
    V : np.ndarray
        Estimated value function (shape depends on env.observation_space.n).
    """

    for episode in range(episodes):
        visitedStatesInEpisode = []

        rewardInVisitedState = []
        currentState, prob = env.reset()

        for _ in range(max_steps):
            # randomAction = env.action_space.sample()
            policyAction = policy(currentState)

            next_state, currentReward, done, _, _ = env.step(policyAction)

            rewardInVisitedState.append(int(currentReward))
            visitedStatesInEpisode.append(int(currentState))

            currentState = next_state
            if done:
                break

        rewardInVisitedState = np.array(rewardInVisitedState)
        visitedStatesInEpisode = np.array(visitedStatesInEpisode)
        numberofVisitedStates = len(visitedStatesInEpisode)

        Gt = 0.0

        for episode_iter in reversed(range(len(visitedStatesInEpisode))):
            # print(f"current epsiode {current_episode}")

            stateTmp = visitedStatesInEpisode[episode_iter]
            returnTmp = rewardInVisitedState[episode_iter]

            Gt = gamma * Gt + returnTmp

            if stateTmp not in visitedStatesInEpisode[:episode]:
                V[stateTmp] = V[stateTmp] + (alpha * (Gt - V[stateTmp]))

    return V
