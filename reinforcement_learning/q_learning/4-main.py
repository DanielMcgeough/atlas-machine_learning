#!/usr/bin/env python3

"""This function plays trained an agent """

import numpy as np


def play(env, Q, max_steps=100):
    """Plays out the trained agent
    env - the environment in which we play
    Q - the already trained Q-table
    max_steps - Boiyo is not allowed any more steps than this

    returns
    rewards - the toal rewards for the episode
    board_states - list of rendered outputs of the board state

    What exactly makes epsilon greedy though?

    Also, is this just a limited version of task 3???
    I'm so confused."""

    # Okay so first things first
    # We need vars for the returns
    # One of them holding each state I guess
    # I have no idea how to get the "rendered output" that they're asking for

    rendered_output = []
    total_reward = 0

    state = env.reset()[0]
    rendered_output.append(env.render())

    # ^^^ Not sure about this one but we'll try it
    terminated, truncated = False, False
    step_counter = 0

    while not terminated and not truncated:
        action = np.argmax(Q[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)

        step_counter += 1

        state = new_state

        rendered_output.append(env.render())

        if step_counter >= max_steps:
            break

    # if reward == 1:
        # total_reward = 1
    # I know that doesn't need to be that way but oh well

    env.close()

    return reward, rendered_output
