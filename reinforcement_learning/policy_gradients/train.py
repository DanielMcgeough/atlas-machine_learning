#!/usr/bin/env python3

"""This is where we're going to train table
so that it becomes less stupid in a more optimal set up.
Honestly I'm thinking of just following this tutorial
and saying F it? I'm not sure how I'll impliment the policy gradient.
https://www.janisklaise.com/post/rl-policy-gradients/
That tutorial proved to be relatively useless.
https://www.youtube.com/watch?v=5eSh5F8gjWU
This one seems to be more useful.
Oh???? This stack overflow looks promising
https://stackoverflow.com/questions/46597809/policy-gradient-methods-for-open-ai-gym-cartpole
Nevermind ignore that one
https://github.com/drozzy/reinforce/blob/main/reinforce.py
Ok here's that youtuber's github.

I'll be honest Sajid, I circled around this task quite a bit
I have a vague understanding of the concepts, but I think I have to take the L on this task
You can see a previous attempt under "dead.py"
These projects were pretty dense, I regret procrastinating
But I think I did fairly well all things considered.
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """I'm *pretty* sure that we're not training a full model here
    Just filling out the Q table
    Unless it's not the Q table anymore?
    env - initial environment
    nb_episodes - the number of episodes
    alpha - learning rate
    gamma - discount factor"""

    # Initializing weights
    # Or trying to rather
    weights = np.random.rand(4)
    episode_rewards = []

    for i in range(nb_episodes):
        done = False

        actions = []
        states = []
        rewards = []

        # while not done:

    return
