#!/usr/bin/env python3
"""
Plays Atari's Breakout using a trained agent (keras-rl2).
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,  # For preprocessing Atari frames
    FrameStack,         # To stack multiple frames
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,           # Convolutional layer
    Flatten,          # Flatten layer
    Dense,            # Dense (fully connected) layer
    InputLayer,
)
from rl.agents import DQNAgent
from rl.memory import SequentialMemory  # Import, but won't be used, but may cause error if removed
from rl.policy import GreedyQPolicy  # Changed to GreedyQPolicy
from tensorflow import keras  # Import Keras
import time  # Import time for slowing down



def create_atari_environment(game_name="BreakoutNoFrameskip-v4", noop_max=30):
    """
    Creates and preprocesses the Atari environment for use with DQN.  This
    function is identical to the one in train.py.

    Args:
        game_name (str, optional): Name of the Atari game.
            Defaults to "BreakoutNoFrameskip-v4".
        noop_max (int, optional): Maximum number of "noop" actions at the
            start of each episode. Defaults to 30.

    Returns:
        gym.Env: The preprocessed Atari environment.
    """
    env = gym.make(game_name, render_mode="human")  # Use human render mode here
    env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=4)
    env = FrameStack(env, num_stack=4)
    return env



def create_dqn_model(input_shape, num_actions):
    """
    Creates the DQN model using Keras. This function is identical to the
    one in train.py.

    Args:
        input_shape (tuple): Shape of the input frames.
        num_actions (int): Number of possible actions.

    Returns:
        keras.Sequential: The DQN model.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu", data_format="channels_last"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu", data_format="channels_last"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", data_format="channels_last"))
    model.add(Flatten(data_format="channels_last"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_actions, activation="linear"))  # No activation for Q-values
    return model



def create_dqn_agent(model, num_actions, policy, weights_path=None):
    """
    Creates the DQN agent using keras-rl2.  This function is modified
    from train.py to load pre-trained weights.  The memory is not
    needed for playing, but is included to avoid errors.

    Args:
        model (keras.Sequential): The DQN model.
        num_actions (int): Number of possible actions.
        policy (rl.policy.Policy): The action selection policy.
        weights_path (str, optional): Path to the saved weights.
            Defaults to None.

    Returns:
        rl.agents.DQNAgent: The DQN agent.
    """
    agent = DQNAgent(
        model=model,
        nb_actions=num_actions,
        policy=policy,
        memory=SequentialMemory(limit=1000000, window_length=4),  # Dummy memory
        nb_steps_warmup=0,       # No warmup for playing
        target_model_update=1e-2, # won't be used, but needs a value
        gamma=0.99,            # won't be used, but needs a value
        train_interval=4,      # won't be used, but needs a value
        gradient_steps=1,        # won't be used, but needs a value
    )
    agent.compile(optimizer="adam", metrics=["mae"])  # Compile, but won't train
    if weights_path:
        agent.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    return agent



def play_agent(agent, env, num_episodes=5):
    """
    Plays the game using the trained agent.

    Args:
        agent (rl.agents.Agent): The trained agent.
        env (gym.Env): The Atari environment.
        num_episodes (int, optional): The number of episodes to play.
            Defaults to 5.
    """
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        episode_reward = 0
        print(f"Episode: {episode + 1}")
        while not done and not truncated:
            #env.render()  # Removed:  We set render_mode='human' in create_atari_environment
            action = agent.forward(state) # Use agent.forward for single action
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            time.sleep(0.02)  # Slow down for better visualization
        print(f"Episode {episode + 1} finished with reward: {episode_reward}")
    env.close()



if __name__ == "__main__":
    # 1. Create the Atari environment
    env = create_atari_environment()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Number of actions: {env.action_space.n}")

    # 2. Create the DQN model
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    model = create_dqn_model(input_shape, num_actions)
    model.summary()

    # 3. Create the policy (GreedyQPolicy for playing)
    policy = GreedyQPolicy()

    # 4. Create the DQN agent (and load weights)
    weights_path = "policy.h5"  # Path to the saved weights
    dqn_agent = create_dqn_agent(model, num_actions, policy, weights_path=weights_path)

    # 5. Play the game
    play_agent(dqn_agent, env, num_episodes=5)

    env.close()
