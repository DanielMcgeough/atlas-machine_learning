#!/usr/bin/env python3
"""
Trains a DQN agent to play Atari's Breakout using keras-rl2.
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,  # For preprocessing Atari frames
    FrameStack,         # To stack multiple frames
    #    Monitor            # For recording videos (optional, for later)
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,           # Convolutional layer
    Flatten,          # Flatten layer
    Dense,            # Dense (fully connected) layer
    InputLayer,       # Input Layer
)
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from tensorflow import keras  # Import Keras


def create_atari_environment(game_name="BreakoutNoFrameskip-v4", noop_max=30):
    """
    Creates and preprocesses the Atari environment for use with DQN.

    Args:
        game_name (str, optional): Name of the Atari game.
            Defaults to "BreakoutNoFrameskip-v4".
        noop_max (int, optional): Maximum number of "noop" actions at the
            start of each episode. Defaults to 30.

    Returns:
        gym.Env: The preprocessed Atari environment.
    """
    env = gym.make(game_name, render_mode="rgb_array")  # Use rgb_array for Keras-RL
    env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=4)
    env = FrameStack(env, num_stack=4)
    # env = Monitor(env, directory="training_runs", force=True)  # Optional: for video
    return env



def create_dqn_model(input_shape, num_actions):
    """
    Creates the DQN model using Keras.

    Args:
        input_shape (tuple): Shape of the input frames.
        num_actions (int): Number of possible actions.

    Returns:
        keras.Sequential: The DQN model.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))  # Explicit InputLayer
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu", data_format="channels_last"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu", data_format="channels_last"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", data_format="channels_last"))
    model.add(Flatten(data_format="channels_last"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_actions, activation="linear"))  # No activation for Q-values
    return model



def create_dqn_agent(model, num_actions, memory, policy):
    """
    Creates the DQN agent using keras-rl2.

    Args:
        model (keras.Sequential): The DQN model.
        num_actions (int): Number of possible actions.
        memory (rl.memory.Memory): The memory buffer.
        policy (rl.policy.Policy): The action selection policy.

    Returns:
        rl.agents.DQNAgent: The DQN agent.
    """
    agent = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=memory,
        policy=policy,
        nb_steps_warmup=10000,  # From the Breakout example
        target_model_update=1e-2,
        gamma=0.99,
        train_interval=4,
        gradient_steps=1,
    )
    agent.compile(Adam(learning_rate=1e-4), metrics=["mae"])  # Using Adam
    return agent



def train_dqn_agent(agent, env, num_episodes=10000):
    """
    Trains the DQN agent in the given environment.

    Args:
        agent (rl.agents.DQNAgent): The DQN agent.
        env (gym.Env): The Atari environment.
        num_episodes (int, optional): Number of episodes to train for.
            Defaults to 10000.

    Returns:
        rl.agents.DQNAgent: The trained DQN agent.
    """
    agent.fit(env, nb_steps=num_episodes * 4000, visualize=False, verbose=1)
    return agent



def save_policy_network(agent, filepath="policy.h5"):
    """
    Saves the policy network weights to a file.

    Args:
        agent (rl.agents.DQNAgent): The trained DQN agent.
        filepath (str, optional): Path to save the weights.
            Defaults to "policy.h5".
    """
    agent.save_weights(filepath, overwrite=True)



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

    # 3. Create the memory and policy
    memory = SequentialMemory(limit=1000000, window_length=4)  # From Breakout example
    policy = EpsGreedyQPolicy(eps=1.0, eps_decay=0.9999, min_eps=0.1)

    # 4. Create the DQN agent
    dqn_agent = create_dqn_agent(model, num_actions, memory, policy)

    # 5. Train the agent
    trained_agent = train_dqn_agent(dqn_agent, env, num_episodes=1000)  # Reduced for testing

    # 6. Save the policy network
    save_policy_network(trained_agent)

    print("Training complete. Policy network saved to policy.h5")
    env.close()
