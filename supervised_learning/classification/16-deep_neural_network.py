#!/usr/bin/env python3
"""Module for my first Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """establishes class for DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        """Initialises deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)

        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):
            if not isinstance(layers[i-1], int) or layers[i-1] <= 0:
                raise TypeError("layers must be a list of positive integers")

            layer_size = layers[i - 1]

            prev_layer_size = nx if i == 1 else layers[i - 2]

            self.weights['W' + str(i)] = (
                np.random.randn(layer_size, prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )

            self.weights['b' + str(i)] = np.zeros((layer_size, 1))
