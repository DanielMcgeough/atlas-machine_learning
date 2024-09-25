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

        self.__L = len(layers)

        self.__cache = {}
        self.__weights = {}

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

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculate forward propagation of a deep neural network"""
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]

            z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-z))

            self.__cache[f'A{i}'] = A

        return A, self.__cache
