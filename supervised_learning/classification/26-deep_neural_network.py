#!/usr/bin/env python3
"""Module for my first Deep Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def cost(self, Y, A):
        """Calculate the cost of model using logical regression"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural networks prediction"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent on one pass of neural network"""
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache[f'A{i-1}']

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            dA_prev = np.dot(self.__weights[f'W{i}'].T, dZ)
            dZ = dA_prev * A_prev * (1 - A_prev)

            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

    def train(self, X, Y, iterations=2000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be a positive integer " +
                                 "less than or equal to iterations")

        graph_matrix = [[],[]]
        for i in range(iterations + 1):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {self.cost(Y, A)}")
                if graph:
                    graph_matrix[0].append(i)
                    graph_matrix[1].append(self.cost(Y, A))

        if graph:
            plt.plot(graph_matrix[0], graph_matrix[1])
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file.
        Its in byte stream and is called
        pickling. Tried so hard to find what the issue is, I just don't"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork
        object. Weird it uses JSON syntax
        but is completely byte stream."""

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
