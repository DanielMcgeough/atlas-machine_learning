#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases of a neural network using gradient descent with L2 regularization.

    Args:

        Y: A one-hot numpy.ndarray of shape (classes, m) containing the correct labels.
        weights: A dictionary of the weights and biases of the neural network.
        cache: A dictionary of the outputs of each layer of the neural network.
        alpha: The learning rate.
        lambtha: The L2 regularization parameter.
        L: The number of layers of the network.

    """

    m = Y.shape[1]

    # Backward propagation
    for i in reversed(range(L)):
        if i == L - 1:
            dA_prev = cache["A" + str(i + 1)] - Y
    else:
        dA_prev = np.dot(weights["W" + str(i + 1)].T, dA_prev) * (1 - np.square(cache["A" + str(i)]))

    dW = (1 / m) * np.dot(dA_prev, cache["A" + str(i - 1)].T) + (lambtha / m) * weights["W" + str(i)]
    db = (1 / m) * np.sum(dA_prev, axis=1, keepdims=True)

    weights["W" + str(i)] -= alpha * dW
    weights["b" + str(i)] -= alpha * db
