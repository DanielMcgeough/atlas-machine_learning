#!/usr/bin/env python3
"""even in a job I wont stop this i bet."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases of a neural network   
 using gradient descent with Dropout.

    Args:
        Y: A one-hot numpy.ndarray of shape (classes, m) containing the correct labels.
        weights: A dictionary of the weights and biases of the neural network.
        cache: A dictionary of the outputs and dropout masks of each layer.   

        alpha: The learning rate.
        keep_prob: The probability that a node will be kept.
        L: The number of layers in the network.
    """

    m = Y.shape[1]

    # Backward propagation
    for l in reversed(range(L)):
        if l == L - 1:
            dA_prev = cache["A" + str(l + 1)] - Y
        else:
            dA_prev = np.dot(weights["W" + str(l + 1)].T, dA_prev) * (1 - np.square(cache["A" + str(l)]))

        # Apply dropout mask
        dA_prev *= cache["D" + str(l)]
        dA_prev /= keep_prob

        dW = (1 / m) * np.dot(dA_prev, cache["A" + str(l - 1)].T)
        db = (1 / m) * np.sum(dA_prev, axis=1, keepdims=True)

        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db
