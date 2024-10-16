#!/usr/bin/env python3
"""even in a job I wont stop this i bet."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    Args:
        X: A numpy.ndarray of shape (nx, m) containing the input data.
        weights: A dictionary of the weights and biases of the neural network.
        L: The number of layers in the network.
        keep_prob: The probability that a node will be kept.

    Returns:
        A dictionary containing the outputs of each layer
        and the dropout mask used on each layer.
    """

    # cache = {}
    # D = {}

    # # Input layer
    # A0 = X
    # cache["A0"] = A0

    # # Hidden layers
    # for l in range(1, L):
    #     Z = np.dot(weights["W" + str(l)], A0) + weights["b" + str(l)]
    #     A = np.tanh(Z)

    #     # Dropout
    #     D["D" + str(l)] = np.random.rand(*A.shape) < keep_prob
    #     A *= D["D" + str(l)]
    #     A /= keep_prob

    #     cache["A" + str(l)] = A
    #     A0 = A

    # # Output layer
    # ZL = np.dot(weights["W" + str(L)], A0) + weights["b" + str(L)]
    # AL = np.exp(ZL) / np.sum(np.exp(ZL), axis=0)

    # cache["AL"] = AL

    # return cache, D
    cache = {}
    cache['A0'] = X
    A = X

    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        # Calculate current layer output
        z = np.matmul(W, A) + b

        if layer == L:
            # Softmax activation for output layer
            t = np.exp(z)
            cache[f'A{layer}'] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # tanh activation for hidden layers
            A = np.tanh(z)

            # Apply dropout
            drop = np.random.rand(* A.shape) < keep_prob
            A *= drop
            A /= keep_prob

            cache[f'A{layer}'] = A
            cache[f'D{layer}'] = drop.astype(int)

    return cache
