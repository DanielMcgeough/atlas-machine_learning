#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: The cost of the network without L2 regularization.
        lambtha: The regularization parameter.
        weights: A dictionary of
        the weights and biases of the neural network.
        L: The number of layers in the neural network.
        m: The number of data points used.

    Returns:
        The cost
        of the network accounting for L2 regularization.   

    """

    regularization_cost = 0
    for l in range(1, L + 1):
        regularization_cost += np.sum(np.square(weights["W" + str(l)]))

    regularization_cost *= lambtha / (2 * m)
    regularized_cost = cost + regularization_cost

    return regularized_cost
