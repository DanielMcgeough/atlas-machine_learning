#!/usr/bin/env python3
"""Getting ugly wid it"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp
    optimization algorithm.


    Args:
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.
        var: A numpy.ndarray containing the variable to be updated.
        grad: A numpy.ndarray containing the gradient of var.
        s: The previous second moment of var.

    Returns:
        The updated variable and the new moment, respectively.

    """

    # Update the second moment
    s = beta2 * s + (1 - beta2) * grad**2

    # Calculate the denominator for the update
    denominator = np.sqrt(s) + epsilon

    # Update the variable
    var = var - alpha * grad / denominator

    return var, s
