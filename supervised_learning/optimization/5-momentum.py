#!/usr/bin/env python3
"""Stupid stuff making me"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient
    descent with momentum optimization
    algorithm.


    Args:
        alpha: The learning rate.
        beta1: The momentum weight.
        var: A numpy.ndarray containing the variable to be updated.
        grad: A numpy.ndarray containing the gradient of var.
        v: The previous first moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """

    # Update the first moment
    v = beta1 * v + (1 - beta1) * grad

    # Update the variable
    var = var - alpha * v

    return var, v
