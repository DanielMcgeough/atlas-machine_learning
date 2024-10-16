#!/usr/bin/env python3
""" should this maybe be more professional """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: A tensor containing the cost of the network without L2 regularization.   

        model: A Keras model with L2 regularization applied to all layers.

    Returns:
        A tensor containing the total cost for each layer of the network, accounting for L2 regularization.
    """

    regularization_cost = 0
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            regularization_cost += tf.reduce_sum(layer.kernel_regularizer(layer.kernel))

    total_cost = cost + regularization_cost

    return total_cost
