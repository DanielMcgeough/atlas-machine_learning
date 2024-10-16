#!/usr/bin/env python3
"""even in a job I wont stop this i bet."""
import tensorflow as tf

def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: A tensor containing the cost of the network without L2 regularization.   

        model: A Keras model with L2 regularization applied to all layers.

    Returns:
        A tensor containing the total cost for each layer of the network, accounting for L2 regularization.
    """

    l2_costs = []

    for lay in model.layers:
        if isinstance(lay, tf.keras.layers.Dense) and lay.kernel_regularizer:
            l2_cost = lay.kernel_regularizer(lay.kernel)
            l2_costs.append(l2_cost)

    return cost + tf.stack(l2_costs)
