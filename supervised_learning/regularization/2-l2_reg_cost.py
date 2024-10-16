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

    regularization_cost = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.kernel_regularizer:
            regularization_cost = layer.kernel_regularizer(layer.kernel)
            regularization_cost.append(regularization_cost)

    return cost + tf.stack(regularization_cost)
