#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Saves a model's weights.

    Args:
        network: The model whose weights should be saved.
        filename: The path of the file that the weights should be saved to.
        save_format:
        The format in which the weights should be saved.
    """

    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Loads a model's weights.

    Args:
        network: The model to which the weights should be loaded.
        filename: The path of the file that the weights should be loaded
        from.
    """

    network.load_weights(filename)
