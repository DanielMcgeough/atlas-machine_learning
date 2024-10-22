#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format.

    Args:
        network: The model whose configuration should be saved.
        filename: The path of the file that the configuration should be   
     saved to.
    """

    config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(network.to_json())


def load_config(filename):
    """Loads a model with a specific configuration.

    Args:
        filename: The path of the file containing the model's configuration in JSON format.

    Returns:
        The loaded model.
    """

    with open(filename, 'r') as json_file:

        return K.models.model_from_json(json_file.read())
