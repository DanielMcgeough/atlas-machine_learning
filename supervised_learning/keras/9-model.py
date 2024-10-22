#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def save_model(network, filename):
    """does what it says"""
    network.save(filename)


def load_model(filename):
    """loads an entire model"""
    model = K.models.load_model(filename)
    return model
