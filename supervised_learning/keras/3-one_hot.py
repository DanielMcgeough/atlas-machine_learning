#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a vector into a
    one-hot matrix"""

    return K.utils.to_categorical(labels, num_classes=classes)
