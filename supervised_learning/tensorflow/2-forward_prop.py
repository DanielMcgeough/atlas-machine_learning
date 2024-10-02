#!/usr/bin/env python3
"""for tensorflow without Keras... explicitly"""
import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that allows for forward propagation"""
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
