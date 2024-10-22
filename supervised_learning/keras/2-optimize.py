#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimization which
    we know is one of the best gradient
    descent optimizations out there with
    categorical crossentropy loss whatever
    that is and accuracy metrics"""

    network.compile(loss="categorical_crossentropy", optimizer="adam")

    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )
    network.optimizer = optimizer

    return None
