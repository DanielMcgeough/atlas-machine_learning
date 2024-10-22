#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                verbose=True,
                shuffle=False):
    """Training a model with mini batch GD"""

    network.compile(loss='categorical_crossentropy',
                    optimizer=network.optimizer, metrics=['accuracy'])

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
