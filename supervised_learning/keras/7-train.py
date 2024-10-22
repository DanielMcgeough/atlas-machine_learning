#!/usr/bin/env python3
"""Keras yay"""

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                verbose=True,
                shuffle=False):
    """This is the same as 4 in that
    it is training a model for mini batch
    GD and adds a validation data
    parameter to analyze. Now we are
    also adding an early stop functionality.
    New function is a learning rate decay
    feature to determine whether or not
    it should be used"""

    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
        ))

    if learning_rate_decay and validation_data:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearingRateScheduler(lr_schedule, verbose=1))

    K.backend.set_value(network.optimizer.learning_rate, alpha)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
