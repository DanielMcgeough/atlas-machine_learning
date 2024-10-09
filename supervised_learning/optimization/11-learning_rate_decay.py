#!/usr/bin/env python3
"""shorter than I remember"""
import numpy as np

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy.

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine the rate at which alpha will decay.
        global_step: The number of passes of gradient descent that have elapsed.
        decay_step: The number of passes of gradient descent that should occur before alpha is decayed further.

    Returns: The updated value for alpha.
    """

    # Calculate the decay factor
    decay_factor = 1.0 / (1.0 + decay_rate * (global_step // decay_step))

    # Update the learning rate
    updated_alpha = alpha * decay_factor

    return updated_alpha
