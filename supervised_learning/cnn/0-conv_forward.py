#!/usr/bin/env python3
"""Cool but complex"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation for a convolutional layer.

    Args:
        A_prev: Output activations of the previous layer (m, h_prev, w_prev, c_prev)
        W: Weights (filters) (kh, kw, c_prev, c_new)
        b: Biases (1, 1, 1, c_new)
        activation: Activation function
        padding: 'same' or 'valid'
        stride: Tuple (sh, sw)

    Returns:
        Output activations of the current layer
    """

    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, _, c_new) = W.shape
    (sh, sw) = stride

    # Calculate output dimensions
    h_new = int(1 + (h_prev + 2 * padding - kh) / sh)
    w_new = int(1 + (w_prev + 2 * padding - kw) / sw)

    # Pad the input if 'same' padding
    if padding == 'same':
        A_prev = np.pad(A_prev, ((0, 0), (kh//2, kh//2), (kw//2, kw//2), (0, 0)), 'constant')

    # Initialize output array
    Z = np.zeros((m, h_new, w_new, c_new))

    # Perform convolution
    for h in range(h_new):
        for w in range(w_new):
            for c in range(c_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

        Z[:, h, w, c] = np.sum(A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] * W[:, :, :, c], axis=(1, 2, 3)) + b[:, :, :, c]

    # Apply activation function
    A = activation(Z)

    return A
