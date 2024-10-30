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

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev + 1) // 2

    elif padding == 'valid':
        ph, pw = 0, 0

    else:
        ph, pw = padding

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output_h = (h_prev + 2 * ph - kh) // sh + 1
    output_w = (w_prev + 2 * pw - kw) // sw + 1
    Z = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                location = A_prev_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

                Z[:, i, j, k] = np.sum(location * W[..., k],
                                       axis=(1, 2, 3)) + b[..., k]

    A = activation(Z)

    return A
