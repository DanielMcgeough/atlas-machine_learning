#!/usr/bin/env python3
"""Sometimes I hate this"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation for a
    convolutional layer.

    Args:
        dZ: Gradient of the cost with respect
        to the output of the current layer
        (m, h_new, w_new, c_new)
        A_prev: Output activations of the
        previous layer (m, h_prev, w_prev,
        c_prev)
        W: Weights (filters) (kh, kw,
        c_prev, c_new)
        b: Biases (1, 1, 1, c_new)
        padding: 'same' or 'valid'
        stride: Tuple (sh, sw)

    Returns:
        dA_prev: Gradient of the cost with
        respect to the activations of the
        previous layer
        dW: Gradient of the cost with respect
        to the weights
        db: Gradient of the cost with respect
        to the biases
  """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
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
    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for c in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw
                    A_slice = A_prev_padded[i, vert_start:vert_end,
                                            horiz_start:horiz_end, :]
                    dA_prev_padded[i, vert_start:vert_end,
                                   horiz_start:horiz_end, :] += \
                        W[:, :, :, c] * dZ[i, j, k, c]
                    dW[:, :, :, c] += A_slice * dZ[i, j, k, c]
    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:-ph or None, pw:-pw or None, :]
    else:
        dA_prev = dA_prev_padded[:, :h_prev, :w_prev, :]

    return dA_prev, dW, db
