#!/usr/bin/env python3
"""Sometimes I hate this"""
import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation for a pooling layer.

    Args:
        dA: Gradient of the cost with respect
        to the output of the pooling layer
        (m, h_new, w_new, c_new)
        A_prev: Output activations of the
        previous layer (m, h_prev, w_prev,
        c_prev)
        kernel_shape: Tuple (kh, kw)
        specifying the kernel size
        stride: Tuple (sh, sw) specifying the
        stride
        mode: 'max' or 'avg', indicating the
        pooling mode

    Returns:
        dA_prev: Gradient of the cost with
        respect to the activations of the
        previous layer
    """

    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (m, h_new, w_new, c_new) = dA.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    dA_prev = np.zeros_like(A_prev)


    for i in range(h_new):
        for j in range(w_new):
            for c in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

        if mode == 'max':
          # Find the index of the maximum value in the pool region
          max_idx = np.argmax(A_prev[:, vert_start:vert_end, horiz_start:horiz_end, c], axis=(1, 2))
          # Set the gradient at the maximum position
          for k in range(m):
            dA_prev[k, vert_start:vert_end, horiz_start:horiz_end, c][k, max_idx[k, 0], max_idx[k, 1]] = dA[k, i, j, c]
        elif mode == 'avg':
          dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, c] += dA[:, i, j, c][:, np.newaxis, np.newaxis] / (kh * kw)

    return dA_prev
