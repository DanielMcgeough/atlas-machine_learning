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

    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for c in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_slice = A_prev[i, vert_start:vert_end,
                                         horiz_start:horiz_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += \
                            mask * dA[i, j, k, c]

                    elif mode == 'avg':
                        dA_curr = dA[i, j, k, c]
                        dA_avg = dA_curr / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += dA_avg

    return dA_prev
