#!/usr/bin/env python3
"""Cool but complex"""
import numpy as np


def pool_forward(A_prev,
                 kernel_shape,
                 stride=(1, 1),
                 mode='max'):
    """
    Performs forward propagation for a
    pooling layer.

    Args:
        A_prev: Output activations of the
        previous layer (m, h_prev, w_prev,
        c_prev)
        kernel_shape: Tuple (kh, kw)
        specifying the kernel size
        stride: Tuple (sh, sw) specifying
        the stride
        mode: 'max' or 'avg', indicating
        the pooling mode

    Returns:
        Output activations of the pooling
        layer
  """

    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )

    return output
