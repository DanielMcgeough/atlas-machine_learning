#!/usr/bin/env python3
"""Cool but complex"""
import numpy as np

def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
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

    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    # Calculate output dimensions
    h_new = int(1 + (h_prev - kh) / sh)
    w_new = int(1 + (w_prev - kw) / sw)

    # Initialize output array
    A = np.zeros((m, h_new, w_new, c_prev))

    for h in range(h_new):
        for w in range(w_new):
            for c in range(c_prev):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

        if mode == 'max':
          A[:, h, w, c] = np.max(A_prev[:, vert_start:vert_end, horiz_start:horiz_end, c], axis=(1, 2))
        elif mode == 'avg':
          A[:, h, w, c] = np.mean(A_prev[:, vert_start:vert_end, horiz_start:horiz_end, c], axis=(1, 2))

    return A
