#!/usr/bin/env python3
"""I would rather dig ditches than muddle
my way through this"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images: A numpy.ndarray with shape
        (m, h, w, c) containing multiple
        images.
        kernel_shape: A tuple (kh, kw)
        containing the kernel shape for
        the pooling.
        stride: A tuple (sh, sw) specifying
        the stride for the height and width.
        mode: The pooling mode, either
        'max' or 'avg'.

  Returns:
    A numpy.ndarray containing the pooled
    images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )

    return output
