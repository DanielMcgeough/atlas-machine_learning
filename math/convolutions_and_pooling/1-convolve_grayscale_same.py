#!/usr/bin/env python3
"""This is brutal"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
    images:
        A numpy.ndarray with shape (m, h, w)
        containing multiple grayscale images.
        kernel: A numpy.ndarray with shape
        (kh, kw) containing the kernel for
        the convolution.


    Returns:
        A numpy.ndarray containing the
        convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    output = np.zeros((m, h, w))

    padded = np.pad(images,
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant')

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
