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

    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    images = np.pad(images, ((0, 0), (pad_h, pad_h),
                             (pad_w, pad_w)), mode='constant')

    output_shape = (m, h, w)
    output = np.zeros(output_shape)

    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
