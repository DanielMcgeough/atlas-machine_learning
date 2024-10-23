#!/usr/bin/env python3
"""This is brutal"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    Args:
        images:
    A numpy.ndarray with shape (m, h, w) containing multiple grayscale images.
        kernel: A numpy.ndarray with shape (kh, kw) containing the kernel for the convolution.   


    Returns:
        A numpy.ndarray containing the convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape


    output_shape = (m, h - kh + 1, w - kw + 1)
    output = np.zeros(output_shape)

    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
