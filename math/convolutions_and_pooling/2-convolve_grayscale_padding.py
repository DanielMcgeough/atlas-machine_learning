#!/usr/bin/env python3
"""This is brutal"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale
    images with custom padding.

    Args:
        images:
    A numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images.
        kernel: A numpy.ndarray with shape
        (kh, kw) containing the kernel for
        the convolution.
        padding:
    A tuple (ph, pw) specifying the padding
    for the height and width of the image.

    Returns:
        A numpy.ndarray containing the
        convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Calculate the output shape
    output_shape = (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1)
    output = np.zeros(output_shape)

    # Perform the convolution
    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
