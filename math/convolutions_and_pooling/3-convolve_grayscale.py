#!/usr/bin/env python3
"""This is brutal"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale
    images.

    Args:
        images:
    A numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images.
    kernel: A numpy.ndarray with shape
    (kh, kw) containing the kernel for
    the convolution.
    padding: Either 'same', 'valid', or a
    tuple (ph, pw) specifying the padding.
    stride: A tuple (sh, sw) specifying
    the stride for the height and width.

    Returns:
    A numpy.ndarray containing the
    convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Calculate padding based on padding mode
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (kh % 2 == 0)
        pw = ((w - 1) * sw + kw - w) // 2 + (kw % 2 == 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad the images
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Calculate the output shape
    output_shape = (m, (h + 2*ph - kh + 1) // sh,
                    (w + 2*pw - kw + 1) // sw)
    output = np.zeros(output_shape)

    # Perform the convolution
    for i in range(0, output_shape[1], sh):
        for j in range(0, output_shape[2], sw):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
