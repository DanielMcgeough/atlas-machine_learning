#!/usr/bin/env python3
"""This is brutal"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images
    using multiple kernels.

    Args:
        images:
    A numpy.ndarray with shape (m, h, w, c)
    containing multiple images.
    kernels: A numpy.ndarray with shape
    (kh, kw, c, nc) containing the kernels
    for the convolution.
    padding: Either 'same', 'valid', or a
    tuple (ph, pw) specifying the padding.
    stride: A tuple (sh, sw) specifying
    the stride for the height and width.

    Returns:
        A numpy.ndarray containing the convolved images.
    """

    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Calculate padding based on padding mode
    if padding == 'same':
        ph = (h * sh - h + kh - 1) // 2
        pw = (w * sw - w + kw - 1) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    # Pad the images
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Calculate the output shape
    output_shape = (m, (h + 2*ph - kh + 1) // sh, (w + 2*pw - kw + 1) // sw, nc)
    output = np.zeros(output_shape)

    # Perform the convolution
    for i in range(0, output_shape[1], sh):
        for j in range(0, output_shape[2], sw):
            for k in range(nc):
                output[:, i, j, k] = np.sum(images[:, i:i+kh, j:j+kw, :] * kernels[:, :, :, k], axis=(1, 2, 3))

    return output
