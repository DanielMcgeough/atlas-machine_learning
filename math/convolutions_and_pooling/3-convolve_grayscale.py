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

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2)
            )

    return output
