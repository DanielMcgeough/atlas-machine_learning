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
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    assert c == kc, "Image channels and kernel channels must match"

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2

    elif padding == 'valid':
        ph, pw = 0, 0

    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                    * kernels[..., k],
                    axis=(1, 2, 3)
                )

    return output
