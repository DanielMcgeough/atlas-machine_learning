#!/usr/bin/env python3
"""changes contrast in an image in tensorflow"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """adjust contrast in tf"""
    contrast_factor = tf.random.uniform([], lower, upper)

    adjusted_image = tf.image.adjust_contrast(image, contrast_factor)

    return adjusted_image
