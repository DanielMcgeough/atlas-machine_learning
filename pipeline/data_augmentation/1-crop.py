#!/usr/bin/env python3
"""crops an image in tensorflow"""
import tensorflow as tf


def crop_image(image, size):
    """crops image in tf"""
    crop_size = (size[0], size[1], image.shape[-1])
    return tf.image.random_crop(image, size=crop_size)
