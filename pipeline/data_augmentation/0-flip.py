#!/usr/bin/env python3
"""flips an image in tensorflow"""
import tensorflow as tf


def flip_image(image):
    """flips image in tf"""
    return tf.image.flip_left_right(image)
