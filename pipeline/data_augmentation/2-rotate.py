#!/usr/bin/env python3
"""rotates an image in tensorflow"""
import tensorflow as tf


def rotate_image(image):
    """rotates an image in tf"""
    return tf.image.rot90(image, k=1)
