#!/usr/bin/env python3
"""changes brightness in an image in tensorflow"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """change brightness"""
    return tf.image.random_brightness(image, max_delta)
