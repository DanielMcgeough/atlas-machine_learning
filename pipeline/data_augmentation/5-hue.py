#!/usr/bin/env python3
"""changes hue in an image in tensorflow"""
import tensorflow as tf


def change_hue(image, delta):
    """changes hue"""
    return tf.image.adjust_hue(image, delta)
