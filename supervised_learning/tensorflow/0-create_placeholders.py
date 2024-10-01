#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """returns placeholders named x and y for
    input data to the neural network and
    one-hot labels for input data respectivelys"""
    x = tf.placeholder(tf.float32, shape=[None, nx])
    y = tf.placeholder(tf.float32, shape=[None, classes])

    return x, y
