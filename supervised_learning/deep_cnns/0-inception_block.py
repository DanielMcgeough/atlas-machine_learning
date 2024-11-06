#!/usr/bin/env python3
"""document is cool, inception is about
using multiple kernels of different sizes
during training and combining the results"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an Inception block.

    Args:
        A_prev: The output from the previous layer.
        filters: A tuple or list containing
        F1, F3R, F3, F5R, F5, FPP, respectively.

    Returns:
        The concatenated output of the
        inception block.
    """

    # 1x1 Convolution Branch
    conv1_1x1 = K.layers.Conv2D(filters[0],
                                (1,1),
                                activation='relu',
                                padding='same')(A_prev)

    # 3x3 Convolution Branch
    conv3_3x3 = K.layers.Conv2D(filters[1],
                                (1,1),
                                activation='relu',
                                padding='same')(A_prev)
    conv3_3x3 = K.layers.Conv2D(filters[2],
                                (3,3),
                                activation='relu',
                                padding='same')(conv3_3x3)

    # 5x5 Convolution Branch
    conv5_5x5 = K.layers.Conv2D(filters[3],
                                (1,1),
                                activation='relu',
                                padding='same')(A_prev)
    conv5_5x5 = K.layers.Conv2D(filters[4],
                                (5,5),
                                activation='relu',
                                padding='same')(conv5_5x5)

    # Pooling Branch
    pool = K.layers.MaxPooling2D((3,3),
                                 strides=(1,1),
                                 padding='same')(A_prev)
    pool = K.layers.Conv2D(filters[5],
                           (1,1),
                           activation='relu',
                           padding='same')(pool)

    # Concatenate the outputs of the 4 branches
    output = K.layers.Concatenate()([conv1_1x1,
                                     conv3_3x3,
                                     conv5_5x5,
                                     pool])

    return output
