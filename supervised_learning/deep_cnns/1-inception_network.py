#!/usr/bin/env python3
"""hate this ish."""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
    """
    Builds the Inception network as described
    in Going Deeper with Convolutions (2014).

    Args:
        None

    Returns:

    The Keras model.
    """

    # Input Layer
    input_layer = K.Input(shape=(224, 224, 3))

    # Convolutional Layers
    x =  K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x =  K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x =  K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x =  K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception Blocks
    x = inception_block.inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block.inception_block(x, [128, 128, 192, 32, 96, 64])
    x =  K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_block.inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block.inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block.inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block.inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block.inception_block(x, [256, 160, 320, 32, 128, 128])
    x =  K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_block.inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block.inception_block(x, [384, 192, 384, 48, 128, 128])

    # Global Average Pooling
    x =  K.layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x =  K.layers.Dense(1000, activation='relu')(x)

    # Output Layer (Softmax)
    output =  K.layers.Dense(1000, activation='softmax')(x)

    # Create the model
    model = K.Model(inputs=input_layer, outputs=output)

    return model
