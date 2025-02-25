#!/usr/bin/env python3
"""brick squad"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder A
    convolutional encoder is a neural
    network component that uses convolutional
    layers to progressively extract and
    compress features from input data,
    like images. It shrinks the spatial
    dimensions while increasing the depth
    (number of feature maps), creating a
    compact, abstract representation of
    the input. Think of it as a series
    of filters that identify patterns and
    downsample the data.
        input_dims: tuple of int with dimensions of the model input
        filters: list of int filters for each convolutional layer
        latent_dims: tuple of int dimensions of the latent space representation
    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
 
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs
    for i in filters:
        x = keras.layers.Conv2D(
            i, (3, 3), activation='relu',
            padding='same')(x)
        x = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(x)
    encoder_outputs = x
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs
    for i in reversed(filters[1:]):
        x = keras.layers.Conv2D(
            i, (3, 3), activation='relu',
            padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu',
        padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid',
        padding='same')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    autoencoder_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, autoencoder_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
