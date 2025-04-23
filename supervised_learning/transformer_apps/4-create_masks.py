#!/usr/bin/env python3
"""shebang for mask making"""
import tensorflow as tf


def create_masks(inputs, target):
    """create masks to cover everything tnut current word"""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = tf.shape(target)[1]
    lookahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    decoder_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_padding_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]

    combinded_mask = tf.maximum(decoder_padding_mask, lookahead_mask)

    decoder_mask = encoder_mask

    return encoder_mask, combinded_mask, decoder_mask
