#!/usr/bin/env python3
"""Class for a dataset"""
# facehugger is a monster from the Aliens franchise
import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """class for a dataset"""

    def __init__(self, batch_size, max_len):
        """initialization with boundaries in place"""

        self.max_len = max_len
        self.batch_size = batch_size

        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en", split="train", as_supervised=True
        )

        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation", as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = (
            self.data_train
            .map(self.tf_encode)
            .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                  tf.size(en) <= max_len))
            .cache()
            .shuffle(buffer_size=20000)
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.data_valid = (
            self.data_valid
            .map(self.tf_encode)
            .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                  tf.size(en) <= max_len))
            .padded_batch(batch_size, padded_shapes=([None], [None]))
        )

    def tokenize_dataset(self, data):
        """Documentation"""

        en_base = []
        pt_base = []

        for en, pt in data:
            en_base.append(en.numpy().decode("utf-8"))
            pt_base.append(pt.numpy().decode("utf-8"))

        def en_iterator():
            for en in en_base:
                yield en

        def pt_iterator():
            for pt in pt_base:
                yield pt

        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

        vocab_size = 2**13

        en_model_trained = tokenizer_en.train_new_from_iterator(
            text_iterator=en_iterator(), vocab_size=vocab_size
        )

        pt_model_trained = tokenizer_pt.train_new_from_iterator(
            text_iterator=pt_iterator(), vocab_size=vocab_size
        )

        # They need to be trained on the data passed in
        return pt_model_trained, en_model_trained

    def encode(self, pt, en):
        """encoding and decoding tokens passed to function"""

        pt_tokens = self.tokenizer_pt.encode(pt.numpy().decode("utf-8"))
        en_tokens = self.tokenizer_en.encode(en.numpy().decode("utf-8"))

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """encoding and decoding the output from the translate func"""
        pt_output, en_output = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_output.set_shape([None])
        en_output.set_shape([None])

        return pt_output, en_output
