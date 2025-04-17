#!/usr/bin/env python3
"""
This module defines the Dataset class for loading and preprocessing a dataset
for machine translation.
"""
import transformers
import tensorflow_datasets as tfds


class Dataset:
    """class for a dataset"""

    def __init__(self):
        """initialize"""

        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en", split="train", as_supervised=True
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation", as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenize the dataset to prepare it for translation word by word"""

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
