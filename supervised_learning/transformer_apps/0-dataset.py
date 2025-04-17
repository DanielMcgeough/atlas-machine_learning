#!/usr/bin/env python3
"""
This module defines the Dataset class for loading and preprocessing a dataset
for machine translation.
"""
import tensorflow_datasets as tfds
import transformers

class Dataset:
    """
    Loads and preprocesses a dataset for machine translation.
    """

    def __init__(self):
        """
        Class constructor.
        Creates the instance attributes:
            data_train: tf.data.Dataset containing the train split,
                        loaded as_supervised.
            data_valid: tf.data.Dataset containing the validate split,
                        loaded as_supervised.
            tokenizer_pt: Portuguese tokenizer created from the training set.
            tokenizer_en: English tokenizer created from the training set.
        """
        # Load the training and validation data.
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        # Create the tokenizers.
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple (pt, en).
                pt: tf.Tensor containing the Portuguese sentence.
                en: tf.Tensor containing the corresponding English sentence.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        # Use pre-trained tokenizers.
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            max_len=128,  # Add max_len for consistency
            do_lower_case=False,  # Important for this specific model
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            max_len=128, # Add max_len for consistency
        )

        def tokenize_fn(pt, en):
            """
            Tokenizes the input Portuguese and English sentences.

            Args:
                pt: tf.Tensor containing the Portuguese sentence.
                en: tf.Tensor containing the English sentence.

            Returns:
                A tuple containing the tokenized Portuguese and English sentences
                as Python lists of integers.
            """
            pt_tokens = tokenizer_pt.encode(pt.numpy().decode('utf-8'))
            en_tokens = tokenizer_en.encode(en.numpy().decode('utf-8'))
            return (
                pt_tokens,
                en_tokens,
            )

        # Map the tokenization function
        tokenized_data = data.map(
            lambda pt, en: tokenize_fn(pt, en),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Train the tokenizers with a maximum vocabulary size of 2**13.
        tokenizer_pt.train_new_from_iterator(
            (text.numpy().decode('utf-8') for text, _ in self.data_train),
            vocab_size=2**13
        )
        tokenizer_en.train_new_from_iterator(
            (text.numpy().decode('utf-8') for _, text in self.data_train),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
