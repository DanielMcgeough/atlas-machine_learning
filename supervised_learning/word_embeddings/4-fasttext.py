#!/usr/bin/env python3
"""module for fasttext"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                    window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Function that creates, builds and trains a gensim fastText model
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: min number of occurrences of a word for use in training
        window: max distance between current and predicted word within sentence
        negative: size of negative sampling
        cbow: a boolean to determine the training type;
            True is for CBOW
            False is for Skip-gram
        epochs: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model

        Returns: the trained model
        FastText extends Word2Vec by representing
        words as character n-grams, enabling it to
        handle out-of-vocabulary words and capture
        subword information. It can use either CBOW
        or Skip-gram architectures, like Word2Vec,
        but its character n-gram approach makes it
        particularly effective for morphologically rich languages.
    """
    # Set training algorithm for cbow
    # Because it's CBOW sg = 0, otherwise it would be sg = 1
    sg = 0 if cbow else 1

    # fastText model with specific parameters
    model = FastText(
        sentences=sentences, #Pass sentences during initialization.
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs
    )

    return model
