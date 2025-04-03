#!/usr/bin/env python3
"""module for fastext"""
from gensim.models import FastText

def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim FastText model.

    Args:
        sentences (list): A list of sentences to be trained on.
        vector_size (int, optional): The dimensionality of the embedding layer. Defaults to 100.
        min_count (int, optional): The minimum number of occurrences of a word for use in training. Defaults to 5.
        negative (int, optional): The size of negative sampling. Defaults to 5.
        window (int, optional): The maximum distance between the current and predicted word within a sentence. Defaults to 5.
        cbow (bool, optional): A boolean to determine the training type; True is for CBOW; False is for Skip-gram. Defaults to True.
        epochs (int, optional): The number of iterations to train over. Defaults to 5.
        seed (int, optional): The seed for the random number generator. Defaults to 0.
        workers (int, optional): The number of worker threads to train the model. Defaults to 1.

    Returns:
        gensim.models.fasttext.FastText: The trained FastText model.
    """

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=int(not cbow),  # sg=0 for CBOW, sg=1 for Skip-gram
        epochs=epochs,
        seed=seed,
        workers=workers,
    )

    return model

# Example Usage (demonstration):
if __name__ == "__main__":
    sentences = [
        ["this", "is", "the", "first", "sentence"],
        ["this", "is", "the", "second", "sentence"],
        ["and", "this", "is", "the", "third", "sentence"],
        ["another", "sentence", "here"]
    ]

    model = fasttext_model(sentences, vector_size=50, epochs=10)

    # Example of accessing word vectors:
    print(model.wv["sentence"])
    print(model.wv.most_similar("sentence"))
