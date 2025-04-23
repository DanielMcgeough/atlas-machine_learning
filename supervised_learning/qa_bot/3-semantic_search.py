#!/usr/bin/env python3
"""
This module defines a function for performing semantic search on a corpus of documents.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): The path to the corpus of reference documents.
        sentence (str): The sentence from which to perform semantic search.

    Returns:
        str: The reference text of the document most similar to the sentence.
             Returns None if no documents are found or an error occurs.
    """
    # Load a pre-trained sentence encoder model (e.g., from TensorFlow Hub)
    # Using a Universal Sentence Encoder
    embed_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    try:
        embed_model = hub.load(embed_model_url)
        print(f"Sentence encoder model loaded from TF Hub: {embed_model_url}")
    except Exception as e:
        print(f"Error loading sentence encoder model: {e}")
        return None

    # Read documents from the corpus path
    corpus_documents = []
    try:
        if os.path.isdir(corpus_path):
            for filename in os.listdir(corpus_path):
                filepath = os.path.join(corpus_path, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        corpus_documents.append(f.read())
        else:  # Assume it's a single file
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_documents.append(f.read())

        if not corpus_documents:
            print(f"No documents found in corpus path: {corpus_path}")
            return None

    except Exception as e:
        print(f"Error reading corpus documents: {e}")
        return None

    # Encode the input sentence and corpus documents
    try:
        embeddings = embed_model([sentence] + corpus_documents)
        sentence_embedding = embeddings[0]
        corpus_embeddings = embeddings[1:]
    except Exception as e:
        print(f"Error encoding sentences: {e}")
        return None

    # Calculate cosine similarity between the input sentence and corpus documents
    try:
        similarities = tf.keras.metrics.CosineSimilarity()(
            sentence_embedding, corpus_embeddings
        )
        best_match_index = np.argmax(similarities)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None

    # Return the most similar document
    return corpus_documents[best_match_index]



if __name__ == '__main__':
    # Create a dummy corpus for testing
    corpus_path = "dummy_corpus"
    os.makedirs(corpus_path, exist_ok=True)
    with open(os.path.join(corpus_path, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("This is the first document. It talks about cats.")
    with open(os.path.join(corpus_path, "doc2.txt"), "w", encoding="utf-8") as f:
        f.write("The second document discusses dogs and their habits.")
    with open(os.path.join(corpus_path, "doc3.txt"), "w", encoding="utf-8") as f:
        f.write("Document three mentions both cats and dogs.")

    search_sentence = "Tell me about cats."
    most_similar_document = semantic_search(corpus_path, search_sentence)

    if most_similar_document:
        print(f"Search Sentence: {search_sentence}")
        print(f"Most Similar Document: {most_similar_document}")
    else:
        print(f"No similar document found for: {search_sentence}")

    # Clean up the dummy corpus
    import shutil
    shutil.rmtree(corpus_path)
