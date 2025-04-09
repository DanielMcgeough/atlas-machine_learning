#!/usr/bin/env python3
"""Calculates the n-gram BLEU score for a sentence."""
import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
                           Each reference translation is a list of the words
                           in the translation.
        sentence (list): A list containing the model proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: The n-gram BLEU score.
    """
    if not sentence or n <= 0:
        return 0.0

    def _get_ngrams(words, n_gram):
        """Generates n-grams from a list of words."""
        ngrams = []
        for i in range(len(words) - n_gram + 1):
            ngrams.append(tuple(words[i:i + n_gram]))
        return ngrams

    sentence_ngrams = _get_ngrams(sentence, n)
    if not sentence_ngrams:
        return 0.0

    clipped_count = 0
    for ngram in sentence_ngrams:
        max_ref_count = 0
        for reference in references:
            ref_ngrams = _get_ngrams(reference, n)
            ref_count = Counter(ref_ngrams)[ngram]
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        if Counter(sentence_ngrams)[ngram] > 0 and max_ref_count > 0:
            clipped_count += min(Counter(sentence_ngrams)[ngram], max_ref_count)

    precision = clipped_count / len(sentence_ngrams) if sentence_ngrams else 0.0

    closest_ref_len = float('inf')
    sentence_len = len(sentence)
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - sentence_len)
        closest_diff = abs(closest_ref_len - sentence_len)
        if diff < closest_diff:
            closest_ref_len = ref_len
        elif diff == closest_diff and ref_len < closest_ref_len:
            closest_ref_len = ref_len

    brevity_penalty = 1.0
    if sentence_len < closest_ref_len:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    bleu_score = brevity_penalty * precision
    return bleu_score
