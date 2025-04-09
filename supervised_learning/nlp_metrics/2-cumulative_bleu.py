#!/usr/bin/env python3
"""Calculates the cumulative n-gram BLEU score for a sentence."""
import numpy as np
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
                           Each reference translation is a list of the words
                           in the translation.
        sentence (list): A list containing the model proposed sentence.
        n (int): The size of the largest n-gram to use for evaluation.
                 All n-gram scores (from 1 to n) should be weighted evenly.

    Returns:
        float: The cumulative n-gram BLEU score.
    """
    if not sentence or n <= 0:
        return 0.0

    def _get_ngrams(words, n_gram):
        """Generates n-grams from a list of words."""
        ngrams = []
        for i in range(len(words) - n_gram + 1):
            ngrams.append(tuple(words[i:i + n_gram]))
        return ngrams

    sentence_len = len(sentence)
    closest_ref_len = float('inf')
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

    log_sum = 0.0
    precisions = []
    for i in range(1, n + 1):
        sentence_ngrams = _get_ngrams(sentence, i)
        if not sentence_ngrams:
            precision = 0.0
        else:
            clipped_count = 0
            for ngram in sentence_ngrams:
                max_ref_count = 0
                for reference in references:
                    ref_ngrams = _get_ngrams(reference, i)
                    ref_count = Counter(ref_ngrams)[ngram]
                    if ref_count > max_ref_count:
                        max_ref_count = ref_count
                clipped_count += min(Counter(sentence_ngrams)[ngram], max_ref_count)
            precision = clipped_count / len(sentence_ngrams)
        precisions.append(precision)
        log_sum += np.log(precision) if precision > 0 else 0

    print(f"Precisions: {precisions}")  # Added for debugging

    cumulative_bleu_score = brevity_penalty * np.exp(log_sum / n) if n > 0 else 0.0
    return cumulative_bleu_score


if __name__ == '__main__':
    references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
    sentence = ["there", "is", "a", "cat", "here"]

    print(cumulative_bleu(references, sentence, 4))
