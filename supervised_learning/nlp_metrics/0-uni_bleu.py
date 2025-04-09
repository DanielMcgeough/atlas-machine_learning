#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence."""
import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
                           Each reference translation is a list of the words
                           in the translation.
        sentence (list): A list containing the model proposed sentence.

    Returns:
        float: The unigram BLEU score.
    """
    if not sentence:
        return 0.0

    clipped_count = 0
    for word in sentence:
        max_ref_count = 0
        for reference in references:
            ref_count = Counter(reference)[word]
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        if Counter(sentence)[word] > 0 and max_ref_count > 0:
            clipped_count += 1 if Counter(sentence)[word] <= max_ref_count else max_ref_count

    precision = clipped_count / len(sentence) if sentence else 0.0

    closest_ref_len = float('inf')
    sentence_len = len(sentence)
    for reference in references:
        ref_len = len(reference)
        if abs(ref_len - sentence_len) < abs(closest_ref_len - sentence_len):
            closest_ref_len = ref_len
        elif abs(ref_len - sentence_len) == abs(closest_ref_len - sentence_len):
            if ref_len < closest_ref_len:
                closest_ref_len = ref_len

    brevity_penalty = 1.0
    if sentence_len < closest_ref_len:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    bleu_score = brevity_penalty * precision
    return bleu_score
