#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def precision(confusion):
    """Calculates precision of
    how good the model is at identifying
    negative cases correctly. It does this
    by focusing on the positive predictions
    that are correct."""

    num_classes = confusion.shape[1]
    precision_values = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - true_positives
        precision_values[i] = true_positives /\
        (true_positives + false_positives)
