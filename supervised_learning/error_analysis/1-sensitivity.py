#!/usr/bin/env python3
"""We like it like this"""
import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity
    otherwise known as the True
    Positive Rate."""

    num_classes = confusion.shape[0]
    sensitivity_values = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :]) - true_positives
        sensitivity_values[i] = true_positives / \
            (true_positives + false_negatives)

    return sensitivity_values
