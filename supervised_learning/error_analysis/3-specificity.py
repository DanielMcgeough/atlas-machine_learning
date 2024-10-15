#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def specificity(confusion):
    """Calculates specificity-
    ability of a model to correctly
    identify negative cases.
    It's also known as the true negative rate."""

    num_classes = confusion.shape[0]
    specificity_values = np.zeros(num_classes)

    for i in range(num_classes):
        true_negatives = np.sum(confusion) - np.sum(confusion[i, :]) - \
            np.sum(confusion[:, i]) + confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        specificity_values[i] =\
        true_negatives / (true_negatives + false_positives)

    return specificity_values
