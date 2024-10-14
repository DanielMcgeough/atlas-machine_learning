#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def specificity(confusion):
    """Calculates specificity for each class in a confusion matrix.

    Args:
        confusion: A confusion matrix as a numpy.ndarray.

    Returns:
        A numpy.ndarray containing the specificity for each class.
    """

    num_classes = confusion.shape[1]
    specificity_values = np.zeros(num_classes)

    for i in range(num_classes):
        true_negatives = np.sum(confusion[np.arange(num_classes) != i, np.arange(num_classes) != i])
        false_positives = np.sum(confusion[np.arange(num_classes) != i, i])
        specificity_values[i] = true_negatives / (true_negatives + false_positives)

    return specificity_values
