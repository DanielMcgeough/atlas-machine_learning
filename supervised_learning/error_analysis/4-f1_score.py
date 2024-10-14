#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def f1_score(confusion):
    """Calculates F1 score for each class in a confusion matrix.

    Args:
        confusion: A confusion matrix as a numpy.ndarray.

    Returns:
        A numpy.ndarray containing the F1 score for each class.
    """

    sensitivity_func = __import__('1-sensitivity').sensitivity
    precision_func = __import__('2-precision').precision

    sensitivity = sensitivity_func(confusion)
    precision = precision_func(confusion)

    f1_scores = 2 * (sensitivity * precision) / (sensitivity + precision)
    f1_scores[np.isnan(f1_scores)] = 0
    # Handle cases where sensitivity or precision is 0

    return f1_scores
