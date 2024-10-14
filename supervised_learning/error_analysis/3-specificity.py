#!/usr/bin/env python3
"""Never know what to put here"""
import numpy as np


def specificity(confusion):
    """Measures the ability of the model to
    correctly identify negative cases.
    A high specificity means the model is
    good at avoiding false positives."""
    specificity_return = []
    num_classes = confusion.shape[0]
    true_positives = np.diag(confusion)
    true_negatives = np.diag(confusion)
    false_positives = np.array([np.sum(row) - tp for tp, row in zip(true_positives, confusion)])

    for i in range(num_classes):
        true_negatives = true_negatives[i]
        false_positives = false_positives[i]

        specificity = true_negatives / (true_negatives + false_positives) if false_positives > 0 else 1
        specificity_return.append(specificity)

    return specificity_return
