#!/usr/bin/env python3
"""saying stuff"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creating a confusion matrix with labels(actual)
    and logits(predicted)"""
    true_labels = labels.argmax(axis=1)
    predicted_labels = logits.argmax(axis=1)

    confused_matrix = np.zeros((len(np.unique(true_labels)
                            ), len(np.unique(predicted_labels))))

    for i, true_label in enumerate(true_labels):
        confused_matrix[true_label, predicted_labels[i]] += 1

    return confused_matrix
