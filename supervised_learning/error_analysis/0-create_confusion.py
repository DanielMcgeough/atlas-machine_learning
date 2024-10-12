#!/usr/bin/env python3
"""saying stuff"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Documentation"""
    true_labels = labels.argmax(axis=1)
    predicted_labels = logits.argmax(axis=1)

    con_mat = np.zeros((len(np.unique(true_labels)
                            ), len(np.unique(predicted_labels))))

    for i, true_label in enumerate(true_labels):
        con_mat[true_label, predicted_labels[i]] += 1

    return con_mat
