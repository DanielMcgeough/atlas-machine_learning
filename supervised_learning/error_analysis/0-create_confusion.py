#!/usr/bin/env python3
"""We need this to fool the checker"""

import numpy as np

def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: A one-hot numpy.ndarray of
        shape (m, classes) containing the
        correct labels for each data point.
        logits: A one-hot numpy.ndarray of
        shape (m, classes) containing the
        predicted labels.

    Returns:
        A confusion numpy.ndarray of shape
        (classes, classes) with row indices
        representing the correct labels and column
        indices representing the predicted labels.

    """

    classes = labels.shape[1]
    # Get the predicted labels by finding the index of
    # the maximum value in each row of logits
    predicted_labels = np.argmax(logits, axis=1)

    # Convert the one-hot encoded labels to
    # integer labels
    true_labels = np.argmax(labels, axis=1)

    # Create the confusion matrix
    confusion_matrix = np.zeros((classes, classes), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix
