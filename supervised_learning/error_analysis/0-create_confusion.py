#!/usr/bin/env python3
"""saying stuff"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: A one-hot numpy.ndarray of shape (m, classes) containing the correct
            labels for each data point.
        logits: A one-hot numpy.ndarray of shape (m, classes) containing the predicted
            labels.

    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column
            indices representing the predicted
        labels.

    """

    # Get the number of classes and data points
    classes, _ = labels.shape
    m, _ = logits.shape

    # Create a confusion matrix initialized with zeros
    confusion_matrix = np.zeros((classes, classes), dtype=int)

    # Iterate over each data point
    for i in range(m):
        # Get the correct label index
        correct_label_index = np.argmax(labels[i])
        # Get the predicted label index
        predicted_label_index = np.argmax(logits[i])

        # Increment the corresponding element in the confusion matrix
        confusion_matrix[correct_label_index, predicted_label_index] += 1

    return confusion_matrix
