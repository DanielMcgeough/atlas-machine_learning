#!/usr/bin/env python3
"""saying stuff"""
import numpy as np

def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: A one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point.
        logits: A one-hot numpy.ndarray of shape (m, classes) containing the predicted labels.

    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices representing the correct labels and column   
        indices representing the predicted labels.   

    """

    # Get the predicted labels by finding the index of the maximum value in each row of logits
    predicted_labels = np.argmax(logits, axis=1)

    # Convert the one-hot encoded labels to integer labels
    true_labels = np.argmax(labels, axis=1)

    # Get the number of classes from the shape of the labels array
    classes = labels.shape[1]

    # Create the confusion matrix
    confusion_matrix = np.zeros((classes, classes), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label]+= 1

    # Print the confusion matrix with periods after each number
    for row in confusion_matrix:
        for element in row:
            print(f"{element:.0f}.", end=" ")
    print()

    return confusion_matrix
