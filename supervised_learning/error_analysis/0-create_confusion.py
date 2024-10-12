#!/usr/bin/env python3
"""saying stuff"""
import numpy as np


def create_confusion_matrix(labels, logits):
    # Convert one-hot encoded labels to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)
    
    # Get the number of classes
    classes = labels.shape[1]
    
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((classes, classes), dtype=int)
    
    # Populate the confusion matrix
    for true, pred in zip(true_labels, predicted_labels):
        confusion_matrix[true, pred] += 1
    
    return confusion_matrix
