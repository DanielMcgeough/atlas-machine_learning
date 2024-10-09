#!/usr/bin/env python3
"""Stupid stuff making me"""
import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set.

    Args:
        data: A list of data to calculate the moving average of.
        beta: The weight used for the moving average.

    Returns:
        A list containing the moving averages of data.   

    """

    # Initialize the moving average
    ma = [data[0]]

    # Calculate the moving average for the rest of the data points
    for i in range(1, len(data)):
        ma.append(beta * ma[-1] + (1 - beta) * data[i])

    # Apply bias correction
    ma = [ma[0]] + [ma[i] / (1 - beta**i) for i in range(1, len(ma))]

    return ma
