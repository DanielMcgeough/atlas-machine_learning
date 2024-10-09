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

    weighted_average = 0
    moving_average = []
    for i, x in enumerate(data, 1):
        weighted_average = beta * weighted_average + (1 - beta) * x
        moving_average.append(weighted_average / (1 - beta**i))
    return moving_average
