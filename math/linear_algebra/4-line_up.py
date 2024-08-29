#!/usr/bin/env python3
""" module for element wise ops"""


def add_arrays(arr1, arr2):
    """adding arrays element wise
        elements are ints or floats
        for now
    """
    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
