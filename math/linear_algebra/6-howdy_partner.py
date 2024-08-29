#!/usr/bin/env python3
"""this defines the method that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    concat_array = []
    concat_array.extend(arr1)
    concat_array.extend(arr2)
    return concat_array
