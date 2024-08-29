#!/usr/bin/env python3
"""this defines the method that concatenates matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concates two matrices with an axis passed in"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
