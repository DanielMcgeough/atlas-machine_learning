#!/usr/bin/env python3
"""shebang needed to run module """


def matrix_shape(matrix):
    """function to calculate shape of matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if matrix:
            matrix = matrix[0]
        else:
            break
    return shape
