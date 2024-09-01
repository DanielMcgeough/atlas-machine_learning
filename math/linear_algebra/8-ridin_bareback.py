#!/usr/bin/env python3
"""module defines the matrix multiplication function"""


def mat_mul(mat1, mat2):
    """multiplies two matrices containing ints or floats"""
    if len(mat1[0]) != len(mat2):
        return None

    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
