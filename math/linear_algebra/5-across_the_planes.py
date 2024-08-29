#!/usr/bin/env python3
"""module defines the across the planes method"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """this addeds two 2d matrices together"""
    mat3 = []
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])
        mat3.append(row)
    return mat3
