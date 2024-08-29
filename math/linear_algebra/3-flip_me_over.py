#!/usr/bin/env python3
"""New module for transposing matrix"""


def matrix_transpose(matrix):
    """transposing a 2d Matrix by 'rotating' it"""
    transposed_matrix = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transposed_matrix.append(row)
    return transposed_matrix
