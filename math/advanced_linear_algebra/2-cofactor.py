#!/usr/bin/env python3
"""the third major number tied into square
matrices along with determinant and minor"""
determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor
sub_matrix = __import__('1-minor').sub_matrix


def cofactor(matrix):
    """The cofactor of an element in a
    square matrix is a signed minor.
    It's a key concept in calculating
    determinants and matrix inverses.
    """
    n = len(matrix)
    result = [[0 for _ in range(n)] for _ in range(n)]
    new_matrix = minor(matrix)

    for r in range(n):
        for c in range(n):
            x = sub_matrix(matrix, r, c)
            if x == [[]]:
                return [[1]]
            result[r][c] = determinant(x) * ((-1) ** (r + c))
    return result
