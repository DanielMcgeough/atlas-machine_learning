#!/usr/bin/env python3
"""Need to put the shebang here"""


def determinant(matrix):
    """Get the determinant of a matrix
     a useful number associated with a
     square matrix that tells you important
     things about the matrix, like 
     it's invertible and how it affects
     volumes under linear transformations."""

    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    #calculating the cofactor
    cof = 0
    for j in range(n):
        minor = [[matrix[i][k] for k in range(n) if k != j]
                 for i in range(1, n)]
        cof += matrix[0][j] * ((-1) ** j) * determinant(minor)

    return cof
