#!/usr/bin/env python3
"""shebang stuff"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """to create these matrices and to utilize
    some of the number required depend upon earlier
    operations."""
    cofactor_matrix = cofactor(matrix)
    n = len(matrix)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            result[c][r] = cofactor_matrix[r][c]
    return result
