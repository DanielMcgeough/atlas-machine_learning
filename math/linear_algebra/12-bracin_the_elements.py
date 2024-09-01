#!/usr/bin/env python3
"""module defines elementwise method for numpy matrix mathematics"""


def np_elementwise(mat1, mat2):
    """elementwise arithmetic methods"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
