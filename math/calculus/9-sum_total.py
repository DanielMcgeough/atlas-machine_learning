#!/usr/bin/env python3
"""Module defines the summation_i_squared method"""


def summation_i_squared(n):
    """Calculates the sum of squares of numbers from 1 to n
    n is undefined and could represent any integer >= 0
    """
    try:
        n = int(n)
        if n <= 0:
            return None

        # return the sum of squares
        return (n * (n + 1) * (2 * n + 1) / 6)

    # Handle n is not an int or float
    except (ValueError, TypeError):
        return None
