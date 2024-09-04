#!/usr/bin/env python3
"""Module defines the poly_derivative method"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial represented as a list of coefficients.

    Args:
        poly: A list of coefficients representing the polynomial.
        C: An integer representing the integration constant.

    Returns:
        A list of coefficients representing the integral of the polynomial, or None if the input is invalid.
    """

    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, int):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        if i == 0:
            continue
        new_coeff = coeff / (i + 1)
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)
        integral.append(new_coeff)

    return integral
