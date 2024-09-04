#!/usr/bin/env python3
"""Module defines the poly_derivative method"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list):
        return None
    """this part checks to make sure that poly is a valid input"""
    if len(poly) == 0:
        return None
    """this is to make sure the list isn't empty"""
    derivative = []
    """initializes empty list to store the coeff"""
    for i, coeff in enumerate(poly):
        """this is to iterate through the coeff using the index i"""
        if i == 0:
            continue
        """this is to skip 0 if its the constant term since the derivative
        of a constant or the rate of change is always 0"""
        derivative.append(i * coeff)
        """ for all other index values the coeff
        is multiplied and appened to the list
        this is based on the power rule for differentiation"""
    if not derivative or all(coeff == 0 for coeff in derivative):
        """This part just checks for 0"""
        return [0]
    return derivative
