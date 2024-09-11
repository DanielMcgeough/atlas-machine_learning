#!/usr/bin/env python3
"""Poisson functions
    Poisson-
    A Poisson distribution is a probability distribution
    that describes the number of times an event occurs
    in a fixed interval of time or space. It's often used
    to model random events that happen independently at
    a constant average rate. The key parameters of a Poisson
    distribution are the average rate of events (lambda)
    and the time or space interval being considered.
"""


class Poisson:
    """Represents a Poisson Distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson Distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Create a PMF instance which calculates successes"""
        k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        e_lambda = e ** (-self.lambtha)

        lambda_k = self.lambtha ** k

        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i
        
        return (e_lambda * lambda_k) / k_factorial
        """base of natural logarithm multiplied by
        the average rate of events equals k factorial"""
