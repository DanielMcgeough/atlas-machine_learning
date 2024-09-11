#!/usr/bin/env python3
"""Poisson functions
    Poisson-
    A Poisson distribution is a probability distribution that describes the number of times an event occurs in a fixed interval of time
    or space. It's often used to model random events that happen independently at a constant average rate. The key parameters of a Poisson 
    distribution are the average rate of events (lambda) and the time or space interval being considered.
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
