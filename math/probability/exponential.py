#!/usr/bin/env python3
"""Exponential Distribution
Exponential- The exponential distribution is a continuous
probability distribution that describes the time between
events in a Poisson process. It is often used to model the
time between occurrences of random events, such as the time
between arrivals at a service station or the time until a
machine failure.
"""


class Exponential:
    """creates a class for Expo Dist"""

    def __init__(self, data=None, lambtha=1.):
        """Initializes exponential distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculate value of PDF over time"""

        if x < 0:
            return 0
        
        e = 2.7182818285

        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculate the value of CDF over time"""
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - e ** (-self.lambtha * x)
