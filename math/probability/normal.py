#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution.
        Normal Distribution- Also known as the Gaussian
        distribution or bell curve, the normal distribution
        is a continuous probability distribution that is
        widely used to model real-world data. It's characterized
        by its bell-shaped curve, with the peak representing
        the most likely value and the tails extending outwards.
        """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = (sum((x - self.mean) ** 2 for x in data) /
                           len(data)) ** 0.5
            self.stddev = float(self.stddev)

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given x-value.
        """
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coefficient * e ** exponent

    def cdf(self, x):
        """
        Calculate the Cumulative
        Distribution Function (CDF)
        for a givenx-value.
        """
        z = self.z_score(x)
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Use numerical integration (Simpson's rule) to approximate the CDF
        num_steps = 1000
        step_size = z / num_steps
        integral = 0

        for i in range(num_steps):
            x0 = i * step_size
            x1 = (i + 1) * step_size
            integral += (self.pdf(x0) + 4 * self.pdf((x0 + x1) / 2) +
                         self.pdf(x1)) * step_size / 6

        return 0.5 + integral
