#!/usr/bin/env python3
"""Module documentation for py files"""
import numpy as np


class Neuron:
    """A class that defines a neuron"""
    def __init__(self, nx):
        """Initializes neuron class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
