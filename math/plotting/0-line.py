#!/usr/bin/env python3
"""This is the module to create a line plot using Matplotlib"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Creates a line plot"""
    x = np.arrange(0, 11)
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='red', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
