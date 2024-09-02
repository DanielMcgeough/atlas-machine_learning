#!/usr/bin/env python3
"""This is the module to create a line plot using Matplotlib"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """creates a line plot"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(np.arange(0, 11), y, color='red', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
