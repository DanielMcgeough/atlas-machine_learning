#!/usr/bin/env python3
"""Histogram stuff """
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Histogram function"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black',
             linewidth=1)
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.title('Project A')
    plt.xticks(range(0, 101, 10))

    plt.show()