#!/usr/bin/env python3

"""love how short these are, pandas is nice"""


def analyze(df):
    """computes descriptive statistics for columns"""

    df_stats = df.drop('Timestamp', axis=1)

    stats = df_stats.describe()

    return stats
