#!/usr/bin/env python3

"""some pruning from a DataFrame"""


def prune(df):
    """Drops NaN values. Very commonly used as
    a placeholder for missing or
    undefined values in numerical data as well"""

    df_pruned = df.dropna(subset=["Close"])

    return df_pruned
