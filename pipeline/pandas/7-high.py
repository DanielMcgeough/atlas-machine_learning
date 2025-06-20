#!/usr/bin/env python3

"""Sorts by highest price in descending order"""


def high(df):
    """Almost like an SQL query with its
    changing to sort things like a SELECT"""

    df_sort = df.sort_values(by=["High"], ascending=False)

    return df_sort
