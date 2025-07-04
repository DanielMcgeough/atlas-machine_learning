#!/usr/bin/env python3

"""Concatenates dataframes"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """concats based on Timestamp people don't
    write it out because its hard to spell."""

    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.truncate(after=1417411920)

    df_concat = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return df_concat
