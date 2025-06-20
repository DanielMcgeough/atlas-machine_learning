#!/usr/bin/env python3

"""Slices a dataframe"""


def slice(df):
    """We want every 60th row
    because reasons"""

    df_sliced = df[["High", "Low", "Close", "Volume_(BTC)"]]

    df_sliced = df_sliced.iloc[::60, :]

    return df_sliced
