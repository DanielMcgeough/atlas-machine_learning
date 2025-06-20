#!/usr/bin/env python3
"""takes a numpy array and returns it as a
pandas array"""

import pandas as pd


def from_numpy(array):
    """Sort of work for a simple library.
    Pandas seems to like NaNs for placeholders"""

    # print(f"len of array.shape[0]: {array.shape[1]}")

    columns = []

    columns = (chr(65+x) for x in range(array.shape[1]))

    # print(f"column names: {columns}")

    df_return = pd.DataFrame(array, columns=columns)

    return df_return
