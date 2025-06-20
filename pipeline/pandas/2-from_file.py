#!/usr/bin/env python3

"""loads from a file and makes a dataframe
out of it"""

import pandas as pd


def from_file(filename, delimiter):
    """Pulls a file and makes a dataframe
    of it"""
    
    df_return = pd.read_csv(filename, delimiter=delimiter)

    return df_return
