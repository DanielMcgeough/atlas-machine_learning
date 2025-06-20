#!/usr/bin/env python3

"""sets new index rather than some useless id column"""


def index(df):
    """sets timestamp as the index"""

    df = df.set_index('Timestamp')

    return df
