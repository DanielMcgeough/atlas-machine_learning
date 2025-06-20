#!/usr/bin/env python3

"""flip and reverse a DataFrame"""


def flip_switch(df):
    """sort in reverse chronological and then
    transpose and display"""

    df_flip_switch = df.sort_values(by=['Timestamp'], ascending=False)

    df_flip_switch = df_flip_switch.T

    return df_flip_switch
