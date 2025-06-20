#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns=['Weighted_Price'], inplace=True)

df.rename(columns={'Timestamp': 'Date'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df.set_index('Date', inplace=True)

df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df = df[df.index >= '2017-01-01']

daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

daily_df.plot(subplots=True, figsize=(12, 16))
plt.suptitle('Daily Cryptocurrency 2017 onwards')
plt.show()

print (daily_df)
