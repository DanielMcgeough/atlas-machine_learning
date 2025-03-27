#!/usr/bin/env python3
"""
This module loads, preprocesses, and saves Bitcoin price data from CSV files.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DATA_DIR = [r'C:\Users\danny\Downloads\bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv',
            r'C:\Users\danny\Downloads\coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv']
OUTPUT_FILE = 'preprocessed_btc_data.npz'
SEQUENCE_LENGTH = 24 * 60  # 24 hours in minutes
FORECAST_HORIZON = 60     # Predict the close of the next hour (60 minutes)


def load_data(filepath: str) -> pd.DataFrame or None:
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): The full path to the CSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(filepath, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_usd', 'vwap'])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray or None, MinMaxScaler or None]:
    """
    Preprocesses the Bitcoin data by selecting features and scaling them.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple[np.ndarray or None, MinMaxScaler or None]: The scaled data and the fitted MinMaxScaler,
                                                        or (None, None) if the input DataFrame is None.
    """
    if df is None:
        return None, None

    print(f"Preprocessing data from {df.head(1)['timestamp'].values[0]} to {df.tail(1)['timestamp'].values[0]}")

    features = ['open', 'high', 'low', 'close', 'volume_btc', 'volume_usd', 'vwap']
    df_processed = df[features].copy()

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_processed)

    return df_scaled, scaler


def create_sequences(data: np.ndarray, seq_length: int, forecast_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences and target values from the scaled time series data.

    Args:
        data (np.ndarray): The scaled time series data.
        seq_length (int): The length of each input sequence (in minutes).
        forecast_horizon (int): The number of steps ahead to predict (in minutes).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the input sequences and the target values.
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length + forecast_horizon - 1, 3])  # Predict 'close' price
    return np.array(sequences), np.array(targets)


def main():
    """
    Main function to load, preprocess, and save Bitcoin data.
    """
    all_dfs = []
    for filepath in DATA_DIR:
        df = load_data(filepath)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("Error: No data loaded.")
        return

    combined_df = pd.concat(all_dfs).sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    combined_df = combined_df.reset_index(drop=True)

    scaled_data, scaler = preprocess(combined_df)

    if scaled_data is not None:
        sequences, targets = create_sequences(scaled_data, SEQUENCE_LENGTH, FORECAST_HORIZON)

        np.savez_compressed(OUTPUT_FILE, sequences=sequences, targets=targets, scaler_min=scaler.data_min_, scaler_max=scaler.data_max_)
        print(f"Preprocessed data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
# trying to get this to work.
