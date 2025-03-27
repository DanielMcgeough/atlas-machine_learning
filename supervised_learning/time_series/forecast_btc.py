#!/usr/bin/env python3
"""
This module loads preprocessed Bitcoin data, trains an RNN model for forecasting,
and evaluates its performance.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

PREPROCESSED_FILE = 'preprocessed_btc_data.npz'
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 20
SEQUENCE_LENGTH = 24 * 60  # Must match preprocess_data.py


def load_preprocessed_data() -> tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None, np.ndarray or None]:
    """
    Loads the preprocessed Bitcoin data from the .npz file.

    Returns:
        tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None, np.ndarray or None]:
        A tuple containing sequences, targets, scaler min, and scaler max, or None values if the file is not found.
    """
    try:
        data = np.load(PREPROCESSED_FILE)
        sequences = data['sequences']
        targets = data['targets']
        scaler_min = data['scaler_min']
        scaler_max = data['scaler_max']
        return sequences, targets, scaler_min, scaler_max
    except FileNotFoundError:
        print(f"Error: Preprocessed data file not found at {PREPROCESSED_FILE}. Run preprocess_data.py first.")
        return None, None, None, None


def build_rnn_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """
    Builds a Keras LSTM-based RNN model for time series forecasting.

    Args:
        input_shape (tuple[int, int]): The shape of the input sequences (sequence_length, num_features).

    Returns:
        tf.keras.Model: The compiled RNN model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def inverse_transform(predictions: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """
    Inverse transforms the scaled predictions back to the original price scale.

    Args:
        predictions (np.ndarray): The scaled predictions.
        min_val (np.ndarray): The minimum values of the original training data for scaling.
        max_val (np.ndarray): The maximum values of the original training data for scaling.

    Returns:
        np.ndarray: The predictions in the original price scale.
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    dummy_array = np.zeros((predictions.shape[0], 7))  # Assuming 7 features were scaled
    dummy_array[:, 3] = predictions[:, 0]
    scaler = MinMaxScaler()
    scaler.min_ = min_val
    scaler.max_ = max_val
    inverted_data = scaler.inverse_transform(dummy_array)
    return inverted_data[:, 3]


def main():
    """
    Main function to load preprocessed data, train the RNN model, and evaluate it.
    """
    sequences, targets, scaler_min, scaler_max = load_preprocessed_data()

    if sequences is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=TEST_SIZE, shuffle=False)

    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
    model = build_rnn_model(input_shape)
    model.summary()

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean Squared Error on Test Data: {loss:.4f}")

    predictions = model.predict(X_test)

    predicted_prices = inverse_transform(predictions, scaler_min, scaler_max)
    actual_prices = inverse_transform(y_test.reshape(-1, 1), scaler_min, scaler_max)

    mse_original_scale = mean_squared_error(actual_prices, predicted_prices)
    print(f"Mean Squared Error on Original Scale: {mse_original_scale:.2f} USD^2")

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual BTC Price')
    plt.plot(predicted_prices, label='Predicted BTC Price')
    plt.title('BTC Price Forecasting')
    plt.xlabel('Time Steps (in hours, approx.)')
    plt.ylabel('BTC Price (USD)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
