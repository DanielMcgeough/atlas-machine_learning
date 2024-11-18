#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np


def preprocess_data(X, Y):
    # Normalize the image data
    X = X.astype('float32') / 255.0

    # One-hot encode the labels
    Y = to_categorical(Y, 10)

    return X, Y

def train_cifar10_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Create the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False


    # Add custom layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # Save the model
    model.save('cifar10.h5')

if __name__ == '__main__':
    train_cifar10_model()
