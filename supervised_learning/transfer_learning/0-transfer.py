#!/usr/bin/env python3
"""Lots of stuff to setup"""
import tensorflow as tf
from tensorflow import keras as K
from keras.layers import Lambda
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical 
import numpy as np


def preprocess_data(X, Y):
    # Normalize the image data because CIFAR-10
    # has pictures that are 32 by 32 pixels so
    # need to normalize them on 255 by 255 format
    X = X.astype('float32') / 255.0

    # One-hot encode the labels
    Y = to_categorical(Y, 10)
    print(f"Error:31")
    return (X, Y)

def train_cifar10_model():
    print(f"Error: 1")
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Create the base model using VGG16 as a base is usually adequate
    # It was trained on imagenet so for image classification it is fantastic
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    print(f"Error:2")
    # Freeze the base model this prevents the layers in the model from being adjusted
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    print(f"Error:3")
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"Error:4")
    # Training the model with ten epochs
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # Save the model
    model.save('cifar10.h5')
    print(f"Error:5")
    # The script won't run when imported per the instructions
    if __name__ == '__main__':
        train_cifar10_model()
