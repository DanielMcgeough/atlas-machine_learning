import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers, activation='sig'):
        """
        Initializes a deep neural network.

        Args:
            nx: The number of input features.
            layers: A list containing the number of neurons in each layer.
            activation: The type of activation function to use (default: 'sig').
                Must be 'sig' or 'tanh'.
        """

        # Check for valid activation function
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__layers = layers
        self.__weights = {}
        self.__biases = {}
        self.__activation = activation

        # Initialize weights and biases
        for i in range(1, len(layers)):
            prev_layer = layers[i - 1]
            current_layer = layers[i]
            self.__weights[f"W{i}"] = np.random.randn(current_layer, prev_layer) * np.sqrt(2 / prev_layer)
            self.__biases[f"b{i}"] = np.zeros((current_layer, 1))

    @property
    def activation(self):
        """
        Getter for the activation function.
        """
        return self.__activation

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.

        Args:
            X: The input data (numpy.ndarray).

        Returns:
            A list containing the activations of each layer.
        """

        A = X
        activations = [A]

        for i in range(1, len(self.__layers)):
            W = self.__weights[f"W{i}"]
            b = self.__biases[f"b{i}"]
            Z = np.dot(W, A) + b

            # Apply activation function based on self.__activation
            if self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            elif self.__activation == 'tanh':
                A = np.tanh(Z)
            else:
                raise ValueError("Unexpected activation function")

            activations.append(A)

        return activations

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the network.

        Args:
            Y: The correct labels (numpy.ndarray).
            cache: A dictionary containing the activations from forward prop.
            alpha: The learning rate.
        """

        m = Y.shape[1]  # Number of examples
        L = len(self.__layers) - 1  # Number of hidden layers

        # Backpropagate through the network
        dA = -(Y - cache["A" + str(L)])

        for i in range(L, 0, -1):
            if self.__activation == 'sig':
                dZ = dA * cache["A" + str(i)] * (1 - cache["A" + str(i)])
            elif self.__activation == 'tanh':
                dZ = dA * (1 - np.power(cache["A" + str(i)], 2))
            else:
                raise ValueError("Unexpected activation function")

            dW = np.dot(dZ, cache["A" + str(i - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights and biases
            self.__weights[f"W{i}"] -= alpha * dW
            self.__biases[f"b{i}"] -= alpha * db

            # Update dA for the next iteration
            dA = np.dot(self.__weights[f"W{i}"].T, dZ)
