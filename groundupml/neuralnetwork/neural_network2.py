from __future__ import print_function, division

import numpy as np

from groundupml.utils.functions import to_one_hot, one_hot_to_hard_pred
from groundupml.utils.data_manipulation import shuffle_data


class LinearLayer():
    """Linear Layer

    Args:
        n_nodes (int):
            Number of nodes in layer.
        n_inputs (int):
            Number of inputs to layer.
        learning_rate (float, optional):
            Step magnitude used for updating layer weights when relevant.
    """
    def __init__(self, n_nodes, n_inputs, learning_rate=.01):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate

    def init_weights(self):
        """Initialize weights and biases"""
        self.weights = np.random.randn(self.n_nodes, self.n_inputs)
        self.biases = np.zeros(self.n_nodes)

    def set_learning_rate(self, learning_rate):
        """Set learning rate

        Args:
            learning_rate (float):
                Step magnitude used for updating layer weights when relevant.
        """
        self.learning_rate = learning_rate

    def forward_propogate(self, X):
        """Propogate input forward through layer

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Input data

        Returns:
            numpy array of shape [n_samples, n_nodes]:
                Output of layer
        """
        self.X = X
        self.z = np.dot(X, self.weights.T) + self.biases

        return self.z

    def back_propogate(self, gradient):
        """Propogate error gradient backward through layer using chain rule

        Args:
            gradient (numpy array of shape [n_samples, n_nodes]):
                Error gradient

        Returns:
            numpy array of shape [n_samples, n_inputs]:
                Error gradient for layer below
        """
        self.gradient = gradient
        self.d_weights = np.dot(gradient.T, self.X)
        self.d_biases = np.sum(gradient, axis=0)
        self.d_X = np.dot(gradient, self.weights)

        return self.d_X

    def update_weights(self):
        """Update weights and biases"""
        self.weights -= self.learning_rate * self.d_weights
        self.biases -= self.learning_rate * self.d_biases

    def __repr__(self):
        return 'LinearLayer(n_nodes={}, n_inputs={}, learning_rate={})'.format(
            self.n_nodes, self.n_inputs, self.learning_rate)


if __name__ == '__main__':
    # Test LinearLayer
    layer = LinearLayer(3, 2, learning_rate=.01)
    layer.init_weights()
    layer.set_learning_rate(.01)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print('X.shape:', X.shape)
    print('y.shape:', y.shape)

    z = layer.forward_propogate(X)
    print('z:', z.shape)

    gradient = layer.back_propogate(y)
    print('gradient.shape:', gradient.shape)
    print('gradients', gradient)

    layer.update_weights()

    print(layer)
