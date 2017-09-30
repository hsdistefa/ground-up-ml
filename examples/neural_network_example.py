from __future__ import division, print_function

from neuralnetwork.layers import Activation, FullyConnected
from neuralnetwork.neural_network import NeuralNetwork

import numpy as np

if __name__ == '__main__':
    np.random.seed(100)

    # Set up data
    X_train = np.array([[2.6, 2.5], [8, 0.3], [3, 3], [3, 4], [1.4, 1.8],
                        [6.3, 1.0], [8.5, -0.2], [1.5, 2.4], [7, 3], [7, 0.5]])
    y_train = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])

    X_test = np.array([[3.1, 3.3], [2.8, 3.7], [7.2, 0]])
    y_test = np.array([0, 0, 1])

    # Set up neural network
    # NOTE: this network only has one hidden layer
    nn = NeuralNetwork(learning_rate=.01)
    n_hidden_nodes = 2

    nn.add(FullyConnected(n_hidden_nodes, np.shape(X_train)[1]))
    nn.add(Activation(len(np.unique(y_train)), activation_func='sigmoid'))

    # Train network and get test predictions
    nn.fit(X_train, y_train, 200)
    print('predictions: ', nn.predict(X_test))
    print('actual: ', y_test)
