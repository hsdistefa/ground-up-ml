from __future__ import division, print_function

import numpy as np
from sklearn import datasets

from groundupml.neuralnetwork.layers import Activation, FullyConnected
from groundupml.neuralnetwork.neural_network import NeuralNetwork
from groundupml.utils.data_manipulation import split_data
from groundupml.utils.functions import one_hot_to_class


if __name__ == '__main__':
    np.random.seed(100)

    # Set up data
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, y_train, X_test, y_test = split_data(X, y)

    # Set up neural network
    nn = NeuralNetwork(learning_rate=.01)
    n_input_nodes = np.shape(X_train)[1]
    n_hidden_nodes = 30
    n_output_nodes = len(np.unique(y_train))

    nn.add(FullyConnected(n_hidden_nodes, n_input_nodes))
    nn.add(FullyConnected(n_output_nodes))
    #nn.add(Activation('sigmoid'))

    # Train network and get test predictions
    nn.fit(X_train, y_train, 1)
    pred = nn.predict(X_test)
    print('predictions: ', pred[:10])
    print('predictions hard: ', one_hot_to_class(pred))
    print('actual: ', y_test)
