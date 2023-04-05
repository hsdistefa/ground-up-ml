from __future__ import print_function, division

import numpy as np

from groundupml.utils.functions import sigmoid, sigmoid_prime, softmax


class LinearLayer():
    """Fully connected layer that applies a linear transformation to the input:
    :math:`y = XW^T + b`
    Where W is the learnable weights matrix, b is the learnablebiases vector, 
    and X is the input.

    Args:
        n_nodes (int):
            Number of nodes in layer.
        n_inputs (int):
            Number of inputs to layer.
        learning_rate (float, optional):
            Step magnitude used for updating layer weights when relevant.
    """
    def __init__(self, n_nodes, n_inputs, learning_rate=.01):
        # TODO: Remove unnecessary attributes
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.weights = None
        self.biases = None
        self.X = None
        self.z = None
        self.gradient = None
        self.d_weights = None
        self.d_biases = None
        self.d_X = None

    def init_weights(self):
        """Initialize weights and biases"""
        stdv = np.sqrt(1. / self.n_inputs)
        self.weights = np.random.uniform(-stdv, stdv, 
                                         size=(self.n_nodes, self.n_inputs))
        self.biases = np.random.uniform(-stdv, stdv,
                                        size=(self.n_nodes))

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
                Linearly transformed output of layer
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
    import matplotlib.pyplot as plt

    from sklearn import datasets

    from groundupml.utils.data_manipulation import split_data, scale_min_max
    from groundupml.utils.functions import one_hot_to_class, to_one_hot
    from groundupml.utils.data_tools import confusion_matrix

    # Set seed for reproducibility
    np.random.seed(1)

    # Load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test sets
    X_train, y_train, X_test, y_test = split_data(X, y, proportion=0.8)

    # Scale features to be between 0 and 1 for NN
    X_train = scale_min_max(X_train)
    X_test = scale_min_max(X_test)

    # Convert labels to one hot vectors
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    print('Train shapes:', X_train.shape, y_train.shape)
    print('Test shapes:', X_test.shape, y_test.shape)

    # Train neural network
    n_epochs = 10000
    learning_rate = 1e-2

    fc_layer1 = LinearLayer(n_nodes=3, n_inputs=4, learning_rate=learning_rate)
    fc_layer1.init_weights()
    print(fc_layer1)
    fc_layer2 = LinearLayer(n_nodes=3, n_inputs=3, learning_rate=learning_rate)
    fc_layer2.init_weights()
    print(fc_layer2)
    sm_layer = SoftmaxLayer()

    costs = []
    for i in range(n_epochs):
        # Forward propogate
        z1 = fc_layer1.forward_propogate(X_train)
        #print('z1:', z1.shape)
        z2 = fc_layer2.forward_propogate(z1)
        #print('z2:', z2.shape)
        #print('z2:', z2[:5])
        activations = sm_layer.forward_propogate(z2)
        #print('activations.shape:', activations.shape)
        #print('activations:', activations[:5])

        # Calculate costs using squared error
        squared_error = 0.5 * np.sum((y_train - activations)**2)
        costs.append(squared_error)

        # Backward propogate error gradients
        d_error = activations - y_train
        #print('d_error:', d_error)
        gradients_a = sm_layer.back_propogate(d_error)
        #print('Gradients a:', gradients_a)
        #print('gradients_a.shape:', gradients_a.shape)
        gradients_z2 = fc_layer2.back_propogate(gradients_a)
        #print('Weights shape:',fc_layer2.weights.shape)
        #print('d_weights shape:', fc_layer2.d_weights.shape)
        fc_layer2.update_weights()
        #print('gradients_z2.shape:', gradients_z2.shape)
        gradients_z1 = fc_layer1.back_propogate(gradients_z2)
        fc_layer1.update_weights()
        #print('gradients_z1.shape:', gradients_z1.shape)

        if i % 1000 == 0:
            print('Epoch:', i, 'Cost:', costs[-1])

    # Get test predictions
    z1 = fc_layer1.forward_propogate(X_test)
    z2 = fc_layer2.forward_propogate(z1)
    a = sm_layer.forward_propogate(z2)
    predictions = np.argmax(a, axis=1)
    actual = one_hot_to_class(y_test)

    print(predictions)
    print(actual)

    print(confusion_matrix(actual, predictions))
