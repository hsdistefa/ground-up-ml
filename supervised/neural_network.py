from __future__ import print_function, division

import numpy as np


class NNLayer():
    """A fully connected activation layer
    """
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights, self.bias = self._init_weights()
        self.inputs = None
        self.outputs = None
        self.activations = None

    def get_activations(self, X):
        # Gives the output of each neuron to be sent to the next layer
        self.inputs = X
        self.outputs = np.dot(X, self.weights) + self.bias
        self.activations = _sigmoid(self.outputs)

        return self.activations

    def _init_weights(self):
        # Initialize weights
        # Add bias weight for input to each neuron
        W = np.random.randn(self.n_inputs, self.n_outputs) / \
            np.sqrt(self.n_inputs)
        b = np.zeros((1, self.n_outputs))

        return W, b


class NeuralNetwork():
    def __init__(self, learning_rate=.01):
        self.learning_rate = learning_rate
        self.network = []

    def add(self, n_inputs, n_outputs):
        self.network.append(NNLayer(n_inputs, n_outputs))

    def fit(self, X, y, n_epochs):
        self.n_samples, self.n_features = np.shape(X)
        self.n_output_nodes = len(np.unique(y))

        fs = 'epoch {}, learning rate: {}, error: {:.3f}'

        # Convert training labels to one-hot encoding
        y = _to_1_hot(y)

        for epoch_i in range(n_epochs):
            # Get predictions by propogating input forward through network
            predictions = self._forward_propogate(X)

            # Propogate error gradient backward through network using chain rule
            gradient = -(y - predictions)  # Square loss gradient
            self._back_propogate(gradient)

            # Calculate and print epoch error
            hard_pred = np.zeros_like(predictions)
            hard_pred[np.arange(len(predictions)), predictions.argmax(1)] = 1
            error = 1 - (np.sum(np.all(y == hard_pred, axis=1)) / float(len(y)))

            print(fs.format(epoch_i+1, self.learning_rate, error))

    def predict(self, X):
        """Returns 1-hot encoded class labels predicted by the neural network
        """
        return self._forward_propogate(X)

    def _back_propogate(self, gradient):
        # Iterate backward over all but output layer and update weights using
        # chain rule
        curr_gradient = gradient
        for layer in reversed(self.network):
            # Calculate activation gradient
            a_gradient = curr_gradient * _sigmoid_derivative(layer.outputs)

            # Calculate gradients w.r.t. layer weights
            w_gradient = np.dot(layer.inputs.T, a_gradient)
            b_gradient = np.sum(curr_gradient, axis=0, keepdims=True)

            # Update weights and biases using gradient descent
            layer.weights -= self.learning_rate * w_gradient
            layer.bias -= self.learning_rate * b_gradient

            # Calculate gradient to be passed to next layer
            curr_gradient = np.dot(curr_gradient, layer.weights.T)

    def _forward_propogate(self, X):
        # Take input and send forward through each layer to get output
        layer_output = X
        for layer in self.network:
            # Calculate activations to feed to next layer
            layer_output = layer.get_activations(layer_output)
        return layer_output

    def _gradient_check(self, weights, gradients):
        # TODO: gradient checking
        pass


def _to_1_hot(a):
    one_hot = np.zeros((a.size, a.max()+1))  # Might be slow for large input
    one_hot[np.arange(a.size), a] = 1

    return one_hot


def _sigmoid_derivative(z):
    return _sigmoid(z) * (1.0 - _sigmoid(z))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test():
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

    nn.add(np.shape(X_train)[1], n_hidden_nodes)
    nn.add(n_hidden_nodes, len(np.unique(y_train)))

    # Train network and get test predictions
    nn.fit(X_train, y_train, 200)
    print('predictions: ', nn.predict(X_test))
    print('actual: ', y_test)

if __name__ == '__main__':
    test()
