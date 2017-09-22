import numpy as np


class Layer(object):
    def set_n_inputs(self, n_inputs):
        self.n_inputs = n_inputs

    def forward_propogate(self, X):
        raise NotImplementedError()

    def back_propogate(self, gradient):
        raise NotImplementedError()


class FullyConnected(Layer):
    """Neural network layer where each node is connected to every node in
    the next layer
    """
    def __init__(self, n_nodes, n_inputs=None):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.weights, self.bias = self._init_weights()
        self.learning_rate = None
        self.inputs = None
        self.outputs = None

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def forward_propogate(self, X):
        # Gives the output of each neuron to be sent to the next layer
        self.inputs = X
        self.outputs = np.dot(X, self.weights) + self.bias

        return self.outputs

    def back_propogate(self, gradient):
        # Calculate gradients w.r.t. layer weights
        w_gradient = np.dot(self.inputs.T, gradient)
        b_gradient = np.sum(gradient, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights -= self.learning_rate * w_gradient
        self.bias -= self.learning_rate * b_gradient

        return np.dot(gradient, self.weights.T)

    def _init_weights(self):
        # Initialize weights
        # Add bias weight for input to each neuron
        W = np.random.randn(self.n_inputs, self.n_nodes) / \
            np.sqrt(self.n_inputs)
        b = np.zeros((1, self.n_nodes))

        return W, b


class Activation(Layer):
    """Neural network layer that computes an activation function on its input
    before passing it on to the next layer
    """
    def __init__(self, n_nodes, n_inputs=None):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.activations = None

    def forward_propogate(self, X):
        # Gives the activation output of each neuron to be sent to the next
        # layer
        self.activations = _sigmoid(X)

        return self.activations

    def back_propogate(self, gradient):
        # Calculate activation gradient
        a_gradient = gradient * _sigmoid_derivative(self.activations)

        return a_gradient


def _sigmoid_derivative(z):
    return _sigmoid(z) * (1.0 - _sigmoid(z))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
