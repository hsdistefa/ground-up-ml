import numpy as np


class Layer(object):
    """Base class for a neural network Layer class.
    """
    def set_n_inputs(self, n_inputs):
        self.n_inputs = n_inputs

    def set_learning_rate(self, learning_rate):
        self.learning_rate = None

    def forward_propogate(self, X):
        raise NotImplementedError()

    def back_propogate(self, gradient):
        raise NotImplementedError()


class FullyConnected(Layer):
    """Neural network layer where each node is connected to every node in
    the next layer

    Args:
        n_nodes (int):
            The number of neural nodes to include in the layer
        n_inputs (:obj: `int`, optional):
            The number of nodes in the previous layer or if the layer is an
            input layer, the number of features in the input data. Required
            argument when adding the first layer, i.e. the input layer.
    """
    def __init__(self, n_nodes, n_inputs=None):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.weights, self.bias = self._init_weights()
        self.learning_rate = None
        self.inputs = None

    # Override superclass method
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def forward_propogate(self, X):
        # Gives the output of each neuron to be sent to the next layer
        self.inputs = X
        outputs = np.dot(X, self.weights) + self.bias

        return outputs

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

    Args:
        n_nodes (int):
            Number of neural nodes to include in the layer
        activation_function (:obj: `ActivationFunction`, optional):
            Function to use for activation computations.
    """
    def __init__(self, n_nodes, activation_func=Sigmoid):
        self.n_nodes = n_nodes
        self.activation_func = activation_function
        self.inputs = None

    def forward_propogate(self, X):
        # Gives the activation output of each neuron to be sent to the next
        # layer
        self.inputs = X
        activations = self.activation_function(self.inputs)

        return activations

    def back_propogate(self, gradient):
        # Calculate activation gradient
        a_gradient = gradient * self.activation_func.gradient(self.inputs)

        return a_gradient
