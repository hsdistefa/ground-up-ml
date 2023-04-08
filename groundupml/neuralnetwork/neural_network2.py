from __future__ import print_function, division

import numpy as np


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
        # NOTE: Uniform initialization
        # Scale weights by 1/sqrt(n inputs) to help prevent exploding gradients
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

class EmbeddingLayer():
    """Embedding layer that applies a linear transformation to the input:
    :math:`y = XW^T`
    Where W is the learnable weights matrix, b is the learnable biases vector, 
    and X is the categorical input encoded as a 1-hot vector.

    Essentially the weights act as embeddings for each input. As these weights
    are updated during training, the embeddings are updated as well. This
    allows the network to learn the best embeddings for the task at hand.

    Embedding layers are used instead of the similar linear layers because the
    input X is encoded as a one-hot vector, so we avoid matrix multiplication
    by simply selecting the row of W corresponding to the index of the one-hot
    vector as if it is a lookup table. This is equivalent to :math:`y=XW^T`.

    Args:
        vocab_size (int):
            Size of the dictionary of embeddings.
        embedding_size (int):
            The size of each embedding vector.
        learning_rate (float, optional):
            Step magnitude used for updating layer weights when relevant.
    """
    def __init__(self, vocab_size, embedding_size, learning_rate=.01):
        # TODO: Remove unnecessary attributes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.inputs = None
        self.weights = None  # Embedding weights
        self.d_weights = None # Gradient of loss with respect to weights


    def init_weights(self):
        """Initialize embedding weights"""
        self.weights = np.random.randn(self.vocab_size, self.embedding_size)

    def forward(self, X):
        self.inputs = X  # Save inputs for backpropogation step

        # Lookup embedding for each input
        batch_size, seq_length = X.shape
        embedding_output = np.zeros((batch_size, seq_length, self.embedding_size))
        for i in range(batch_size):
            for j in range(seq_length):
                embedding_output[i, j, :] = self.weights[X[i, j], :]

        return embedding_output

    def backward(self, gradient):
        batch_size, seq_length, embedding_size = gradient.shape

        # Gradient of loss with respect to embedding weights
        self.d_weights = np.zeros((self.vocab_size, self.embedding_size))
        for i in range(batch_size):
            for j in range(seq_length):
                self.d_weights[self.inputs[i, j], :] += gradient[i, j, :]

        # Gradient of loss with respect to inputs
        d_X = np.zeros(self.inputs.shape)
        for i in range(batch_size):
            for j in range(seq_length):
                #gradient[i, j, :]
                d_X[i, j] = np.dot(self.embedding_weights.T, gradient[i, j, :])

        return d_X

    def update_weights(self):
        """Update weights"""
        self.weights -= self.learning_rate * self.d_weights

    def __repr__(self):
        return 'EmbeddingLayer(vocab_size={}, embedding_size={}, learning_rate={})'.format(
            self.vocab_size, self.embedding_size, self.learning_rate)
