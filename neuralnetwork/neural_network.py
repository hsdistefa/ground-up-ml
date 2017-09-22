from __future__ import print_function, division

import numpy as np


class NeuralNetwork():
    def __init__(self, learning_rate=.01):
        self.learning_rate = learning_rate
        self.network = []

    def add(self, layer):
        if self.network:
            prev_layer_n_nodes = self.network[-1].n_nodes
            layer.set_n_inputs(prev_layer_n_nodes)

        if hasattr(layer, 'learning_rate'):
            layer.set_learning_rate(self.learning_rate)

        self.network.append(layer)

    def fit(self, X, y, n_epochs):
        self.n_samples, self.n_features = np.shape(X)
        self.n_output_nodes = len(np.unique(y))

        # Convert training labels to one-hot encoding
        y = _to_1_hot(y)

        for epoch_i in range(n_epochs):
            # Get predictions by propogating input forward through network
            predictions = self._forward_propogate(X)

            # Propogate error gradient backward through network using chain rule
            gradient = -(y - predictions)  # Square loss gradient
            self._back_propogate(gradient)

            # Calculate and print epoch error
            self._epoch_summary(epoch_i, y, predictions)

    def predict(self, X):
        """Returns 1-hot encoded class labels predicted by the neural network
        """
        return self._forward_propogate(X)

    def _back_propogate(self, gradient):
        # Iterate backward over all but output layer and update weights using
        # chain rule
        curr_gradient = gradient
        for layer in reversed(self.network):
            # Calculate gradient to be passed to next layer
            curr_gradient = layer.back_propogate(curr_gradient)

    def _forward_propogate(self, X):
        # Take input and send forward through each layer to get output
        layer_output = X
        for layer in self.network:
            # Calculate activations to feed to next layer
            layer_output = layer.forward_propogate(layer_output)
        return layer_output

    def _epoch_summary(self, epoch_i, y, predictions):
        fs = 'epoch {}, learning rate: {}, error: {:.3f}'

        hard_pred = np.zeros_like(predictions)
        hard_pred[np.arange(len(predictions)), predictions.argmax(1)] = 1
        error = 1 - (np.sum(np.all(y == hard_pred, axis=1)) / float(len(y)))

        print(fs.format(epoch_i+1, self.learning_rate, error))

    def _gradient_check(self, weights, gradients):
        # TODO: gradient checking
        pass


def _to_1_hot(a):
    one_hot = np.zeros((a.size, a.max()+1))  # Might be slow for large input
    one_hot[np.arange(a.size), a] = 1

    return one_hot
