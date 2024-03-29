from __future__ import print_function, division

import numpy as np

from groundupml.utils.functions import cross_entropy, sigmoid, sigmoid_prime, \
                                       softmax


class SigmoidLayer():
    """Sigmoid Layer

    Args:
        n_nodes (int):
            Number of nodes in layer.
        n_inputs (int):
            Number of inputs to layer.
    """
    def __init__(self):
        self.inputs = None
        self.activations = None
    
    def forward_propogate(self, X):
        """Propogate input forward through layer

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Input data

        Returns:
            numpy array of shape [n_classes, 1]:
                Activated output where values correspond to sigmoid activations
        """
        self.inputs = X
        self.activations = sigmoid(X)

        return self.activations

    def back_propogate(self, gradient):
        """Propogate error gradient backward through layer using chain rule

        Args:
            gradient (numpy array of shape [n_samples, n_nodes]):
                Error gradient

        Returns:
            numpy array of shape [n_samples, n_inputs]:
                Error gradient for layer below
        """
        # Derivative of sigmoid function is sigmoid(x) * (1 - sigmoid(x))
        # We can use the cached activations to save computation
        return gradient * sigmoid_prime(self.activations)

class ReLuLayer():
    """Applies the rectified linear unit function element-wise. 
    
    ReLu activation functions act as a thresholding mechanism, where all values
    below 0 are set to 0. ReLu layers can be used to add nonlinearities while
    avoiding the vanishing gradient problem that arises with the sigmoid and
    hyperbolic tangent activation functions in deep neural networks.
    """
    def __init__(self):
        self.inputs = None
        self.activations = None
    
    def forward_propogate(self, X):
        """Propogate input forward through layer

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Input data

        Returns:
            numpy array of shape [n_classes, 1]:
                Activated output where values correspond to sigmoid activations
        """
        self.inputs = X
        self.activations = np.maximum(0.0, X)  # Use 0.0 to avoid int division

        return self.activations

    def back_propogate(self, gradient):
        """Propogate error gradient backward through layer using chain rule

        Args:
            gradient (numpy array of shape [n_samples, n_nodes]):
                Error gradient

        Returns:
            numpy array of shape [n_samples, n_inputs]:
                Error gradient for layer below
        """
        # Derivative of relu function is 1 if x > 0, 0 otherwise
        # We can use the cached activations to save computation
        return gradient * (self.activations > 0)


class SoftmaxLayer():
    """Softmax Layer

    Softmax activation functions are used in the output layer of a neural
    network to get a probability distribution over the classes. The outputs
    can be interpreted as the probability that the neural network 'believes'
    a sample belongs to each possible class. For example, 2% chance of being a
    duck, 18% chance of being a dog, 80% chance of being a cat. Notice that the
    sum of the probabilities is 1.
    """
    def __init__(self):
        self.inputs = None
        self.activations = None
    
    def forward_propogate(self, X):
        """Propogate input forward through layer

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Input data

        Returns:
            numpy array of shape [n_classes, 1]:
                Activated output where values correspond to softmax activations
        """
        self.inputs = X
        self.activations = softmax(X)

        return self.activations

    def back_propogate(self, gradient):
        """Propogate error gradient backward through layer using chain rule

        Args:
            gradient (numpy array of shape [n_samples, n_nodes]):
                Error gradient

        Returns:
            numpy array of shape [n_samples, n_inputs]:
                Error gradient for layer below
        """
        # Derivations of softmax function Jacobian matrix can be found here:
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # NOTE: We save some computation by caching activations
        # NOTE: Vectorized implementation
        # Does the same thing as constructing the jacobian matrix like this:
        # For each softmax vector sm:
        #     jacobian_m = np.diag(sm)
        #     for i in range(jacobian_m.shape[0]):
        #         for j in range(jacobian_m.shape[1]):
        #             if i == j:
        #                 jacobian_m[i][j] = sm[i] * (1 - sm[i])
        #             else:
        #                 jacobian_m[i][j] = -sm[i] * sm[j]
        # TODO: Backpropogate Jacobians
        # TODO: Test gradients
        #I = np.eye(self.activations.shape[-1])
        #sm = self.activations
        #d_softmax = np.einsum('ij, jk -> ijk', sm, I) - \
        #    np.einsum('ij, ik -> ijk', sm, sm)
        # NOTE: This is only the diagonal of the Jacobian matrix
        d_softmax = self.activations * (1 - self.activations)

        return gradient * d_softmax

class SoftmaxCrossEntropyLayer():
    """Softmax Cross Entropy Layer

    This layer is intented for use **only** as the final layer in a neural
    network. It combines the softmax activation function with cross entropy
    loss. This combination makes calculation of the gradients much more
    simple and efficient than when using a general softmax layer separately.
    
    These two functions are very commonly used together in multiclass
    classification because the softmax creates a distribution (where
    probabilities sum to 1) over the classes, and the cross entropy loss
    requires a probability distribution as input.
    """
    def __init__(self):
        self.activations = None
        self.cross_entropy_loss = None
        self.targets = None

    def forward_propogate(self, X):
        """Propogate input forward through layer

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Input data

        Returns:
            numpy array of shape [n_classes, 1]:
                Activated output where values correspond to softmax activations
        """
        self.activations = softmax(X)

        return self.activations

    def back_propogate(self, targets):
        """Propogate error gradient backward through layer using chain rule

        Args:
            targets (numpy array of shape [n_samples, n_classes]):
                One-hot encoded target classes

        Returns:
            numpy array of shape [n_samples, n_inputs]:
                Error gradient for layer below
        """
        # Gradient derived here: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        return (self.activations - targets) / self.activations.shape[0]


    def calculate_loss(self, actual):
        """Calculates the cross entropy loss for the current activations

        Args:
            targets (numpy array of shape [n_samples, n_classes]):
                One-hot encoded target classes

        Returns:
            numpy array of shape [n_samples, 1]:
                Cross entropy loss for each sample
        """
        self.cross_entropy_loss = cross_entropy(self.activations, actual, 
                                                reduction='mean')

        return self.cross_entropy_loss
