from __future__ import division, print_function

import numpy as np


class LogisticRegression():
    """Logistic Regression

    Args:
        num_iterations (:obj: `int`, optional):
            Number of iterations to do using gradient descent
        learning_rate (:obj: `float`, optional):
            Step magnitude used for updating the weights during gradient
            descent.
    """
    def __init__(self, num_iterations=100, learning_rate=.001):
        self.num_iterations = num_iterations
        self.w = None
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit given training data to a logistic model using regression

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Training data
            y (numpy array of shape [n_samples]):
                Training labels
        """
        # Add bias term to input
        X = np.insert(X, 0, 1, axis=1)

        n_samples, n_features = np.shape(X)

        # Initialize weights
        self.w = np.random.randn(n_features)

        for _ in range(self.num_iterations):
            # Gradient descent
            h_x = self._sigmoid(X.dot(self.w))
            gradient = X.T.dot(h_x - y)
            self.w -= self.learning_rate * gradient
            # TODO: implement batch optimization

    def predict(self, X):
        """Predict given test data using the logistic model

        Args:
            X (numpy array of shape [t_samples, n_features]):
                Test data

        Returns:
            C (numpy array of shape [n_samples]):
                Predicted class labels for test data
        """
        # Add bias weights to input
        X = np.insert(X, 0, 1, axis=1)
        # Round model prediction to nearest class
        return np.round(self._sigmoid(X.dot(self.w))).astype(np.int32)

    def _sigmoid(self, thetaTX):
        return 1.0 / (1.0 + np.exp(-thetaTX))
