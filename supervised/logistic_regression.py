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
        threshold (:obj: `float`, optional):
            Cut-off value between 0 and 1 model confidence for labeling a
            prediction positive or negative
    """
    def __init__(self, num_iterations=100, learning_rate=0.001, threshold=0.5):
        self.num_iterations = num_iterations
        self.w = None
        self.learning_rate = learning_rate
        self.threshold = threshold

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
        # Compute soft predictions
        soft_predictions = self._sigmoid(np.dot(X, self.w))
        # Round model prediction to nearest class
        y_pred = _threshold_predictions(soft_predictions, self.threshold)

        return y_pred

    def _sigmoid(self, thetaTX):
        return 1.0 / (1.0 + np.exp(-thetaTX))


def _threshold_predictions(soft_predictions, threshold):
    return np.where(soft_predictions >= threshold, 1, 0)
