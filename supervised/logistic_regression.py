from __future__ import division, print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

# FIX: uncomment when not testing from __main__
#from ..unsupervised.pca import PCA
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/../unsupervised/')
from pca import PCA


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
        # Add bias weights to input
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
                Predicted values from test data
        """
        # Add bias weights to input
        X = np.insert(X, 0, 1, axis=1)
        # Round model prediction to nearest class
        return np.round(self._sigmoid(X.dot(self.w))).astype(np.int32)

    def _sigmoid(self, thetaTX):
        return 1 / (1 + np.exp(-thetaTX))


def test():
    np.random.seed(seed=10)

    # Load dataset and normalize
    iris = sklearn.datasets.load_iris()
    data = iris.data[iris.target != 0]
    l2 = np.atleast_1d(np.linalg.norm(data, 2, 1))
    l2[l2 == 0] = 1
    X = data / np.expand_dims(l2, 1)

    y = iris.target[iris.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    # Shuffle data
    y = y[np.newaxis].T
    stacked = np.hstack((X, y))
    np.random.shuffle(stacked)
    X = stacked[:, :stacked.shape[1]-1]
    y = stacked[:, stacked.shape[1]-1]

    # Split into test and training sets
    split_index = int(.7*len(data))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Run the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute the model error
    error = np.mean(np.abs(y_test - y_pred))
    accuracy = 1 - error

    # Plot the results
    pca = PCA()
    X_reduced = pca.transform(X_test, 2)
    pc1 = X_reduced[:, 0]
    pc2 = X_reduced[:, 1]
    plt.scatter(pc1, pc2, c=y_pred)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Logistic Regression (Accuracy: {:2f})'.format(accuracy))
    plt.show()


if __name__ == '__main__':
    test()
