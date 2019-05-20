from __future__ import division, print_function

import numpy as np


class LDA():
    """Linear Determinant Analysis Classifier

    NOTE: Currently only works for 2 classes.
    """
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        # For each class calculate the covariance matrix and mean of
        # the samples for that class

        # Separate the data by class
        X_C1 = X[y == 0]
        X_C2 = X[y == 1]

        # Calculate the mean of each class
        mean_c1 = X_C1.mean(axis=0)
        mean_c2 = X_C2.mean(axis=0)
        mean_diff = np.atleast_1d(mean_c1 - mean_c2)

        # Calculate the total covariance matrix for each class
        covariance_c1 = self._calculate_covariance_matrix(X_C1)
        covariance_c2 = self._calculate_covariance_matrix(X_C2)
        covariance_total = covariance_c1 + covariance_c2

        # Determine the weight vector w that maximizes the projected distance
        # between the total mean and the covariance for each class
        self.w = np.linalg.pinv(covariance_total).dot(mean_diff)

    def predict(self, X):
        y_pred = np.empty(len(X))
        for i, sample in enumerate(X):
            h = sample.dot(self.w)
            if h < 0:
                y = 1
            else:
                y = 0
            y_pred[i] = y

        return y_pred

    def _calculate_covariance_matrix(self, X):
        n_samples = X.shape[0]

        # Center X around its mean
        X = X - X.mean(axis=0)

        return X.T.dot(X) / (n_samples - 1)
