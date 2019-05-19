from __future__ import division, print_function

import numpy as np


class LDA():
    """Linear Determinant Analysis Classifier

    NOTE: Currently only works for 2 classes.
    """
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # For each class calculate the covariance matrix and mean of
        # the samples for that class
        classes = np.unique(y)
        covariance_total = np.zeros(shape=(n_features, n_features))
        mean_diff = np.zeros(shape=n_features)
        for class_n in classes:
            # Separate the data by class
            X_n = X[y == class_n]

            # Only need the total sum covariance matrix
            covariance = X_n.T.dot(X_n) / (n_samples - 1)
            covariance_total += covariance

            # Keep track of the differences between the means
            mean_n = X_n.mean(axis=0)
            mean_diff = np.atleast_1d(mean_diff - mean_n)

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
