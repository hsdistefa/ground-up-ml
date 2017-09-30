from __future__ import division, print_function

import numpy as np
import scipy.stats

from utils.functions import euclidean_distance


class KNN():
    def __init__(self, k, method):
        """K Nearest Neighbors

        Args:
            k (int):
                Number of neighbors to use when evaluating test points
            method (str):
                class: Classification
                reg:   Regression
        """
        self.k = k
        self.method = method

    def predict(self, X_train, y_train, X_test):
        """Using the k nearest neighbors in the given training data, lazy
        evaluate the value of the given test data

        Args:
            X_train (numpy array of shape [n_samples, n_features]):
                Training data
            y_train (numpy array of shape [n_samples]):
                Training labels
            X_test (numpy array of shape [n_samples, n_features]):
                Test data

        Returns:
            C (numpy array of shape [n_samples]):
                Predicted values from test data
        """
        # TODO: implement Kernel Regression (locally weighted)
        # For each sample in the test set, find the k closest samples in the
        # train set and the corresponding labels
        labels_matrix = np.empty((X_test.shape[0], self.k))
        labels_matrix[:] = np.NAN
        for (i, test_sample) in enumerate(X_test):
            knn_dist = np.full(self.k, np.inf)
            knn_labels = np.empty(self.k)
            knn_labels[:] = np.NAN

            max_neighbor_i = 0
            for (j, train_sample) in enumerate(X_train):
                dist = euclidean_distance(test_sample, train_sample)
                if dist < knn_dist[max_neighbor_i]:
                    knn_dist[max_neighbor_i] = dist
                    knn_labels[max_neighbor_i] = y_train[j]
                    max_neighbor_i = np.argmax(knn_dist)
            labels_matrix[i] = knn_labels

        if self.method == 'class':
            # Take the majority vote of each of the k closest train samples to
            # determine the label to apply to each of the test samples
            predictions, _ = scipy.stats.mode(labels_matrix, axis=1)
            predictions = predictions.T

        elif self.method == 'reg':
            # Take the average of each of the k closest train samples to
            # determine the values to apply to the test samples.
            predictions = np.mean(labels_matrix, axis=1)

        return predictions
