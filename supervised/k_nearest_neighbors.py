from __future__ import division, print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.datasets

# FIX: uncomment when not testing from __main__
#from ..unsupervised.pca import PCA
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/../unsupervised/')
from pca import PCA


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
                dist = self._euclidean_distance(test_sample, train_sample)
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

    def _euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)


def test():
    np.random.seed(seed=7778)

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
    knn = KNN(2, 'class')
    y_pred = knn.predict(X_train, y_train, X_test)

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
    plt.title('K Nearest Neighbors (Accuracy: {:2f})'.format(accuracy))
    plt.show()


if __name__ == '__main__':
    test()
