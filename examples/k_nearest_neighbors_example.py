from __future__ import division, print_function

from supervised.k_nearest_neighbors import KNN
from unsupervised.pca import PCA

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


if __name__ == '__main__':
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
