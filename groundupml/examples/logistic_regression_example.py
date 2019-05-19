from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from groundupml.supervised.logistic_regression import LogisticRegression
from groundupml.unsupervised.pca import PCA
from groundupml.utils.data_manipulation import split_data


if __name__ == '__main__':
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

    # Shuffle and split into test and training sets
    X_train, y_train, X_test, y_test = split_data(X, y, proportion=0.7)

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
