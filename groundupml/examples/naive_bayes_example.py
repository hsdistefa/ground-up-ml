from __future__ import division, print_function

import numpy as np
import sklearn.datasets

from groundupml.supervised.naive_bayes import NaiveBayes
from groundupml.utils.data_manipulation import split_data


if __name__ == '__main__':
    np.random.seed(seed=2)

    # Load dataset
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    # Shuffle and split data
    X_train, y_train, X_test, y_test = split_data(X, y)

    # Run the model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    # Compute error
    error = np.sum(y_test != y_pred) / float(len(y_test))
    accuracy = 1 - error
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
