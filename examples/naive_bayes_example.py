from __future__ import division, print_function

from supervised.naive_bayes import NaiveBayes

import numpy as np

if __name__ == '__main__':
    # TODO: better testing
    X = np.array([[.75, .32, 1.8],
                  [1.76, 2.0, .22],
                  [.68, .36, 1.9]])
    y = np.array([1, 0, 1])

    test = np.array([[.7, .3, 2],
                     [1.7, 1.9, .2],
                     [.7, .3, 2.2]])

    nb = NaiveBayes()
    nb.fit(X, y)
    print(X)
    print(y)
    print(test)
    print(nb.predict(test))
