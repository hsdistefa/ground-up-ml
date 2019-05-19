from __future__ import division, print_function

import numpy as np

from groundupml.supervised.decision_tree import DecisionTree


if __name__ == '__main__':
    # TODO: better testing
    # AND function
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    test = np.array([[0, 1], [1, 1], [1, 0], [1, 0], [0, 0], [1, 1]])

    print('train input', X)
    dt = DecisionTree()
    dt.fit(X, y)
    print('train labels', y)
    print('test input', test)
    print(dt)
    print('prediction', dt.predict(test))
