from __future__ import division, print_function

import numpy as np

from groundupml.supervised.binary_decision_tree import DecisionTree


if __name__ == '__main__':
    # TODO: Add testing
    # Demonstrate tree can learn XOR function
    X_train = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0]
                 ])

    y_train = np.array([0, 1, 1, 0, 0])
    dt = DecisionTree(max_depth=3, impurity_func='gini')
    dt.fit(X_train, y_train)
    print(dt)
    X_test = X_train
    pred = dt.predict(X_test)
    print(pred)
