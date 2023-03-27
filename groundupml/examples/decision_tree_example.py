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
                        [0, 0]])

    y_train = np.array([0, 1, 1, 0, 0])
    dt = DecisionTree(max_depth=3, impurity_func='gini')
    dt.fit(X_train, y_train)
    print(dt)
    X_test = X_train
    pred = dt.predict(X_test)
    print(pred)

    # Try other training sets
    # Feature 1 is number of legs, feature 2 is number of eyes
    # Class 0 is a person (2 legs, 2 eyes), class 1 is a dog (4 legs, 2 eyes),
    # Class 3 is a spider (8 legs, 8 eyes)
    X_train = np.array([[2, 2],
                        [4, 2],
                        [8, 8]])

    y_train = np.array(['person', 'dog', 'spider'])
    dt = DecisionTree(max_depth=3, impurity_func='gini')
    dt.fit(X_train, y_train)
    print(dt)
    X_test = X_train
    pred = dt.predict(X_test)
    print(pred)
