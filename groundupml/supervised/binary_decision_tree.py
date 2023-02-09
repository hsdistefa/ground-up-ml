# coding: utf-8

from __future__ import division, print_function

import numpy as np
import scipy.stats

#from groundupml.utils.functions import information_gain


class DecisionNode():
    def __init__(self, value=None, feature_index=None, children=[]):
        self.value = value
        self.feature_index=feature_index
        self.children = children

    def _print_subtree(self, prefix, is_leaf):
        # TODO: better visualization
        if self.value is not None:
            is_leaf = True
            name = str(self.value)  # Leaf value representation
        else:
            name = 'f' + str(' feature x')  # Nonleaf value representation
        res = prefix + ('└──' if is_leaf else '├──') + name + '\n'

        for i, child in enumerate(self.children):
            if i == len(self.children) - 1:  # Don't put vertical if last child
                res += child._print_subtree(
                    prefix + ('    ' if is_leaf else '│   '), True)
            else:
                res += child._print_subtree(
                    prefix + ('    ' if is_leaf else '│   '), False)

        return res



class DecisionTree():
    """Decision Tree

    Args:
        max_depth (:obj: `int`, optional):
            Maximum depth to allow the decision tree to be built to.

    """
    def __init__(self, root=None, max_depth=None):
        self.root = root

        if max_depth is None:
            self.max_depth = float('inf')
        else:
            self.max_depth = max_depth

    def __str__(self):
        return self.root._print_subtree('', True)


    def fit(self, X, y):
        """Builds a decision tree according to the training data

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Training data
            y (numpy array of shape [n_samples]):
                Training labels
        """
        self.n_samples, self.n_features = X.shape
        self.root = self.id3(X, y)

    def predict(self, X):
        """Predict the labels of the given test data using the fitted decision
        tree

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Test data

        Returns:
            C (numpy array of shape [n_samples]):
                Predicted values from test data
        """
        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, x, subtree=None):
        if subtree is None:
            subtree = self.root

        # If at a leaf, use that leaf's value
        if subtree.value is not None:
            return subtree.value

        # Traverse the branch path corresponding to the sample's values
        feature_label = x[subtree.feature_index]
        y_pred = self._predict_sample(x, subtree=subtree.children[feature_label])

        return y_pred

    def id3(self, X, y, features=None, depth=0):
        """Construct a decision tree using the ID3 algorithm, where each feature
        split is chosen by maximizing information gain.

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Train data
            y (numpy array of shape [n_samples]):
                Train labels

        Returns:
            N (DecisionNode):
                The root node of the fitted decision tree
        """
        depth += 1

        # Instantiate feature list at tree root
        if features is None:
            # Store features as indices to reduce memory usage
            # Use mask to keep track of already split features
            features = np.ma.array(np.arange(self.n_features), mask=False)

        # Create leaf nodes either at max tree depth or when all
        # features have been split on
        if depth > self.max_depth or np.all(features.mask == True):
            leaf_value = self._compute_leaf_value(y)
            return DecisionNode(value=leaf_value)
            
        # Find feature split that would maximize information gain
        max_gain = float('-inf')
        for feature_i, _ in enumerate(features):
            # Split on each feature only once per branch
            if features.mask[feature_i] == True:  
                continue

            # Calculate the probability of seeing each value of feature i
            values = X[:, feature_i]

            _, value_counts = np.unique(values, return_counts=True)
            probs = value_counts / np.float32(len(values))

            # Calculate information gain
            info_gain = 1 + np.sum(probs*np.log2(probs))
            if info_gain > max_gain:
                max_gain = info_gain
                best_feature_i = feature_i

        # Split on feature that gives max information gain
        features.mask[best_feature_i] = True  # Mark feature as used
        # Each branch needs a separate copy of the feature mask
        features_copy = np.ma.array(features, copy=True)  

        X_left_split = X[values == 0]
        y_left_split = y[values == 0]

        X_right_split = X[values == 1]
        y_right_split = y[values == 1]

        # If a split is empty, create a leaf with the only remaining label
        if len(y_left_split) == 0 or len(y_right_split) == 0:
            return DecisionNode(value=self._compute_leaf_value(y))

        left_branch = self.id3(X_left_split, y_left_split, 
                               features=features_copy, depth=depth)
        right_branch = self.id3(X_right_split, y_right_split, 
                                features=features_copy, depth=depth)

        branches = [left_branch, right_branch]

        return DecisionNode(feature_index=best_feature_i, children=branches)

    def _compute_leaf_value(self, labels):
        # Return the most common value
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 0:
            return None
        return unique[counts.argmax()]


if __name__ == '__main__':
    X = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0]
                 ])

    y = np.array([0, 1, 1, 0, 0])
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    print(dt)
    X_test = X
    pred = dt.predict(X_test)
    print(pred)