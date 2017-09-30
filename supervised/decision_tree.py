# coding: utf-8

from __future__ import division, print_function

import numpy as np
import scipy.stats

from utils.functions import information_gain


class DecisionNode():
    def __init__(self, feature_index=None, children=[], value=None):
        self.feature_index = feature_index
        self.children = children
        self.value = value

    def _print_tree(self, prefix, is_leaf):
        # TODO: better visualization
        if self.value is not None:
            is_leaf = True
            name = str(self.value)  # Leaf value representation
        else:
            name = 'f' + str(self.feature_index)  # Nonleaf value representation
        res = prefix + ('└──' if is_leaf else '├──') + name + '\n'

        for i, child in enumerate(self.children):
            if i == len(self.children) - 1:  # Don't put vertical if last child
                res += child._print_tree(
                    prefix + ('    ' if is_leaf else '│   '), True)
            else:
                res += child._print_tree(
                    prefix + ('    ' if is_leaf else '│   '), False)

        return res


class DecisionTree():
    """Decision Tree

    Args:
        max_depth (:obj: `int`, optional):
            Maximum depth to allow the decision tree to be built to.

    """
    def __init__(self, max_depth=None, root=None):
        self.root = root
        if max_depth is None:
            self.max_depth = float('inf')
        else:
            self.max_depth = max_depth

    def __str__(self):
        return self.root._print_tree('', True)

    def fit(self, X, y):
        """Builds a decision tree according to the training data

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Training data
            y (numpy array of shape [n_samples]):
                Training labels
        """
        self.n_samples, self.n_features = np.shape(X)
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
        predictions = np.zeros(np.shape(X)[0])
        for i, sample in enumerate(X):
            predictions[i] = self.predict_sample(sample)

        return np.array(predictions)

    def predict_sample(self, x, subtree=None):
        if subtree is None:
            subtree = self.root

        # If at a leaf, use that leaf's value
        if subtree.value is not None:
            return subtree.value

        feature_label = x[subtree.feature_index]
        y_pred = self.predict_sample(x, subtree=subtree.children[feature_label])

        return y_pred

    def id3(self, X, y, features=None, depth=0):
        """Construct a decision tree using the ID3 algorithm, where each feature
        split is chosen by maximizing information gain.

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Test data
            y (numpy array of shape [n_samples]):
                Test labels

        Returns:
            N (DecisionNode):
                The root node of the fitted decision tree
        """
        depth += 1

        # Instantiate feature list at root
        if features is None:
            features = np.ma.array(np.arange(self.n_features), mask=False)

        # If all features used or max depth reached, return majority vote of
        # labels as leaf node
        if np.all(features.mask == True) or depth > self.max_depth:
            leaf_value = scipy.stats.mode(y)[0][0]
            return DecisionNode(value=leaf_value)

        # Find feature split that maximizes information gain
        max_gain = float('-inf')
        best_feature = None
        best_splits_X = None
        best_splits_y = None
        for feature_i, feature in enumerate(features):
            # If a feature has already been split on don't use it
            if features.mask[feature_i] == True:
                continue

            values = X[:, feature_i]  # All values for given feature
            labels = np.unique(values)  # Unique values for given feature

            # Split into bins based on each possible feature value
            splits = [y[values == label] for label in labels]

            # Calculate information gain for splitting on feature
            info_gain = information_gain(splits)

            # Update maximum information gain
            if info_gain > max_gain:
                max_gain = info_gain
                best_feature = feature_i
                best_splits_X = [X[values == label] for label in labels]
                best_splits_y = splits

        # Split on the feature that maximizes information gain
        features.mask[best_feature] = True  # Mark feature as used
        branches = []
        for i in range(len(best_splits_y)):
            # Copy mask so that each branch has it's own copy, i.e. they don't
            # all reference the same list
            features_copy = np.ma.array(features, copy=True)

            branch = self.id3(best_splits_X[i], best_splits_y[i],
                              features=features_copy, depth=depth)
            branches.append(branch)

        return DecisionNode(feature_index=best_feature, children=branches)
