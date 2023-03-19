# coding: utf-8

from __future__ import division, print_function

import numpy as np
import scipy.stats

from groundupml.utils.functions import information_gain


class DecisionNode():
    def __init__(self, value=None, feature_index=None, threshold=None, 
                 children=[]):
        self.value = value  # Value of node if it is a leaf node
        self.feature_index=feature_index  # Index of the feature to split branches
        self.threshold = threshold
        self.children = children

    def _print_subtree(self, prefix, is_leaf):
        # TODO: better visualization
        if self.value is not None:
            is_leaf = True
            name = str(self.value)  # Leaf value representation
        else:
            name = 'f' + str(self.feature_index)  # Nonleaf value representation
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
                Training thresholds
        """
        self.n_samples, self.n_features = X.shape
        self.root = self.id3(X, y)

    def predict(self, X):
        """Predict the thresholds of the given test data using the fitted decision
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
        feature_value = x[subtree.feature_index]
        if feature_value <= subtree.threshold:
            branch = subtree.children[0]
        else:
            branch = subtree.children[1]
        y_pred = self._predict_sample(x, subtree=branch)

        return y_pred

    def id3(self, X, y, features=None, depth=0):
        """Construct a decision tree using the ID3 algorithm, where each feature
        split is chosen by maximizing information gain.

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Train data
            y (numpy array of shape [n_samples]):
                Train thresholds

        Returns:
            N (DecisionNode):
                The root node of the fitted decision tree
        """
        # Instantiate feature list at tree root
        if features is None:
            # Store features as indices to reduce memory usage
            # Use mask to keep track of already split features in branch
            features = np.ma.array(np.arange(self.n_features), mask=False)

        # Ensure tree does not exceed max depth
        if depth > self.max_depth: 
            leaf_value = self._compute_leaf_value(y)
            return DecisionNode(value=leaf_value)
            
        # Find feature split that would maximize information gain
        best_feature_i, best_threshold, X_best_splits, y_best_splits = self._best_split(features,
                                                                        X, y)
        X_left_split, X_right_split = X_best_splits
        y_left_split, y_right_split = y_best_splits

        # If a split is empty, create a leaf with the remaining values
        if len(y_left_split) == 0 or len(y_right_split) == 0:
            return DecisionNode(value=self._compute_leaf_value(y))

        # Split on feature that gives max information gain
        features_copy = np.ma.array(features, copy=True)  

        left_branch = self.id3(X_left_split, y_left_split, 
                               features=features_copy, 
                               depth=depth+1)
        right_branch = self.id3(X_right_split, y_right_split, 
                                features=features_copy, 
                                depth=depth+1)

        branches = [left_branch, right_branch]

        return DecisionNode(feature_index=best_feature_i, 
                            threshold=best_threshold,
                            children=branches)

    def _best_split(self, features, X, y):
        best_feature_i = None
        best_threshold = None
        max_info_gain = float('-inf')
        best_splits_X = []
        best_splits_y = []

        # Find the feature and threshold that maximize info gain
        for feature_i, _ in enumerate(features):
            # Split on each feature only once per branch
            
            # Get the possible thresholds for splitting this feature
            values = X[:, feature_i]
            unique_values = np.unique(values) 

            for threshold in unique_values:
                # Calculate info gain for splitting on each possible threshold
                y_left_split = y[values <= threshold]
                y_right_split = y[values > threshold]

                X_left_split = X[values <= threshold]
                X_right_split = X[values > threshold]

                y_splits = [y_left_split, y_right_split]
                info_gain = information_gain(y_splits)

                # Keep track of feature and threshold with the best info gain
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature_i = feature_i
                    best_threshold = threshold
                    best_splits_X = [X_left_split, X_right_split]
                    best_splits_y = y_splits

        return best_feature_i, best_threshold, best_splits_X, best_splits_y

    def _compute_leaf_value(self, y):
        # Return the most common value
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 0:
            return None
        return unique[counts.argmax()]



if __name__ == '__main__':
    # TODO: Add testing
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