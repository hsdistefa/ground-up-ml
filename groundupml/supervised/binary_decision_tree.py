# coding: utf-8

from __future__ import division, print_function

import numpy as np

from groundupml.utils.functions import information_gain


class DecisionNode():
    """Binary Tree Decision Node

    Args:
        value (:obj: `float`, optional):
            Value of node if it is a leaf node
        feature_index (:obj: `int`, optional):
            Index of the feature that is split on for the node
        threshold (:obj: `float`, optional):
            Threshold value to split the feature on
        children (:obj: `list` of :obj: `DecisionNode`, optional):
            List of children nodes of the node
    """
    def __init__(self, value=None, feature_index=None, threshold=None,
                 children=None):
        self.value = value  # Value of node if it is a leaf node
        self.feature_index=feature_index  # Index of the feature to split branches
        self.threshold = threshold
        self.children = children

    def print_subtree(self, prefix, is_leaf):
        """Prints the subtree of the Node

        Args:
            prefix (:obj: `str`):
                Prefix to line up the text properly
            is_leaf (:obj: `bool`):
                Whether the node is a leaf node
        """
        # TODO: better visualization
        if self.value is not None:
            is_leaf = True
            name = str(self.value)  # Leaf value representation
        else:
            name = 'f' + str(self.feature_index)  # Nonleaf value representation
        res = prefix + ('└──' if is_leaf else '├──') + name + '\n'

        if self.children is None:  # Allow for empty children list
            self.children = []
        for i, child in enumerate(self.children):
            if i == len(self.children) - 1:  # Don't put vertical if last child
                res += child.print_subtree(
                    prefix + ('    ' if is_leaf else '│   '), True)
            else:
                res += child.print_subtree(
                    prefix + ('    ' if is_leaf else '│   '), False)

        return res



class DecisionTree():
    """Decision Tree

    Args:
        max_depth (:obj: `int`, optional):
            Maximum depth to allow the decision tree to be built to.
        impurity_func (:obj: `str`, optional):
            Name of the impurity function to use when calculating
            information gain. 'gini' or 'entropy'

    """
    def __init__(self, root=None, impurity_func='entropy', max_depth=None):
        self.root = root
        # NOTE: 'gini' or 'entropy'. Gini is slightly faster to compute,
        # entropy is more sensitive to changes in the distribution of
        # classes--use when class proportions are highly unbalanced
        self.impurity_func = impurity_func
        if max_depth is None:
            self.max_depth = float('inf')
        else:
            self.max_depth = max_depth
        self.n_samples = None
        self.n_features = None

    def __str__(self):
        return self.root.print_subtree('', True)


    def fit(self, X, y):
        """Builds a decision tree according to the training data

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Training data
            y (numpy array of shape [n_samples]):
                Training thresholds
        """
        self.n_samples, self.n_features = X.shape
        self.root = self.id3(X, y)  # Build the tree using id3 algorithm

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
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample, subtree=None):
        if subtree is None:
            subtree = self.root

        # If at a leaf, use that leaf's value
        if subtree.value is not None:
            return subtree.value

        # Traverse the branch path corresponding to the sample's values
        feature_value = sample[subtree.feature_index]
        if feature_value <= subtree.threshold:
            branch = subtree.children[0]
        else:
            branch = subtree.children[1]
        y_pred = self._predict_sample(sample, subtree=branch)

        return y_pred

    def id3(self, X, y, depth=0):
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
        # Ensure tree does not exceed max depth
        if depth > self.max_depth:
            leaf_value = self._compute_leaf_value(y)
            return DecisionNode(value=leaf_value)

        # Find feature split that would maximize information gain
        best_feature_i, best_threshold, y_best_splits = self._best_split(X, y)

        y_left_split, y_right_split = y_best_splits

        values = X[:, best_feature_i]
        X_left_split = X[values <= best_threshold]
        X_right_split = X[values > best_threshold]

        # If a split is empty, create a leaf with the remaining values
        if len(y_left_split) == 0 or len(y_right_split) == 0:
            return DecisionNode(value=self._compute_leaf_value(y))

        # Split on feature that gives max information gain
        left_branch = self.id3(X_left_split, y_left_split,
                               depth=depth+1)
        right_branch = self.id3(X_right_split, y_right_split,
                                depth=depth+1)

        return DecisionNode(feature_index=best_feature_i,
                            threshold=best_threshold,
                            children=[left_branch, right_branch])

    def _best_split(self, X, y):
        best_feature_i = None
        best_threshold = None
        max_info_gain = float('-inf')
        best_splits_y = [[],[]]

        # Find the feature and threshold that maximize info gain
        for feature_i in range(self.n_features):
            # Get the possible thresholds for splitting this feature
            values = X[:, feature_i]
            unique_values = np.unique(values)

            for threshold in unique_values:
                # Calculate info gain for splitting on each possible threshold
                y_left_split = y[values <= threshold]
                y_right_split = y[values > threshold]

                info_gain = information_gain([y_left_split, y_right_split],
                                             self.impurity_func)

                # Keep track of feature and threshold with the best info gain
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature_i = feature_i
                    best_threshold = threshold
                    best_splits_y = [y_left_split, y_right_split]

        return best_feature_i, best_threshold, best_splits_y

    @staticmethod
    def _compute_leaf_value(y):
        # Return the most common value
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 0:
            return None
        return unique[counts.argmax()]



if __name__ == '__main__':
    # TODO: Add testing
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
