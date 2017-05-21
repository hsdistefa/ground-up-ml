from __future__ import division, print_function

import math

import numpy as np
import scipy.stats


class DecisionNode():
    def __init__(self, feature_index=None, children=None, value=None):
        self.feature_index = feature_index
        self.children = children
        self.value = value


class DecisionTree():
    def __init__(self, max_depth, root):
        #self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        # Build tree
        self.root = self.id3(X, y)

    def predict(self, X):
        n_samples, n_features = np.shape(X)

        predictions = np.zeros(n_samples)
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
        y_pred = self.predict(x, subtree=subtree.children[feature_label])

        return y_pred

    def id3(self, X, y, features=None):
        # If all features used, return majority vote of labels as leaf node
        if np.all(features.mask is True):
            return DecisionNode(value=scipy.stats.mode(y)[0][0])

        n_samples, n_features = X.shape

        # Instantiate feature list if not specified
        if features is None:
            features = np.ma.array(np.arange(n_features), mask=False)

        # Find feature split that maximizes information gain
        max_gain = float('-inf')
        best_feature = None
        best_splits_X = None
        best_splits_y = None
        for feature_i, feature in enumerate(features):
            values = X[:, feature_i]
            labels = np.unique(values)  # Possible labels for feature

            # Split into bins based on each possible feature value
            splits = [y[values == label] for label in labels]

            # Calculate information gain for splitting on feature
            info_gain = self._information_gain(y, splits)

            # Update maximum information gain
            if info_gain > max_gain:
                max_gain = info_gain
                best_feature = feature_i
                best_splits_X = [X[values == label] for label in labels]
                best_splits_y = splits

        # Split on the feature that maximizes information gain
        features.mask[best_feature] = True  # Only want to split on feature once
        branches = []
        for i in range(len(best_splits_y)):
            branch = self.id3(best_splits_X[i], best_splits_y[i],
                              features=features)
            branches.append(branch)

        return DecisionNode(feature_index=best_feature, children=branches)

    def _entropy(self, y):
        # NOTE: Be careful of floating point arithmetic error
        # Get the possible classes and number of occurences for each
        labels, counts = np.unique(y, return_counts=True)

        # Calculate entropy in bits (shannons)
        p = counts / np.float32(len(y))  # Proportion of each class i in set y
        entropy = 0
        for i, label in enumerate(labels):
            entropy += -p[i] * math.log(p[i]) / math.log(2)
        return entropy

    def _information_gain(self, y, splits):
        # Get proportion for each split of y
        p = np.array([len(a) for a in splits]) / np.float32(len(y))

        # Calculate information gain for given splits
        info_gain = self._entropy(y)
        for i, split in splits:
            info_gain -= p[i]*self._entropy(split)

        return info_gain


def test():
    pass

if __name__ == '__main__':
    test()
