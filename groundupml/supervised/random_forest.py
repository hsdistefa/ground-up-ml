# coding: utf-8

from __future__ import division, print_function

import numpy as np

from groundupml.supervised.binary_decision_tree import DecisionTree


class RandomForest():
    """Random Forest Classifier

    Args:
        n_trees (:obj: `int`, optional):
            Number of trees to build in the forest
        max_depth (:obj: `int`, optional):
            Maximum depth of each tree in the forest
        impurity_func (:obj: `str`, optional):
            Name of the impurity function to use when calculating
            information gain. 'gini' or 'entropy'
        max_features (:obj: `str` or `int`, optional):
            Number of features to consider when building a tree. If 'sqrt',
            then `max_features=sqrt(n_features)`. If an `int`, then
            `max_features=max_features`
    """
    def __init__(self, n_trees=10, max_depth=5, impurity_func='gini',
                 max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.impurity_func = impurity_func
        self.max_features = max_features

        self.trees = []

    def fit(self, X, y):
        """Fit the model to the training data using a random forest model.

        This is done by building `n_trees` decision trees, each of which is
        trained on a random subset of the training data with a random subset
        of the training features.

        Args:
            X (:obj: numpy array of shape [n_samples, n_features]):
                Training data
            y (:obj: numpy array of shape [n_samples]):
                Training labels
        """
        # Build decision trees
        for i in range(self.n_trees):
            # Randomly sample the training data
            # We want to generate a sample input with the same number of
            # rows as the original input, but each row will be selected at
            # random with replacement. This is known as bootstrap aggregation,
            # or bagging.
            n_samples = X.shape[0]
            sample_idx = np.random.choice(n_samples,
                                          size=n_samples,
                                          replace=True)
            X_sample = X[sample_idx, :]
            y_sample = y[sample_idx]

            # Randomly sample the training features
            # This reduces the number of features available to fit each tree and
            # thus the correlation between trees, which helps to prevent 
            # overfitting. The number of features to sample is recommended
            # to be the square root of the total number of
            # features for classification, but should be considered on a
            # case-by-case basis.
            if self.max_features == 'sqrt':
                # Round down to nearest integer since we can't have partial
                # features
                n_sample_features = int(np.sqrt(X.shape[1]))
                n_sample_features = 2
            else:
                n_sample_features = self.max_features
            # Sample without replacement so we don't have duplicate features
            sample_features_idx = np.random.choice(X.shape[1], 
                                                   size=n_sample_features, 
                                                   replace=False)
            X_sample = X_sample[:, sample_features_idx]

            # Initialize a new decision tree
            dt = DecisionTree(max_depth=self.max_depth,
                              impurity_func=self.impurity_func)

            # Fit the decision tree to the sampled data
            dt.fit(X_sample, y_sample)

            # Add the decision tree to the forest
            self.trees.append(dt)

    def predict(self, X):
        """Predict the labels of the given test data using the fitted random
        forest model

        Args:
            X (:obj: numpy array of shape [n_samples, n_features]):
                Test data

        Returns:
            C (:obj: numpy array of shape [n_samples]):
                Predicted labels from test data
        """
        # Get the predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Take the majority vote (mode) from trees to get aggregated predictions
        # NOTE: This assumes that the classes are encoded as integers
        # starting at 0
        return np.array([np.bincount(preds).argmax() for preds in tree_preds.T])
