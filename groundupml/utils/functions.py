from __future__ import division, print_function

import math

import numpy as np


def euclidean_distance(p1, p2):
    """Calculate the euclidean distance between two points
    """
    return np.linalg.norm(p1 - p2, ord=2)


def entropy(y):
    """Calculate the information entropy of a set of labels

    Args:
        y (numpy array of shape [n_samples]):
            Set of labels to calculate entropy on

    Returns:
        entropy (float):
            The entropy of the set of labels in bits
    """
    # NOTE: Be careful of floating point arithmetic error
    # Get the possible classes and number of occurences for each
    labels, counts = np.unique(y, return_counts=True)

    # Calculate entropy in bits (shannons)
    p = counts / np.float32(len(y))  # Proportion of each label i in set y
    entropy = 0
    for i, label in enumerate(labels):
        entropy += -p[i] * math.log(p[i]) / math.log(2)

    return entropy


def information_gain(splits):
    """Calculate the information gain when a set of labels is split into the
    given subsets

    Args:
        splits (list of length [n_splits] where each elements is a numpy array
                of length [n_elements]):
            Groupings where all groups together add up to the original set

    Returns:
        information_gain (float):
            Information gain from breaking the original set into the given
            groupings
    """
    # Rebuild the original set from the splits by adding them together
    y = np.vstack(splits).flatten()

    # Get proportion of elements from original set in each split
    p = np.array([len(a) for a in splits]) / np.float32(len(y))

    # Calculate amount of information gained when using given groupings
    info_gain = entropy(y)
    for i, split in enumerate(splits):
        info_gain -= p[i] * entropy(split)

    return info_gain


def sigmoid(x):
    """Calculate sigmoid function element-wise on a tensor
    """
    return 1.0 / (1.0 + np.exp(-x))


def to_one_hot(a):
    one_hot = np.zeros((a.size, a.max()+1))  # Might be slow for large input
    one_hot[np.arange(a.size), a] = 1

    return one_hot


def one_hot_to_hard_pred(pred):
    hard_pred = np.zeros_like(pred)
    hard_pred[np.arange(len(pred)), pred.argmax(1)] = 1

    return hard_pred


def one_hot_to_class(one_hot):
    hard_pred = one_hot_to_hard_pred(one_hot)
    class_pred = np.where(hard_pred == 1)[1]

    return class_pred
