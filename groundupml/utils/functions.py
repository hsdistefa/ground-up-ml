from __future__ import division, print_function

import numpy as np


def cross_entropy(y_pred, y_actual, reduction='mean'):
    """Calculate the cross entropy between two distributions

    Cross entropy represents the number of bits needed to identify an event from
    a set of possibilities. In this case, the event is the true label and the
    set of possibilities are the predicted labels. The lower the cross entropy,
    the better the model is at predicting the true label.

    Args:
        y_pred (numpy array of shape [n_samples, n_classes]):
            Predicted distribution
        y_actual (numpy array of shape [n_samples, n_classes]):
            True distribution (one-hot encoded)
        reduction (str, optional):
            Method to reduce cross entropy across samples. Options are 'mean'
            and 'sum'

    Returns:
        cross_entropy (float):
            Cross entropy between the two distributions
    """
    # NOTE: This function assumes that the distributions are normalized, i.e.
    #       that the sum of each row is 1.0 in both y_pred and y_actual. This
    #       can be achieved by applying the softmax activation function first.
    log_likelihoods = -y_actual * np.log(y_pred + 1e-8)

    if reduction == 'mean':
        N = y_pred.shape[0]
        cross_entropy = np.sum(log_likelihoods) / N
    elif reduction == 'sum':
        cross_entropy = np.sum(log_likelihoods)

    return cross_entropy


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
    # Calculate the probability of each class in the set (proportions)
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)  # Proportion of each label i in set y
    # Entropy in bits (shannons)
    entropy = -np.sum(probs * np.log2(probs))

    return entropy


def information_gain(y_splits, impurity_func):
    """Calculate the information gain when a set of labels is split into
    the given subsets

    Args:
        splits (list of length [n_splits] where each element is a numpy 
                array of length [n_elements]):
            Groupings where all groups together add up to the original set
        impurity_func (`obj` str):
            Impurity function to use when calculating information gain. 'gini'
            or 'entropy'

    Returns:
        information_gain (float):
            Information gain from breaking the original set into the given
            groupings
    """
    # Set impurity function as lambda
    if impurity_func == 'gini':
        impurity_func = gini
    elif impurity_func == 'entropy':
        impurity_func = entropy

    # Rebuild the parent target set by re-combining splits
    parent_y = np.concatenate(y_splits)

    # Calculate the probability of a sample falling in each child node
    sizes = [len(split) for split in y_splits]
    child_weights = sizes / np.float32(len(parent_y))

    # Information gain is the difference between the parent-set's impurity 
    # and the child-sets' impurities weighted by child-set size
    parent_impurity = impurity_func(parent_y)
    child_impurities = [impurity_func(split) for split in y_splits]
    info_gain = parent_impurity - np.sum(child_weights * child_impurities)

    return info_gain

def gini(y):
    """Calculate the gini impurity of a set of labels

    Args:
        y (numpy array of shape [n_samples]):
            Set of labels to calculate gini impurity on

    Returns:
        gini impurity (float):
            The gini impurity of the set of labels
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    return 1 - np.sum(probs**2)

def sigmoid(x):
    """Calculate sigmoid function element-wise on a tensor
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(sigmoid_x):
    """Calculate the derivative of the sigmoid function element-wise on a
    tensor

    Args:
        sigmoid_x (numpy array):
            The output of the sigmoid activations to calculate the derivatives
            on. This is passed in to avoid having to re-calculate the sigmoid.
    
    Returns:
        sigmoid_prime (numpy array of same size as input):
            The derivative of the sigmoid function
    """
    return sigmoid_x * (1 - sigmoid_x)

def softmax(x):
    """Compute softmax of tensor x
    """
    # TODO: Test softmax function
    # Subtract the maximum value to avoid overflow (make more numerically stable)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))

    return exps / np.sum(exps, axis=1, keepdims=True)


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
