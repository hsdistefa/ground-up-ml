from __future__ import division, print_function

import numpy as np


def scale_min_max(x):
    """Scale the given data by feature so all features have a maximum value
    of 1 and a minimum of 1.

    Args:
        x (numpy array of shape [n_samples, n_features]): 
            Data to scale.

    Returns:
        Scaled data
    """
    col_mins = np.nanmin(x, axis=0)
    col_maxes = np.nanmax(x, axis=0)

    min_max_scaled = (x - col_mins) / (col_maxes - col_mins)

    return min_max_scaled

def shuffle_data(X, y):
    """Shuffle the samples of data set and the corresponding labels.
    """
    indices = np.random.permutation(len(X))

    return X[indices], y[indices]


def split_data(X, y, proportion=0.8, shuffle=True):
    """Split the given data and corresponding labels into two sets with
    the given proportion in the first set and the remainder in the second
    """
    if shuffle:
        X, y = shuffle_data(X, y)

    first_i = int(proportion * len(X))  # Index where first set ends

    X1, y1 = X[:first_i, :], y[:first_i]
    X2, y2 = X[first_i:, :], y[first_i:]

    return X1, y1, X2, y2


def split_cross_test(X, y, cross=0.2, test=0.2):
    """Split given data and corresponding labels into a training set, a
    cross-validation set, and a test set with the given proportions
    """
    train_prop = 1.0 - (cross + test)
    X_train, y_train, X_rem, y_rem = split_data(X, y, train_prop)

    # Split remaining data into a cross-validation and test set
    cross_rem_prop = cross / (cross + test)
    X_cross, y_cross, X_test, y_test = split_data(X_rem, y_rem,
                                                  cross_rem_prop, False)

    return X_train, y_train, X_cross, y_cross, X_test, y_test
