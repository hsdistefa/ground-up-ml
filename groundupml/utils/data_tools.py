import numpy as np


def confusion_matrix(actual, predicted):
    # TODO: documentation
    # Extract the classes so we know confusion matrix size
    classes = np.unique(actual)
    n_classes = len(classes)

    # Initialize the confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes))

    # Construct confusion matrix
    for i, actual_class in enumerate(classes):
        for j, predicted_class in enumerate(classes):
            conf_matrix[i, j] = np.sum((actual == actual_class) & \
                                       (predicted == predicted_class))

    return conf_matrix



