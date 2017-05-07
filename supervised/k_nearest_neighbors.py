import numpy as np
import scipy.stats


class KNN():
    def __init__(self, k):
        """K Nearest Neighbors

        Args:
            k (int):
                Number of neighbors to use when evaluating test points
        """
        self.k = k

    def predict(self, X_train, y_train, X_test):
        """Using the k nearest neighbors in the given training data, lazy
        evaluate the value of the given test data

        Args:
            X_train (numpy array of shape [n_samples, n_features]):
                Training data
            y_train (numpy array of shape [n_samples]):
                Training labels
            X_test (numpy array of shape [n_samples, n_features]):
                Test data
        """
        # TODO: implement KNN regression
        # For each sample in the test set, find the k closest samples in the
        # train set and the corresponding labels
        labels_matrix = np.empty((X_test.shape[0], self.k))
        labels_matrix[:] = np.NAN
        for (i, test_sample) in enumerate(X_test):
            knn_dist = np.full(self.k, np.inf)
            knn_labels = np.empty(self.k)
            knn_labels[:] = np.NAN

            max_neighbor_i = 0
            for (j, train_sample) in enumerate(X_train):
                dist = self._euclidean_distance(test_sample, train_sample)
                if dist < knn_dist[max_neighbor_i]:
                    knn_dist[max_neighbor_i] = dist
                    knn_labels[max_neighbor_i] = y_train[j]
                    max_neighbor_i = np.argmax(knn_dist)
            labels_matrix[i] = knn_labels

        # Take the majority vote of each of the k closest train samples to
        # determine the label to apply to each of the test samples
        modes, _ = scipy.stats.mode(labels_matrix, axis=1)
        return modes

    def _euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)


def test():
    # TODO: more comprehensive testing
    NUM_SAMPLES = 100
    NUM_FEATURES = 4
    np.random.seed(1005)
    X_train = np.random.randn(NUM_SAMPLES, NUM_FEATURES)
    y_train = np.random.randint(0, NUM_FEATURES+1, size=NUM_SAMPLES)
    X_test = np.array([[2, 2, 7, 3],
                      [-1, 5, 6, 10]])
    knn = KNN(2)
    print(knn.predict(X_train, y_train, X_test))


if __name__ == '__main__':
    test()
