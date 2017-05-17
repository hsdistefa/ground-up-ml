from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class DBSCAN():
    """Density-Based Spacial Clustering of Applications with Noise

    Args:
        min_pts (int):
            Minimum number of points for a group to be considered a cluster
        eps (:obj: `float`, optional):
            Maximum distance to check for neighbors to a point
    """
    def __init__(self, min_pts, eps=None):
        self.min_pts = min_pts
        self.eps = eps

    def predict(self, X):
        """Classify the given input into clusters using DBSCAN

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Samples to classify
        Returns:
            C (numpy array of shape [n_samples]):
                Cluster predictions for each sample
        """
        n_samples, n_features = np.shape(X)

        # TODO
        # If the max radius to qualify as a neighbor (epsilon) is not given,
        # find best value
        if self.eps is None:
            self.eps = self._choose_eps(X)

        visited = set()
        cluster_i = 0
        labels = np.zeros(n_samples)
        for point_i, point in enumerate(X):
            if point_i in visited:
                continue
            visited.add(point_i)
            # Get the points within distance epsilon of point
            neighbors = self._find_neighbors(point, X)

            # Label point as noise if it has too few neighbors
            if len(neighbors) < self.min_pts:
                labels[point_i] = -1
            # Otherwise label point, neighbors, and neighbors of neighbors as
            # belonging to the same cluster
            else:
                cluster_i += 1
                labels[point_i] = cluster_i
                labels, visited = self._expand_cluster(X, neighbors, cluster_i,
                                                       visited, labels)
        return labels

    def _choose_epsilon(self, X):
        # TODO: choose epsilon automatically
        pass

    def _expand_cluster(self, X, neighbors, cluster_i, visited, labels):
        for neighbor_i in neighbors:
            if neighbor_i not in visited:
                visited.add(neighbor_i)
                neighbors_prime = self._find_neighbors(X[neighbor_i], X)
                if len(neighbors_prime) >= self.min_pts:
                    neighbors = np.concatenate((neighbors, neighbors_prime))
                    labels, visited = self._expand_cluster(X, neighbors,
                                                           cluster_i, visited,
                                                           labels)
            if labels[neighbor_i] == 0:  # Neighbor is not in any cluster
                labels[neighbor_i] = cluster_i
        return labels, visited

    def _find_neighbors(self, center, X):
        # Note: could improve time complexity to O(logn) using an indexing
        # structure. Could also reduce recomputation with memoization.

        # Calculate distance between center point and each of the other points
        # in input inclusive
        distances = np.linalg.norm(X-center, ord=2, axis=1)  # Euclidean

        # Select points in input closer to the center point than given epsilon
        neighbors = np.where(distances < self.eps)[0]

        return neighbors


def test():
    # Load the dataset
    X, y = datasets.make_moons(n_samples=400, noise=0.1)

    # Cluster using DBSCAN
    dbs = DBSCAN(min_pts=3, eps=.15)
    y_pred = dbs.predict(X)

    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X[:, 0], X[:, 1], c=y)
    ax1.set_title('Actual Clustering')

    ax2.scatter(X[:, 0], X[:, 1], c=y_pred)
    ax2.set_title('DBSCAN Clustering')
    plt.show()

if __name__ == '__main__':
    test()
