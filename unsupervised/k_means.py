from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


class KMeans():
    """K Means

    Args:
        k (int):
            Number of clusters to create.
        max_iterations (:obj: `int`, optional):
            Cap for the number of iterations to do if the clustering has not
            fully converged.
        init_method (:obj: `str`, optional):
            Specifies the way in which centroids are initialized:
                forgy:     Select random points from input as initial centroids.
                rand_part: Randomly partition each sample into clusters, then
                           compute the centroids of those clusters. Tends to
                           select initial centroids clustered near the first
                           moment of input.
                kpp:       Initialize centroids probabilistically so that
                           they are much more likely to start farther from each
                           other.
    """
    def __init__(self, k, max_iterations=1000, init_method='forgy'):
        self.k = k
        self.max_iterations = max_iterations
        self.init_method = init_method
        self._n_samples = None
        self._n_features = None

    def predict(self, X):
        """Classify each of the given samples into K different clusters

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Samples to classify
        Returns:
            C (numpy array of shape [n_samples]):
                Cluster predictions for each sample
        """
        # Store number of samples and features in class variable
        self._n_samples, self._n_features = np.shape(X)

        # Initialize centroids
        centroids = self._initialize_centroids(X)
        prev_centroids = np.zeros((self.k, self._n_features))
        # Iterate only until convergence or max iterations
        for _ in range(self.max_iterations):
            # Has converged if the centroids haven't changed
            difference = centroids - prev_centroids
            if not difference.any():
                break

            # Assign each point to the closest centroid
            clusters = self._get_clusters(centroids, X)

            # Calculate new centroids
            prev_centroids = centroids
            centroids = self._get_new_centroids(X, clusters)
        return self._get_cluster_labels(clusters)

    def _get_clusters(self, centroids, X):
        # Create clusters by assigning each point to the nearest centroid
        clusters = [[] for _ in range(self.k)]
        for i, x in enumerate(X):
            closest_dist = float('inf')
            closest_index = None
            for j, c in enumerate(centroids):
                dist = self._euclidean_distance(x, c)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_index = j
            clusters[closest_index].append(i)
        return np.array(clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.zeros(self._n_samples, dtype=np.int32)
        # Note: can be done better?
        for cluster_i, cluster in enumerate(clusters):
            for point_i in cluster:
                labels[point_i] = cluster_i
        return labels

    def _get_new_centroids(self, X, clusters):
        # Calculate new centroids by taking the mean of each cluster
        centroids = np.zeros((self.k, self._n_features))
        for i, cluster in enumerate(clusters):
            if X[cluster].size != 0:  # Avoid taking mean of empty cluster
                centroids[i] = np.mean(X[cluster], axis=0)
        return centroids

    def _initialize_centroids(self, X):
        # TODO: minimax initialization

        # Forgy initialization
        #   i.e. Choose random points from X as initial centroids
        if self.init_method == 'forgy':
            centroids = np.zeros((self.k, self._n_features))
            for i in range(len(centroids)):
                centroids[i] = X[np.random.choice(range(self._n_samples))]

        # K-means++ initialization
        elif self.init_method == 'kpp':
            centroids = self._kmeanspp(X)

        # Random Partition initialization
        elif self.init_method == 'rand_part':
            centroids = self._random_partition(X)

        return centroids

    def _kmeanspp(self, X):
        # FIX: unfinished
        centroids = np.zeros((self.k, self._n_features))

        # Initialize the probability distribution for sampling each point
        # Distribution is initially uniform
        probs = np.zeros(self._n_samples).astype(np.float32)
        probs.fill(1./self._n_samples)

        for i in range(self.k):
            # Choose a point from the input using the probability distribution
            # and make that point a centroid
            p_i = np.random.choice(range(self._n_samples), p=probs)
            centroids[i] = X[p_i]

            # Update probability distribution so that points farther from the
            # nearest centroid to them are more likely to be chosen next
            # sampling
            for i, x in enumerate(X):
                closest_dist = float('inf')
                for c in centroids:
                    dist = self._euclidean_distance(x, c)
                    if dist < closest_dist:
                        closest_dist = dist
                prob = closest_dist**2
                probs[i] = prob

            # Normalize the probabilities so that they sum to 1
            probs = probs / np.sum(probs)

        return centroids

    def _random_partition(self, X):
        # Assign each sample to a cluster randomly
        clusters = [[] for _ in range(self.k)]
        for i, x in enumerate(X):
            cluster_i = np.random.randint(0, self.k)
            clusters[cluster_i].append(i)
        clusters = np.array(clusters)

        # Compute centroids from partitioned clusters
        centroids = self._get_new_centroids(X, clusters)

        return centroids

    def _euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)


def test():
    NUM_CLUSTERS = 3

    # Make a dataset with clustered points
    np.random.seed(seed=5987)
    X, y = sklearn.datasets.make_blobs(centers=NUM_CLUSTERS)

    # Predict clusters
    km = KMeans(k=3, init_method='kpp')
    predictions = km.predict(X)

    # Plot clusters
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X[:, 0], X[:, 1], c=y)
    ax1.set_title('Actual Clustering')

    ax2.scatter(X[:, 0], X[:, 1], c=predictions)
    ax2.set_title('K Means Clustering')
    plt.show()


if __name__ == '__main__':
    test()
