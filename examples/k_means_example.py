from __future__ import division, print_function

from unsupervised.k_means import KMeans

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == '__main__':
    NUM_CLUSTERS = 3

    # Make a dataset with clustered points
    np.random.seed(seed=5987)
    X, y = datasets.make_blobs(centers=NUM_CLUSTERS)

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
