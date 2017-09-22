from __future__ import division, print_function

from unsupervised.dbscan import DBSCAN

import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == '__main__':
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
