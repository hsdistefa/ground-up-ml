from __future__ import division, print_function

import matplotlib.pyplot as plt
from sklearn import datasets

from groundupml.unsupervised.pca import PCA


if __name__ == '__main__':
    # Get Data
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Transform
    pca = PCA(svd=False)
    X_transformed = pca.transform(X, 2)
    pc1 = X_transformed[:, 0]
    pc2 = X_transformed[:, 1]

    # Plot
    plt.scatter(pc1, pc2, c=y)
    plt.title('Digits Dataset 0-9')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
