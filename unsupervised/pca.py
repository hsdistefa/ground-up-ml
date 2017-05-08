import math

import numpy as np


class PCA():
    def __init__(self, svd=True):
        """Principal Component Analysis

        Args:
            svd (:obj: `bool`, optional):
                If true use single value decomposition to calculate the
                transformation
        """
        self.svd = svd

    def transform(self, X, n_components):
        """Project X onto the n principal components of X

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Data to be transformed
            n_components (int):
                Number of principal components to project the data onto
        """
        n_samples, n_features = X.shape

        # SVD Solution
        if self.svd:
            # Construct matrix Y such that Y^T Y is the covariance matrix
            X = X - np.mean(X)  # First re-center input at the origin
            Y = X / math.sqrt(n_samples - 1)

            # Compute singular value decomposition
            U, S, PC = np.linalg.svd(Y)

            # Pick the first n principal components that minimize redundancy
            PC = PC.T[:, :n_components]
            print(PC)

        # Covariance matrix solution
        else:
            # Calculate the covariance matrix
            X = X - np.mean(X)  # First re-center input at the origin
            cov = X.T.dot(X) / (n_samples - 1)

            # Get the eigenvectors and eigenvalues of the covariance matrix
            # These eigenvectors are necessarily orthogonal since the covariance
            # matrix is symmetric
            eigenvalues, PC = np.linalg.eig(cov)

            # Sort the principal components (eigenvectors of cov-matrix) in
            # decreasing order by the magnitude of their eigenvalues
            # i.e. by how much they contribute to variance
            indices = np.argsort(eigenvalues)[::-1]
            PC = np.atleast_1d(PC[:, indices])

            # Pick the first n principal components that minimize redundancy
            PC = PC[:, :n_components]
            print(PC)

        # Project input onto the n principal components
        return X.dot(PC)


def test():
    # TODO: more comprehensive testing
    np.random.seed(seed=4443)
    pca = PCA(svd=False)
    X = np.random.randn(10, 5)
    transformed = pca.transform(X, 2)
    print(transformed)


if __name__ == '__main__':
    test()
