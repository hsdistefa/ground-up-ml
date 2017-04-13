import numpy as np


class PCA():
    def __init__(self):
        pass

    def transform(self, X, n_components):
        # TODO: implement SVD solution
        n_samples, n_features = X.shape

        # Calculate the covariance matrix
        X = X - np.mean(X)  # First re-center input at 0-mean
        cov = X.T.dot(X) / (n_samples - 1)

        # Get the eigenvectors and eigenvalues of the covariance matrix
        # These eigenvectors are necessarily orthogonal since the covariance
        # matrix is symmetric
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort the principal components (eigenvectors of cov-matrix) in
        # decreasing order by the magnitude of their eigenvalues
        # i.e. by how much they contribute to overall variance
        indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = np.atleast_1d(eigenvectors[:, indices])

        # Pick the first n principal components that minimize redundancy
        eigenvectors = eigenvectors[:, :n_components]

        # Project input onto the n principal components
        return X.dot(eigenvectors)


def test():
    # TODO: more comprehensive testing
    pca = PCA()
    X = np.random.randn(10, 5)
    pca.transform(X, 2)


if __name__ == '__main__':
    test()
