import numpy as np

from pca import PCA


class ICA():
    def __init__(self, max_iterations=200, tolerance=1e-4):
        self.w = None
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def transform(self, X, n_components=None):
        self.n_components = n_components
        self.n_sources, self.n_cols = np.shape(X)

        # Use all components if not set
        if self.n_components is None:
            self.n_components = self.n_cols

        g = self._logcosh

        # Center column variables around the mean
        X -= np.mean(X, axis=1)[:, np.newaxis]

        # Whiten using PCA
        X = PCA().transform(X, self.n_components)

        # Initialize weights
        w = np.random.randn(self.n_components, self.n_components)

        W = np.zeros((self.n_components, self.n_components))
        for i in range(n_components):
            w /= np.sqrt((w**2).sum())

            for _ in range(self.max_iterations):
                gwtx, g_wtx = g(np.dot(w.T, X))

                w1 = np.mean(X * gwtx, axis=1) - np.mean(g_wtx) * w

                # Orthonormalize w with respect to the relevant rows of W
                w1 -= w1.dot(W[:i].T).dot(W[i])

                w1 /= np.sqrt((w1**2).sum())

                cost = np.abs(np.abs((w1 * w).sum()) - 1)
                w = w1
                if cost < self.tolerance:
                    break

            W[i, :] = w

        # Compute sources
        S = np.dot(W, X).T
        return S

    def _logcosh(self, x):
        gx = np.tanh(x, x)
        g_x = np.empty(np.shape(x)[0])

        for i, gx_i in enumerate(gx):
            g_x[i] = np.mean((1 - gx_i**2))

        return gx, g_x


def test():
    # TODO: Better testing
    pass


if __name__ == '__main__':
    test()
