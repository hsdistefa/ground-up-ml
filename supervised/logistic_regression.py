import numpy as np


class LogisticRegression():
    def __init__(self, num_iterations=100, learning_rate=.001):
        self.num_iterations = num_iterations
        self.w = None
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Add bias weights to input
        X = np.insert(X, 0, 1, axis=1)

        n_samples, n_features = np.shape(X)

        # Initialize weights
        self.w = np.random.randn(n_features)

        for _ in range(self.num_iterations):
            # Gradient descent
            h_x = self._sigmoid(X.dot(self.w))
            gradient = X.T.dot(h_x - y)
            self.w -= self.learning_rate * gradient
            # TODO: implement batch optimization

    def predict(self, X):
        # Add bias weights to input
        X = np.insert(X, 0, 1, axis=1)
        # Round model prediction to nearest class
        return np.round(self._sigmoid(X.dot(self.w))).astype(np.int32)

    def _sigmoid(self, thetaTX):
        return 1 / (1 + np.exp(-thetaTX))


def test():
    # TODO: testing
    pass


if __name__ == '__main__':
    test()
