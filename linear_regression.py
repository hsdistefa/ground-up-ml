import numpy as np


class LinearRegression():
    def __init__(self, num_iterations=100, learning_rate=.001):
        self.w = None
        self.num_iterations = num_iterations

    def fit(self, X, y):
        # Append bias weights to input
        X = np.insert(X, X.shape[0], 1, axis=1)
        # Initialize weights
        self.w = np.random.randn(X.shape[1])  # Could use truncated normal
        for _ in range(self.num_iterations):
            # Compute gradient of squared error function w.r.t weights
            gradient = X.T.dot(X.dot(self.w) - y)
            # Update weights in the direction that minimizes loss
            self.w -= self.learning_rate * gradient

    def predict(self, X):
        # Append same bias weights to new input
        X = np.insert(X, X.shape[0], 1, axis=1)
        return X.dot(self.w)


def test():
    """TODO"""
    pass


if __name__ == '__main__':
    test()
