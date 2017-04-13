import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class LinearRegression():
    """Linear Regression

    Args:
        num_iterations (:obj: `int`, optional):
            The number of iterations to do in the case of gradient descent.
        learning_rate (:obj: `float`, optional):
            The step magnitude used for updating the weights when doing gradient
            descent.
    """
    def __init__(self, num_iterations=100, learning_rate=.001):
        self.w = None
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit given training data to a linear model using regression

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Training data
            y (numpy array of shape [n_samples]:
                Labels
        """
        # TODO: implement pseudoinverse solution
        # Append bias weights to input
        X = np.insert(X, 0, 1, axis=1)
        # Initialize weights
        self.w = np.random.randn(X.shape[1])  # Could use truncated normal
        for _ in range(self.num_iterations):
            # Compute gradient w.r.t. weights of squared error function
            gradient = X.T.dot(X.dot(self.w) - y)
            # Update weights in the direction that minimizes loss
            self.w -= self.learning_rate * gradient

    def predict(self, X):
        """Predict given test data using the linear model

        Args:
            X (numpy array of shape [t_samples, n_features]):
                Test data

        Returns:
            C (numpy array of shape [n_samples,]):
                Predicted values from test data
        """
        # Append same bias weights to new input
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w)


def test():
    NUM_SAMPLES = 500

    # Generate random correlated dataset
    X, y = datasets.make_regression(n_features=1, n_samples=NUM_SAMPLES,
                                    bias=50, noise=10, random_state=7901)
    # Split into test and training sets
    split_index = int(.7*NUM_SAMPLES)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Run model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predictions = lr.predict(X_test)

    # Compute and show the model error
    mse = np.mean(np.power(y_predictions - y_test, 2))
    print('Mean Square Error: ', mse)

    # Plot the results
    plt.scatter(X_test[:, 0], y_test)
    plt.plot(X_test[:, 0], y_predictions)
    plt.title('Linear Regression (MSE: {:2f})'.format(mse))
    plt.show()


if __name__ == '__main__':
    test()