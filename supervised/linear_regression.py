import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class LinearRegression():
    """Linear Regression

    Args:
        num_iterations (:obj: `int`, optional):
            Number of iterations to do in the case of gradient descent.
        learning_rate (:obj: `float`, optional):
            Step magnitude used for updating the weights when doing gradient
            descent.
        gradient_descent (bool):
            If true do gradient descent, else get exact solution using
            least squares normal equation
    """
    def __init__(self, num_iterations=100, learning_rate=.001,
                 gradient_descent=True):
        self.w = None
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent

    def fit(self, X, y):
        """Fit given training data to a linear model using regression

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Training data
            y (numpy array of shape [n_samples]:
                Labels
        """
        # Append bias weights to input
        X = np.insert(X, 0, 1, axis=1)

        # Initialize weights
        self.w = np.random.randn(X.shape[1])  # Could use truncated normal
        # Least squares gradient descent
        if self.gradient_descent:
            for _ in range(self.num_iterations):
                # Compute gradient w.r.t. weights of squared error function
                gradient = X.T.dot(X.dot(self.w) - y)
                # Update weights in the direction that minimizes loss
                self.w -= self.learning_rate * gradient
        # Least Squares normal equation
        else:
            # Note: slow for large matrices
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            XTX_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = XTX_inv.dot(X.T).dot(y)

    def predict(self, X):
        """Predict given test data using the linear model

        Args:
            X (numpy array of shape [t_samples, n_features]):
                Test data

        Returns:
            C (numpy array of shape [n_samples]):
                Predicted values from test data
        """
        # Append same bias weights to new input
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w)


def test():
    NUM_SAMPLES = 500

    # Generate random correlated dataset
    X, y = datasets.make_regression(n_features=1, n_samples=NUM_SAMPLES,
                                    bias=50, noise=20, random_state=7901)
    # Split into test and training sets
    split_index = int(.7*NUM_SAMPLES)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Run gradient descent model
    lr_gd = LinearRegression()
    lr_gd.fit(X_train, y_train)
    y_gd_pred = lr_gd.predict(X_test)

    # Run normal model
    lr_pi = LinearRegression(gradient_descent=False)
    lr_pi.fit(X_train, y_train)
    y_pi_pred = lr_pi.predict(X_test)

    # Compute the model error
    mse_gd = np.mean(np.power(y_gd_pred - y_test, 2))
    mse_pi = np.mean(np.power(y_pi_pred - y_test, 2))

    # Plot the results
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(X_test[:, 0], y_test)
    ax1.plot(X_test[:, 0], y_gd_pred)
    ax1.set_title('Linear Regression Gradient Descent (MSE: {:2f})'.format(
        mse_gd))

    ax2.scatter(X_test[:, 0], y_test)
    ax2.plot(X_test[:, 0], y_pi_pred)
    ax2.set_title('Linear Regression Pseudoinverse (MSE: {:2f})'.format(
        mse_pi))
    plt.show()


if __name__ == '__main__':
    test()
