from __future__ import division, print_function

import numpy as np


class LinearRegression():
    """Linear Regression

    Args:
        num_iterations (:obj: `int`, optional):
            Number of iterations to do in the case of gradient descent.
        learning_rate (:obj: `float`, optional):
            Step magnitude used for updating the weights when doing gradient
            descent.
        gradient_descent (:obj: `bool`, optional):
            If True do gradient descent, else get exact solution using
            least squares normal equation.
        stochastic (:obj: `bool`, optional):
            If True and doing gradient descent, computes the loss on a random
            subsample of the input at each step.
        batch_size (:obj: `int`, optional):
            The number of samples to use for computing the loss at each step
            when using stochastic gradient descent.
        reg_term (:obj: `float`, optional)
            The regularization constant to use to penalize larger weights.
    """
    def __init__(self, num_iterations=100, learning_rate=.001,
                 gradient_descent=True, stochastic=False, batch_size=32,
                 reg_term=0.0):
        self.w = None
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.stochastic = stochastic
        self.batch_size = batch_size
        self.reg_term = reg_term

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
        self.w = np.random.randn(np.shape(X)[1])  # Could use truncated normal

        # Least squares gradient descent
        if self.gradient_descent:
            for _ in range(self.num_iterations):
                # If stochastic gradient descent, only use a random subset of
                # the input data at each step
                if self.stochastic:
                    indices = np.random.randint(np.shape(X)[0],
                                                size=self.batch_size)
                    x = X[indices, :]
                    labels = y[indices]
                # Else all input data at each step
                else:
                    x = X
                    labels = y

                # Compute gradient w.r.t. weights of squared error function
                # including regularization term to penalize large weights
                # which may cause overfitting
                loss = np.dot(x, self.w) - labels
                gradient = np.dot(x.T, loss) + self.reg_term * self.w

                # Update weights in the direction that minimizes loss
                self.w -= self.learning_rate * gradient

        # Least Squares normal equation
        else:
            # Note: slow for large matrices

            reg_eye = np.eye(np.shape(X)[1])
            reg_eye[0][0] = 0  # Don't regularize bias term

            # Compute psuedoinverse of X^TX plus regularization term
            XTX = np.dot(X.T, X) + self.reg_term * reg_eye
            XTX_inv = _pseudoinverse(XTX)

            # Compute weights using characteristic equation
            self.w = XTX_inv.dot(X.T).dot(y)

    def predict(self, X):
        """Predict given test data using the linear model

        Args:
            X (numpy array of shape [n_samples, n_features]):
                Test data

        Returns:
            C (numpy array of shape [n_samples]):
                Predicted values from test data
        """
        # Append same bias weights to new input
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w)


def _pseudoinverse(X):
    U, S, V = np.linalg.svd(X)
    S = np.diag(S)
    X_inv = V.dot(np.linalg.pinv(S)).dot(U.T)

    return X_inv
