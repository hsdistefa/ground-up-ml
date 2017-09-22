from __future__ import division, print_function

from supervised.linear_regression import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

if __name__ == '__main__':
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
    lr_gd = LinearRegression(stochastic=True)
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
    ax1.set_title('Linear Regression SGD (MSE: {:2f})'.format(
        mse_gd))

    ax2.scatter(X_test[:, 0], y_test)
    ax2.plot(X_test[:, 0], y_pi_pred)
    ax2.set_title('Linear Regression Pseudoinverse (MSE: {:2f})'.format(
        mse_pi))
    plt.show()
