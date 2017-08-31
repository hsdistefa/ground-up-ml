from __future__ import division, print_function

import math

import numpy as np


class NaiveBayes():

    def __init__(self):
        self.labels = None

    def fit(self, X, y):
        self.n_samples, self.n_features = np.shape(X)
        self.labels = np.unique(y)
        self.params = {}

        # Find the prior P(Y) for each label, i.e. the number of samples with a
        # given label divided by the total number of samples. Also calculate the
        # mean and variance for each feature of each label to make use of in
        # distribution calculations, e.g. Gaussian.
        for label in self.labels:
            x_where_label = X[np.where(y == label)]

            values = {}
            values['prior'] = np.shape(x_where_label)[0] / self.n_samples
            values['means'] = np.mean(x_where_label, axis=0)
            values['vars'] = np.var(x_where_label, axis=0)
            self.params[label] = values

    def predict(self, X):
        # Choose the label that maximizes the naive posterior probability
        predictions = np.empty(self.n_samples)
        for i, sample in enumerate(X):
            max_prob = float('-inf')
            max_label_i = None
            for label_i, label in enumerate(self.labels):
                # Get the parameters for the label to be used in calculating
                # the probability
                # Note: Although O(1), constant re-lookup of params could cause
                # significant overhead
                parameters = self.params[label]
                prior = parameters['prior']
                means = parameters['means']
                variances = parameters['vars']

                # Calculate the probability of seeing this label given the
                # features of the sample, making the assumption that the
                # features are independent
                prob_label = self._calculate_posterior(sample, prior,
                                                       means, variances)

                if prob_label > max_prob:
                    max_prob = prob_label
                    max_label_i = label_i

            predictions[i] = self.labels[max_label_i]

        return predictions

    def _calculate_posterior(self, sample, prior, means, variances):
        # Calculate P(X|Y), i.e. the probability of seeing a sample X with these
        # features, given that it has label Y.
        # We do this by assuming (naively) that each feature is independent
        # such that:
        #   P(X|Y) = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xn_features|Y)
        # where P(Y) is the prior probability of seeing label Y in the input
        posterior = prior
        for feature_i in range(self.n_features):
            feature = sample[feature_i]

            # Here we assume that the possible values for a feature are
            # distributed normally and use that to get the probability of seeing
            # this particular value for that feature
            prob = self._calculate_norm(feature, means[feature_i],
                                        variances[feature_i])
            posterior *= prob
        return posterior

    def _calculate_norm(self, x, mean, variance):
        if variance == 0:
            variance = 10**-100  # Note: maybe a bad idea

        return (1.0 / (math.sqrt((2 * math.pi) * variance))) * \
            math.exp(-(math.pow(x - mean, 2) / (2.0 * variance)))


def test():
    # TODO: better testing
    X = np.array([[.75, .32, 1.8],
                  [1.76, 2.0, .22],
                  [.68, .36, 1.9]])
    y = np.array([1, 0, 1])

    test = np.array([[.7, .3, 2],
                     [1.7, 1.9, .2],
                     [.7, .3, 2.2]])

    nb = NaiveBayes()
    nb.fit(X, y)
    print(X)
    print(y)
    print(test)
    print(nb.predict(test))


if __name__ == '__main__':
    test()
