from __future__ import division, print_function

import numpy as np


class ActivationFunction(object):
    """Activation Function interface
    """
    def __call__(self):
        raise NotImplementedError()

    def _gradient(self):
        raise NotImplementedError()


class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1.0 - self.__call__(x))



