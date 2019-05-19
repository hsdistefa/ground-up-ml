from __future__ import division, print_function

from enum import Enum

import numpy as np


class ActivationFunction(object):
    """Activation Function interface
    """
    def __call__(self):
        raise NotImplementedError()

    def gradient(self):
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


class ActFunctions(Enum):
    """Enum to allow translation between string name arguments and the
    respective activation function class
    """
    sigmoid = Sigmoid()
    relu = ReLU()
