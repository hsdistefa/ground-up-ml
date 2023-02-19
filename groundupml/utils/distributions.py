from __future__ import division, print_function

import numpy as np


def poisson(lmda, k):
    """Poisson Distribution

    Args:
        lmda (:obj: `float`):
            The expected rate of occurrence of the random event within a fixed interval
        
        k (:obj: `int`):
            The number of occurrences

    Returns:
        Probability of k events occurring within the time interval used by lmda
    """
    return (lmda**k * np.exp(-lmda)) / np.math.factorial(k)

