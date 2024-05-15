import numpy as np


def random_unit_vector(dim):
    x = np.random.normal(size = dim)
    return x/np.linalg.norm(x)


def column(vector):
    """Returns column vector (np.array) from np.array or list"""
    return np.array(vector, dtype=float).reshape((len(vector), 1))
