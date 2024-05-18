import numpy as np
import math

def random_unit_vector(dim):
    x = np.random.normal(size = dim)
    return x/np.linalg.norm(x)


def column(vector):
    """Returns column vector (np.array) from np.array or list"""
    return np.array(vector, dtype=float).reshape((len(vector), 1))

def polar_from_cart(x, y, z):
    r = math.hypot(x, y, z)
    theta = math.atan2(y, x)
    phi = math.atan2(math.hypot(x, y), z)
    
    return r, theta, phi
