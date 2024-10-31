import numpy as np
import math
import torch

from datetime import datetime


def get_datetime_string():
    return datetime.now().strftime("%m-%d_%H-%M-%S")


def random_unit_vector(dim):
    x = np.random.normal(size = dim)
    return x/np.linalg.norm(x)


def polar_from_cart(x, y, z):
    r = math.hypot(x, y, z)
    theta = math.atan2(y, x)
    phi = math.atan2(math.hypot(x, y), z)
    
    return r, theta, phi


def cart_from_polar(r, theta, phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z


def inner(v1, v2):
    # Batched inner product
    return torch.einsum('...j,...j->...', v1, v2)


def inner_sum(v1, v2):
    # Contraction over both indices
    return (v1 * v2).sum()


def scalar_matrix_product(scalar, matrix):
    return torch.einsum("...,...jk->...jk", scalar, matrix)


def normalize(v):
    return v / torch.norm(v, dim=1, keepdim=True)