import numpy as np
import scipy.linalg as spl


def inner(A, B):
    return np.trace(A.T @ B)


def bracket(A, B):
    return (A @ B - B @ A)


def get_one_param_subgroup(X, num_elem):
    return_list = []
    for t in np.linspace(-np.pi, np.pi, num_elem):
        return_list.append(spl.expm(t*X))

    return return_list