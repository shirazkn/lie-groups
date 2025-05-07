import torch, numpy, misc
from tqdm import tqdm

from matplotlib import pyplot as plt

from pandas import DataFrame
import seaborn as sns

from scipy.spatial.transform import Rotation
torch.set_default_dtype(torch.float64)


def dimension(dim):
    return int(dim*(dim-1)/2)


def get_basis(dim=3, i=None):
    if i is None:
        basis = torch.zeros([dimension(dim), dim, dim], requires_grad=False)
        for i in range(dimension(dim)):
            basis[i] = get_basis(dim, i+1)
        return basis
    
    mat = torch.zeros([dim, dim], requires_grad=False)
    row = dim - 2
    col = dim - 1
    parity = (-1)**i
    while (i-1) > 0:
        if col == row + 1:
            row -= 1
            col = dim - 1
        else:
            col -= 1
        i -= 1

    mat[row, col] = parity*1.
    mat[col, row] = -parity*1.
    return mat


def vee(mat):
    dim = mat.size()[-1]
    vec = torch.zeros(dimension(dim), 
                      dtype=mat.dtype, device=mat.device)
    row = dim - 2
    col = dim - 1
    for i in range(dimension(dim)):
        vec[i] = mat[row, col]*((-1)**(i+1))
        if col == row + 1:
            row -= 1
            col = dim - 1
        else:
            col -= 1
    return vec  


def log_map(R):
    omega = torch.acos((R.trace()-1)/2)
    if omega > 1e-5:
        return vee(omega/(2*torch.sin(omega))*(R-R.t()))
    else:
        return vee(0.5*(R-R.t()))


def inverse_Jacobian(X):
    # Inverse of the `left' Jacobian as given in Chirikjian (2011)
    omega = torch.sqrt((X**2).sum()*0.5)
    J_inv =  torch.eye(3) - 0.5*X 
    if omega > 1e-8:
        J_inv += + (1./omega**2  - (1.+torch.cos(omega))/(2*omega*torch.sin(omega)))*X@X

    return J_inv