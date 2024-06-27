"""
Most of these work even if the argument is a torch tensor, and each element along the first (leftmost) dimension is a rotation.
"""

import torch
from scipy.spatial.transform import Rotation


# Orthonormal basis of SO(3) with shape [3, 3, 3]
basis = torch.tensor([
    [[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
    [[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]],
    [[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]])

# hat map from vector space R^3 to Lie algebra so(3)
def hat(v): 
    return torch.einsum('...i,ijk->...jk', v, basis)

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def log_vee(g): return torch.tensor(Rotation.from_matrix(g.numpy()).as_rotvec())
    
# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(g): return hat(log_vee(g))

# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(X): 
    return torch.linalg.matrix_exp(X)

# (Riemannian) exponential map from tangent space at g to SO(3)
def exp_g(g, tangent):
    X = torch.einsum('...ij,...ik->...jk', g, tangent)
    return torch.einsum('...ij,...jk->...ik', g, exp(X))

# Return angle of rotation. SO(3) to R^+
def get_angle(R): 
    return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(axis=-1)-1)/2)