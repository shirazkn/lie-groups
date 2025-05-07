from soN import *


basis = get_basis(2)

def hat(vec):
    return torch.einsum('...,...jk->jk', vec, basis)

def exp_map(x):
    X = hat(x)
    omega = torch.sqrt((X**2).sum()*0.5)
    R = torch.eye(X.size(1), requires_grad=False)
    if omega > 1e-8:
       R += (
           misc.scalar_matrix_product(torch.sin(omega)/(omega), X) 
           + misc.scalar_matrix_product((1.-torch.cos(omega))/omega**2, X@X)
        )
    else:
        R += (
            misc.scalar_matrix_product(1-omega**2/6, X) 
            + misc.scalar_matrix_product(0.5-omega**2/24, X@X)
        )
    return R