import so3, torch, misc

bottom_row = torch.tensor([[0., 0., 0., 1.]])
so3_basis = so3.get_basis(3)

def get_pose(R, t):
    se3_matrix = torch.cat((torch.cat((R, t.unsqueeze(1)), dim=1), bottom_row), dim=0)
    return se3_matrix


def get_R(pose):
    return pose[:3, :3]


def get_t(pose):
    return pose[:3, 3].squeeze()


def hat(vec):
    return torch.cat(
        (torch.cat((so3.hat(vec[:3], so3_basis), vec[3:].unsqueeze(1)), dim=1), 
         torch.zeros(1, 4)),
        dim=0)


def exp_map(vec):
    # vec -> pose
    return torch.matrix_exp(hat(vec))


def log_map(g):
    # pose -> vec
    skew = so3.log_map(get_R(g))
    J_inv = so3.inverse_Jacobian(skew)
    return torch.cat(
        (so3.vee(skew), 
         misc.matrix_vector_product(J_inv, get_t(g))),
         dim=0)


def log_energy(error, project_to_basis = None):
    # || log (error) ||^2
    log_error = log_map(error)
    if project_to_basis is not None:
        projection_matrix = torch.einsum('ij,ik->jk', 
                                         project_to_basis, project_to_basis)
        log_error = misc.matrix_vector_product(projection_matrix, log_error)

    return misc.norm_squared(log_error)


def standard_gaussian(var):
    vec = torch.randn(6) * var
    return exp_map(vec)

