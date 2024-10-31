import torch, so3
torch.set_default_dtype(torch.float64)
so2_basis = so3.get_basis(dim=2)


def get_pose(theta, t):
    # Returns an SE(2) Matrix
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)], 
                      [torch.sin(theta), torch.cos(theta)]])
    return torch.cat(
        (torch.cat((R, t.unsqueeze(1)), 1), torch.tensor([[0., 0., 1.]])), 
        0)

def exp_map(vec):
    # vec -> pose
    mat = torch.matrix_exp(hat(vec))
    mat[:2, :2] = so3.exp_map(so3.hat(vec[:1], so2_basis))
    return mat

def hat(vec):
    return torch.cat(
        (torch.cat((so3.hat(vec[:1], so2_basis), vec[1:].unsqueeze(1)), dim=1),
         torch.zeros(1, 3)), 
         dim=0)

def get_t(pose):
    return pose[:2, 2].squeeze()