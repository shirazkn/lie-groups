import torch, so2, misc
torch.set_default_dtype(torch.float64)


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
    mat[:2, :2] = so2.exp_map(vec[:1])
    return mat

def hat(vec):
    return torch.cat(
        (torch.cat((so2.hat(vec[:1]), vec[1:].unsqueeze(1)), dim=1),
         torch.zeros(1, 3)), 
         dim=0)

def get_t(pose):
    return pose[:2, 2].squeeze()

def get_R(pose):
    return pose[:2, :2]

def vee(mat):
    skew = get_R(mat)
    return torch.cat(
        (so2.vee(skew), get_t(mat)))

def log_map(mat):
    return vee(misc.matrix_log(mat))