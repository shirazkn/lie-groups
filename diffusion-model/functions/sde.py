import torch
import constants, learning


class SDE():
    """
    Generates sample paths of an SDE defined on matrix Lie group
    """
    def __init__(self, group, device=None):
        self.group = group
        self.basis = group.get_basis(dtype=constants.datatype, device=device)
        self.dim = len(self.basis)
        self.dt = torch.tensor(constants.params["sde_dt"], 
                               dtype=constants.datatype, device=device,
                               requires_grad=False)
        self.zero_vector = torch.zeros(self.dim, 
                           dtype=constants.datatype, device=device,
                           requires_grad=False)
        
        self.sigma = lambda t: standard_deviation(t)
        self.time = torch.tensor(0., dtype=constants.datatype, 
                                 device=device, requires_grad=False)
    
    def flow(self, g, T):
        self.time *= 0.
        while self.time < T:
            dx = self.get_velocity(g, self.time) * self.dt
            dx += (self.sigma(self.time) * torch.sqrt(self.dt)) \
                * torch.randn_like(dx)
            dX = torch.einsum('...i,ijk->...jk', dx, self.basis)
            g = torch.matmul(g, self.group.exp_map(dX))
            self.time += self.dt

        return g

    def get_velocity(self, g, t):
        return self.zero_vector.repeat(g.size()[0], 1)
    
    

class ScoreSDE(SDE):
    def __init__(self, score_vector_field = None, **kwargs):
        super().__init__(**kwargs)
        self.score_vector_field = score_vector_field
        self.sigma = lambda t: standard_deviation(1. - t)

    def get_velocity(self, g, t):
        times = (1. - t).unsqueeze(0).repeat(g.size()[0]).unsqueeze(1)
        _input = learning.concatenate_input(g, times)
        return self.score_vector_field(_input) * (self.sigma(t)**2)


def linear(a, b, t):
    return (b-a)*t + a


def standard_deviation(t):
    return 1.5
    # return 1.2
