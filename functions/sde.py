
import torch
from numpy import sqrt

class SDE:
    def __init__(self, group, dt, drift = None):
        self.group = group
        self.dim = len(group.basis)
        self.dt = dt
        if drift is None:
            self.drift = lambda g, t: torch.zeros([g.shape[0], 
                                                   self.dim, self.dim])

    # Samples a random tangent vectors at g
    def random_tangent_at(self, g): 
        return torch.einsum('...ij,...jk->...ik', g, 
                            self.group.hat(torch.randn(g.shape[0], self.dim)))
    
    # # Simluation procedure for forward and reverse
    def simulate(self, g, final_time, debug=False):
        t = 0.0
        while t < final_time: # negative for reverse process
            if debug:
                assert torch.all(torch.isfinite(g))

            g = self.group.exp_g(g, 
                                 self.random_tangent_at(g) 
                                 * sqrt(abs(self.dt)))
            t += self.dt

        return g
    