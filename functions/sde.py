import torch
import numpy as np
from functions import so
import constants, learning

class SDE():
    """
    Generates sample paths of an SDE defined on matrix Lie groups
    Currently only supports a pure diffusion process, i.e., Brownian motion
    """
    def __init__(self, bases, dt):
        self.bases = bases
        self.dim = len(bases)
        self.dt = dt

    def flow(self, g, T):
        t = 0.0
        dX = np.zeros([g.shape[0], g.shape[0]])
        while t < T:
            dx = self.get_velocity(g, t) * self.dt 
            + self.standard_deviation(t) * np.sqrt(self.dt) * standard_normal(self.dim)
            
            dX[:] = 0.0
            for dx_i, base_i in zip(dx, self.bases):
                dX += dx_i * base_i

            g = g @ so.expm(dX)
            t += self.dt

        return g

    def get_velocity(self, g, t):
        return np.zeros(self.dim)
    
    @staticmethod
    def standard_deviation(t):
        return linear_schedule(t)
    

class ScoreSDE(SDE):
    def __init__(self, bases, dt, score_vector_field = None):
        super().__init__(bases, dt)
        self.score_vector_field = score_vector_field


    def get_velocity(self, g, t):
        input = learning.concatenate_input(torch.tensor([g]), 
                                           torch.tensor([1. - t]))
        return self.score_vector_field(input.to(dtype=constants.datatype)).detach().numpy() * self.standard_deviation(t)**2
    

    @staticmethod
    def standard_deviation(t):
        return linear_schedule(1. - t)


def standard_normal(dim):
    return np.random.multivariate_normal(np.zeros(dim), np.eye(dim))


def linear(a, b, t):
    return (b-a)*t + a


def linear_schedule(t):
    return linear(0.01, 10., t)