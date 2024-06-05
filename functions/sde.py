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
            dx = self.get_velocity(g, t) * self.dt + np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim) * self.dt)
            
            dX[:] = 0.0
            for dx_i, base_i in zip(dx, self.bases):
                dX += dx_i * base_i

            g = g @ so.expm(dX)
            t += self.dt

        return g

    def get_velocity(self, g, t):
        return np.zeros(self.dim)
    

class ScoreSDE(SDE):
    def __init__(self, bases, dt, score_vector_field = None):
        super().__init__(bases, dt)
        self.score_vector_field = score_vector_field
        self.final_time = constants.simulation["final_time"]


    def get_velocity(self, g, t):
        input_vector = learning.input_from_tuple(g, self.final_time - t)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        return self.score_vector_field(input_tensor).detach().numpy()

    def flow_T(self, g):
        return super().flow(g, self.final_time)