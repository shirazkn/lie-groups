import numpy as np
from functions import so

class SDE():
    def __init__(self, bases, dt = 0.05):
        self.bases = bases
        self.dim = len(bases)
        self.zero_vector = np.zeros(self.dim)
        self.dt = dt

    def flow(self, g, t):
        _t = 0.0
        dX = np.zeros([g.shape[0], g.shape[0]])
        while _t < t:
            dx = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim) * self.dt)
            for dx_i, base_i in zip(dx, self.bases):
                dX += dx_i * base_i

            g = g @ so.expm(dX)
            _t += self.dt
            dX[:] = 0.0

        return g