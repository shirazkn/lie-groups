"""
This code re-implements the notebook, 
https://colab.research.google.com/github/blt2114/SO3_diffusion_example/blob/main/SO3_diffusion_example.ipynb#scrollTo=iOX8ZsCsOvEj
"""

import add_to_path
import torch

# torch.multiprocessing.set_start_method("spawn")
from functions import neural, so3, misc

import numpy as np
from tqdm import tqdm
import os
np.random.seed(42)

import matplotlib.pyplot as plt


### Simulate the geodesic random walk
n_samples = 10000  # Number of samples
T = 5  # Final time 


# Evaluates the isotropic Gaussian density on SO(3) defined w.r.t. the Haar measure; this is the true heat kernel
def igso3_from_angle(angles, t, L=500):
    ls = torch.arange(L)[None]  # of shape [1, L]
    return ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
             torch.sin(angles[:, None]*(ls+1/2)) 
             / torch.sin(angles[:, None]/2)).sum(dim=-1)


def approx_igso3_from_angle(angles, t):
    "See 'Unified framework for generative models in SO(3)...', Jagvaral et al"
    raise NotImplementedError

# Computes the det(J) term, which must be multiplied (resp., divided) to pull a density back from (resp., push a density forward to) G
def det_J(angles):
    return 2*(1-torch.cos(angles))/(angles**2 * 8 * np.pi**2)

# Evaluates the exponential Gaussian on SO(3) defined w.r.t. the Haar measure
def egso3_from_angle(angles, t):
    return (torch.exp(-angles**2/(2*t))/(2*np.pi*t)**(3./2.))/det_J(angles)

def igso3(rotations, t, L=500): 
    return igso3_from_angle(so3.get_angle(rotations), t, L)

# Evaluates the marginal density of rotation angle for uniform density on SO(3)
def uniformso3_marginal(angles):
    return (1-torch.cos(angles))/np.pi

# Samples a random tangent vector at R
def random_tangent_at(R): 
    return torch.einsum('...ij,...jk->...ik', R, 
                        so3.hat(torch.randn(R.shape[0], 3)))

# # Simluation procedure for forward and reverse
def geodesic_random_walk(R_initial, drift, times):
    rotations = {times[0]: R_initial()}
    for i in range(1, len(times)):
        dt = times[i] - times[i-1] # negative for reverse process
        rotations[times[i]] = so3.exp_g(rotations[times[i-1]], 
            drift(rotations[times[i-1]], times[i-1]) * dt 
            + random_tangent_at(rotations[times[i-1]]) * np.sqrt(abs(dt)))
             
    return rotations


times = np.linspace(0, T, 500)  # Discretization of [0, T]
random_walk = geodesic_random_walk(
    R_initial=lambda: so3.exp(torch.zeros(n_samples, 3, 3)), 
    drift=lambda Rt_tensor, t_tensor: 0., 
    times=times)

t_idcs_plot = [10, 50, 100, -1]
_, axs = plt.subplots(1, len(t_idcs_plot), dpi=100, 
                      figsize=(3*len(t_idcs_plot), 3))

print("Generating plots...")
for i, t_idx in enumerate(t_idcs_plot):
  
  # Plot empirical distribution of angle of rotation from geodesic random walk
  bins = np.linspace(0, np.pi, 25)
  axs[i].hist(so3.get_angle(random_walk[times[t_idx]]).cpu(), bins=bins, density=True, histtype='step', label='Empirical')
  axs[i].set_title(f"t={times[t_idx]:0.01f}")

  
  # Compute density on angle of rotation, and the density for the uniform distribution
  angle_grid = torch.linspace(0, np.pi, 1000)
  
  L=20
  pdf_angle = igso3_from_angle(angle_grid, times[t_idx], L=L) \
    * uniformso3_marginal(angle_grid)  # Turns the density into a marginal
  axs[i].plot(angle_grid.cpu(), pdf_angle.cpu().numpy(), 'r-',
              label=f"Iso. Gaussian ({L} terms)")
  
  L = 4
  pdf_angle = igso3_from_angle(angle_grid, times[t_idx], L=L) \
    * uniformso3_marginal(angle_grid)  # Turns the density into a marginal
  axs[i].plot(angle_grid.cpu(), pdf_angle.cpu().numpy(), '-', color="orange",
              label=f"Iso. Gaussian ({L} terms)")
  
  pdf_angle = egso3_from_angle(angle_grid, times[t_idx]) \
    * uniformso3_marginal(angle_grid)  # Turns the density into a marginal
  axs[i].plot(angle_grid.cpu(), pdf_angle.cpu().numpy(),  'k-',
              label="Exp. Gaussian")
  
  axs[i].plot(angle_grid.cpu(), uniformso3_marginal(angle_grid).cpu().numpy(), 'k--', label="Uniform")

  axs[i].set_xlabel("Angle of rotation (radians)")

axs[-1].legend()
axs[0].set_ylabel("p(angle)")

plt.suptitle("Agreement of IGSO3 density and geodesic random walk", y=1.05)
plt.tight_layout()
plt.show()