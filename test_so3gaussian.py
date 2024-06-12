"""
Compares the Gaussians defined in the book, "Harmonic Analysis for Engineers and Applied Scientists."
- - - - - - - - - - - - - - - -
Exponential Gaussian: eq. 16.49
Heat Kernel Gaussian: eq. 16.55
Chetalet's Gaussian: eq. 16.60 and 16.64

Note:
- Chetalet's Gaussian doesn't look right
- The Heat Kernel Gaussian can be computed faster and on GPU using torch (https://colab.research.google.com/github/blt2114/SO3_diffusion_example/blob/main/SO3_diffusion_example.ipynb#scrollTo=rj1zgqcIPBbN)
- We should also look at the approximation given in "Unified framework for diffusion generative models in SO(3)" by Jagvaral et al.
"""

import numpy as np
# from tqdm import tqdm
from matplotlib import pyplot as plt


def plot_all(Kt, lim, resolution):
    plt.figure()

    # Nudging everything by epsilon avoids computations at 0
    angles_rad = np.linspace(lim[0], lim[1] + 5e-5, resolution) 

    exponential_gaussian = []
    heatkernel_gaussian = []
    det_J_list = []
    chetalet_gaussian = []
    for theta in angles_rad:
        det_J =  2*(1-np.cos(theta)) / (theta**2 * 8 * np.pi**2)
        # if MARGINAL:
        #     det_J *= 
        
        det_J_list.append(det_J)

        # Exponential Gaussian  (with D = 2*K*I)
        exponential_gaussian.append(
            np.exp( -0.5 * theta**2 / (2*Kt) )/( (2*np.pi*2*Kt)**1.5 )
        )

        # Heat Kernel Gaussian
        heatkernel_gaussian_val = 0.0
        for l in range(0, 500):
            heatkernel_gaussian_val += (
                (2.0*l + 1.0) 
            * np.sin((l + 0.5)*theta) / np.sin(theta/2)
            * np.exp(-l*(l+1.0)*Kt)
            )
        heatkernel_gaussian.append(heatkernel_gaussian_val * det_J)

        # Chetalet's Gaussian (Needs a det_J term? Off by a constant factor?)
        # chetalet_gaussian.append(
        #     np.exp(Kt) * np.exp(-theta**2 / (4*Kt)) * theta 
        #     / ( 2*(np.pi*Kt)**1.5 * np.sin(theta/2) )
        # )


    plt.plot(angles_rad, exponential_gaussian, 'b-', label="Exponential Gaussian")
    plt.plot(angles_rad, heatkernel_gaussian, 'r--', label="Heat Kernel Gaussian")
    # plt.plot(angles_rad, chetalet_gaussian, label="Chétalet's Gaussian")
    plt.xlabel("angle (rad)")
    plt.ylabel("pdf")
    plt.title(f"Comparing Gaussians for Kt = {Kt}")
    plt.legend()


if __name__ == "__main__":
    # plot_all(Kt = 0.1, lim = [-0.5, 0.5], resolution = 1000)
    plot_all(Kt = 0.01, lim = [0, np.pi], resolution = 1000)
    plot_all(Kt = 0.05, lim = [0, np.pi], resolution = 1000)
    plot_all(Kt = 0.1, lim = [0, np.pi], resolution = 1000)
    plot_all(Kt = 0.5, lim = [0, np.pi], resolution = 1000)
    plt.show()