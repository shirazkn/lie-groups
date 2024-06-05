"""
I'm trying to compare the Gaussians defined in the book, "Harmonic Analysis for Engineers and Applied Scientists."
- - - - - - - - - - - - - - - -
Exponential Gaussian: eq. 16.49
Heat Kernel Gaussian: eq. 16.55
Chetalet's Gaussian: eq. 16.60 and 16.64

Bug: Why doesn't the heat kernel Gaussian seem to integrate to 1?
"""

import numpy as np
# from tqdm import tqdm
from matplotlib import pyplot as plt


def plot_all(Kt, lim, resolution):
    # Nudging everything by 1e-4 avoids computations at 0
    angles_rad = np.linspace(lim[0], lim[1] + 1e-4, resolution) 

    exponential_gaussian = []
    heatkernel_gaussian = []
    chetalet_gaussian = []
    for theta in angles_rad:
        # Exponential Gaussian
        exponential_gaussian.append(
            np.exp(-0.5 * (theta)**2/Kt)/((2*np.pi*Kt)**1.5)
        )

        # Heat Kernel Gaussian
        heatkernel_gaussian_val = 0.0
        for l in range(0, 1000):
            heatkernel_gaussian_val += (
                (2.0*l + 1.0) 
            * (np.sin((l + 0.5)*theta) / np.sin(theta/2))
            * np.exp(-l*(l+1)*Kt)
            )
        heatkernel_gaussian.append(heatkernel_gaussian_val)

        # Chetalet's Gaussian
        chetalet_gaussian.append(
            (np.exp(Kt)
            * np.exp(-theta**2 / (4*Kt))
            * theta/np.sin(theta/2)) / (2*(np.pi*Kt)**1.5)
        )

        # TODO: 
        # Try the approximation given in "Unified framework for diffusion generative models in SO(3)" by Jagvaral et al.
        # Information-geometric interpolation between the exponential and heat kernel Gaussians? Linear interpolation?

    plt.plot(angles_rad, exponential_gaussian, label="Exponential Gaussian")
    plt.plot(angles_rad, heatkernel_gaussian, label="Heat Kernel Gaussian")
    plt.plot(angles_rad, chetalet_gaussian, label="Chétalet's Gaussian")
    plt.xlabel("angle (rad)")
    plt.ylabel("pdf")
    plt.title(f"Comparing Gaussians for Kt = {Kt}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_all(Kt = 0.1, lim = [-0.5, 0.5], resolution = 1000)
    plot_all(Kt = 0.1, lim = [-np.pi, np.pi], resolution = 1000)