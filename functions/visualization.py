from functions import misc
import matplotlib.pyplot as plt
from numpy import pi

import seaborn as sns


def so3_sphere2D(samples, show = True, scatter=False, 
                       color="blue", label = None, alpha=1.0):
    sphere_samples = [sample[:, 0] for sample in samples]
    data = {"theta": [], "phi": []}
    for sample in sphere_samples:
        _, theta, phi = misc.polar_from_cart(*sample)
        data['theta'].append(theta)
        data['phi'].append(phi)

    if scatter:
        sns.scatterplot(data, x = "theta", y = "phi", s = 3.5, 
                        label = label, color = color, alpha=alpha)
    else:
        sns.kdeplot(data, x = "theta", y = "phi", fill = True, 
                    levels = 100, alpha=alpha)
        if label or color:
            print("Label and color are only implemented for scatter plots.")
    plt.xlim(-pi, pi)
    plt.ylim(0, pi)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi$")
    
    if show:
        plt.show()