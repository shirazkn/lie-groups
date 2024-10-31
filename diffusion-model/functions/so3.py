import torch, numpy
from tqdm import tqdm

import pyvista as pv
from functions import misc
from matplotlib import pyplot as plt

from pandas import DataFrame
import seaborn as sns

from scipy.spatial.transform import Rotation


def get_basis(dim=3, i=None, dtype=torch.float64, device="cpu"):
    if i is None:
        basis = torch.zeros([dimension(dim), dim, dim], 
                            dtype=dtype, device=device, requires_grad=False)
        for i in range(dimension(dim)):
            basis[i] = get_basis(dim, i+1, dtype=dtype, device=device)
        return basis
    
    mat = torch.zeros([dim, dim], dtype=dtype, 
                      device=device, requires_grad=False)
    row = dim - 2
    col = dim - 1
    parity = (-1)**i
    while (i-1) > 0:
        if col == row + 1:
            row -= 1
            col = dim - 1
        else:
            col -= 1
        i -= 1

    mat[row, col] = parity*1
    mat[col, row] = -parity*1
    return mat


def dimension(dim):
    return int(dim*(dim-1)/2)


def vee(mat):
    dim = mat.size()[-1]
    vec = torch.zeros(dimension(dim), 
                      dtype=mat.dtype, device=mat.device)
    row = dim - 2
    col = dim - 1
    for i in range(dimension(dim)):
        vec[i] = mat[row, col]*((-1)**(i+1))
        if col == row + 1:
            row -= 1
            col = dim - 1
        else:
            col -= 1
    return vec


def hat(vec, basis):
    return torch.einsum('i,ijk->jk', vec, basis)

def wrappedGaussian(mean = None, cov = None, mean_skew = None, size = 1):
    raise NotImplementedError("See the `main` branch for a working implementation")


def uniformSO3(size = 1):
    return_list = []
    for _ in range(size):
        vec = misc.random_unit_vector(4)
        return_list.append(Rotation.from_quat(vec).as_matrix())
    return return_list


def testSO3(size = 1):
    return_list = []
    for _ in range(size):
        angles = [numpy.random.normal(30, 5.0), 
                  numpy.random.uniform(-90, 90), 
                  numpy.random.normal(-30, 5.0)]
        return_list.append(Rotation.from_euler('xyx', angles, degrees=True).as_matrix())
    
    return return_list

# ---------------------------------
# ----- The following is for SO(3):
# ---------------------------------

def sliceVisualization(samples, show = True, scatter=False, 
                       color="blue", label = None, alpha=1.0):
    sphere_samples = [sample[:, 0] for sample in samples]
    data = {"theta": [], "phi": []}
    for sample in sphere_samples:
        _, theta, phi = misc.polar_from_cart(*sample)
        data['theta'].append(theta)
        data['phi'].append(phi)

    if scatter:
        sns.scatterplot(data = DataFrame(data), x = "theta", y = "phi", s = 3.5, label = label, color = color, alpha=alpha)
    else:
        sns.kdeplot(data = DataFrame(data), x = "theta", y = "phi", fill = True, levels = 100, alpha=alpha)
        if label or color:
            print("Label and color are only implemented for scatter plots.")
    plt.xlim(-numpy.pi, numpy.pi)
    plt.ylim(0, numpy.pi)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi$")
    
    if show:
        plt.show()
    

def sphereVisualization(samples , resolution = 30):
    sphere_samples = [sample[:, 0] for sample in samples]
    sphere = pv.Sphere(radius=1.0, theta_resolution = resolution, phi_resolution = resolution)
    sphere_values = pv.Sphere(radius=1.0, theta_resolution = resolution, phi_resolution = resolution)
    
    kdes = []
    kde_max = 0.0
    for i in tqdm(range(sphere_values.n_points), desc = "Calculating KDE"):
        kdes.append(kde(sphere_samples, sphere_values.points[i]))
        if kdes[-1] > kde_max:
            kde_max = kdes[-1]

    for i in range(sphere_values.n_points):
        sphere_values.points[i] = (0.99999 + kdes[i]*0.75/kde_max)*sphere_values.point_normals[i]

    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color = 'w')
    plotter.add_mesh(sphere_values, color = 'b', opacity = 0.3)
    
    plotter.add_axes()
    plotter.show()


def kde(samples, point):
    val = 0.0
    sigma = 0.1
    for sample in samples:
        val += numpy.exp(
            -numpy.inner(sample - point, sample - point)/(2*sigma**2)
            )/sigma**2

    return val/len(samples)    


def exp_map(X):
    omega = torch.sqrt((X**2).sum(dim=(1,2))*0.5)
    I = torch.eye(3, dtype=X.dtype, device=X.device, requires_grad=False)
    R = (
        I.repeat(X.size()[0], 1, 1)
        + misc.scalar_matrix_product(torch.sin(omega)/(omega), X) 
        + misc.scalar_matrix_product((1.-torch.cos(omega))/omega**2, X@X)
        )

    return R

    # if omega > 1e-5:
    #    ...
    # elif omega > 1e-9:
    #     R += (
    #         misc.scalar_matrix_product(1-omega**2/6, X) 
    #         + misc.scalar_matrix_product(0.5-omega**2/24, X@X)
    #     )