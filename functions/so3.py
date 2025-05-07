from soN import *


basis = get_basis(3)

def hat(vec):
    return torch.einsum('...,...jk->jk', vec, basis)

def exp_map(x):
    X = hat(x)
    omega = torch.sqrt((X**2).sum()*0.5)
    R = torch.eye(X.size(1), requires_grad=False)
    if omega > 1e-8:
       R += (
           misc.scalar_matrix_product(torch.sin(omega)/(omega), X) 
           + misc.scalar_matrix_product((1.-torch.cos(omega))/omega**2, X@X)
        )
    else:
        R += (
            misc.scalar_matrix_product(1-omega**2/6, X) 
            + misc.scalar_matrix_product(0.5-omega**2/24, X@X)
        )
    return R


## ------ MISC FUNCTIONS ------
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