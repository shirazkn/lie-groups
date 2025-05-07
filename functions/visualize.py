import numpy as np
import torch
import se3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def add_frame(g, ax=None, alpha=1.0, color='k'):
    """
    Visualize an SE(3) frame with an option to set the opacity and color.
    """
    import matplotlib.pyplot as plt
    R = g[:3, :3]
    t = g[:3, 3]

    # Define the axes of the frame
    scale = 0.5
    x_axis = torch.tensor([1., 0., 0.])
    y_axis = torch.tensor([0., 1., 0.])
    z_axis = torch.tensor([0., 0., 1.])
    x_transformed = scale * R @ x_axis
    y_transformed = scale * R @ y_axis
    z_transformed = scale * R @ z_axis

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.scatter(t[0].item(), t[1].item(), t[2].item(), color=color, s=10, alpha=alpha, zorder=150)

    # Draw arrows
    ax.quiver(t[0].item(), t[1].item(), t[2].item(), x_transformed[0].item(), x_transformed[1].item(), x_transformed[2].item(), color=color, alpha=alpha, linewidth=0.8, zorder=150)
    ax.quiver(t[0].item(), t[1].item(), t[2].item(), y_transformed[0].item(), y_transformed[1].item(), y_transformed[2].item(), color=color, alpha=alpha, linewidth=0.8, zorder=150)
    ax.quiver(t[0].item(), t[1].item(), t[2].item(), z_transformed[0].item(), z_transformed[1].item(), z_transformed[2].item(), color=color, alpha=alpha, linewidth=0.8, zorder=150)

    return ax


def add_point(t, ax=None, shape='o', color='k'):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    if shape == 'cone':
        height = 0.2
        radius = 0.05
        num_sides = 20
        theta = np.linspace(0, 2 * np.pi, num_sides)
        x = radius * np.cos(theta) + t[0].item()
        y = radius * np.sin(theta) + t[1].item()
        z = np.full_like(x, t[2].item())
        base = np.vstack((x, y, z)).T
        sides = [np.array([[t[0].item(), t[1].item(), t[2].item() + height], base[i], base[(i + 1) % num_sides]]) for i in range(num_sides)]

        # Add the base and sides to the plot
        ax.add_collection3d(Poly3DCollection([base], color=color, alpha=1., zorder=150))
        ax.add_collection3d(Poly3DCollection(sides, color=color, alpha=1., zorder=150))
    else:
        ax.scatter(t[0].item(), t[1].item(), t[2].item(), color=color)

    return ax

def add_line(p1, p2, ax=None, color='k', linestyle='-', linewidth=1.0):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot([p1[0].item(), p2[0].item()], [p1[1].item(), p2[1].item()], [p1[2].item(), p2[2].item()], linestyle=linestyle, color=color, linewidth=linewidth, zorder=50)

    return ax

def set_limits(ax, lim):
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.grid(False)
    ax.set_axis_off()

    # Create a shaded plane for z=0
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 10), np.linspace(-lim, lim, 10))
    ax.plot_surface(xx, yy, 0.0 * np.ones_like(xx), color=(0.88, 0.88, 0.895), alpha=0.75, rstride=100, cstride=100, shade=False, zorder=15)

    # Create a second shaded plane for z=-0.1 to give it thickness
    ax.plot_surface(xx, yy, -0.01 * np.ones_like(xx), color=(0.3, 0.3, 0.32), alpha=0.2, rstride=100, cstride=100, shade=False, zorder=0)

    # Add a faint grid on the plane
    for i in np.arange(-lim, lim + 0.5, 0.5):
        ax.plot([i, i], [-lim, lim], [0, 0], color='gray', alpha=0.3, linewidth=0.25, zorder=20)
        ax.plot([-lim, lim], [i, i], [0, 0], color='gray', alpha=0.3, linewidth=0.25, zorder=20)

    # Add a slightly darker line for the lines that pass through the origin
    ax.plot([0, 0], [-lim, lim], [0, 0], color='gray', alpha=0.1, linewidth=0.35, zorder=15)
    ax.plot([-lim, lim], [0, 0], [0, 0], color='gray', alpha=0.1, linewidth=0.35, zorder=15)

    # Adjust the view so that the x and y axes are going from left to right
    ax.view_init(elev=15, azim=42)
    
    return ax



def plot_trajectory(estimates, axis, num_frames=4, color = 'k'):
    indices = [int(i) 
                for i in torch.linspace(0, len(estimates)-1, num_frames)]

    for i in indices:
        add_frame(estimates[i], ax=axis, color=color,
                  alpha=0.1 + 0.9*i/len(estimates))
        

def plot_trail(estimates, axis, color='k'):
    for i in range(1, len(estimates)):
        alpha = 0.25 + 0.5 * i / len(estimates)
        line = add_line(se3.get_t(estimates[i-1]), 
                        se3.get_t(estimates[i]), ax=axis, color=color, linewidth=0.4)
        line.lines[-1].set_alpha(alpha)
            

def show():
    plt.show()
