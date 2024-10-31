import torch as t
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

t.set_default_dtype(t.float64)
points = t.tensor([
    [0.5, 2.8],
    [0.5, 2.4],
    [0.0, 3.2],
    [1.0, 3.2],
    [1.0, 2.8]
])

# Define edges to connect the points so that the graph is rigid
edges = [
    (4, 1),
    (4, 0),
    (4, 3),
    (0, 3),
    (0, 2),
    (0, 1),
    (3, 2)
]

import matplotlib.pyplot as plt

def visualize_graph(points, edges, color='red', ax=None, show_edges=True, alpha=0.7, show_labels=True, zorder=1):
    x_coords = points[:, 0].numpy()
    y_coords = points[:, 1].numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(x_coords, y_coords, c=color, alpha=alpha, zorder=zorder)
    # Label the nodes using LaTeX if show_labels is True
    offsets = [(0.02, 0.03), (-0.035, 0.005), (-0.01, 0.015), (0.065, 0.005), (0.065, 0.015)]
    if show_labels:
        for i, (x, y, offset) in enumerate(zip(x_coords, y_coords, offsets)):
            ax.text(x + offset[0], y + offset[1], f'${i + 1}$', fontsize=12, ha='right', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', 
                          pad=0.0), zorder=zorder)

    if show_edges:
        for edge in edges:
            point1, point2 = edge
            x_values = [points[point1, 0], points[point2, 0]]
            y_values = [points[point1, 1], points[point2, 1]]
            ax.plot(x_values, y_values, color, alpha=alpha, linewidth=1.1, linestyle='solid', zorder=zorder)

    return ax

m_shift = points.clone()
m_shift[1] += t.tensor([0.0, 0.09])
m_shift[2] += t.tensor([-0.07, -0.06])
m_shift[3] += t.tensor([0.008, -0.007])
m_shift[4] += t.tensor([-0.03, 0.032])
ax = visualize_graph(m_shift, edges, color='#27BB31', alpha=0.8, show_edges=True, show_labels=False, zorder=0)

visualize_graph(points, edges, color='black', alpha=1.0,
                show_edges=True, show_labels=True, ax=ax, zorder=10)

import se2
poses = []
for point in points:
    poses.append(se2.get_pose(t.tensor(0.), point))


trails = [[se2.get_t(pose)] for pose in poses]
for k in range(100):
    vel = t.zeros(3)
    vel[0] = t.sin(t.tensor(k**1.2 * 0.005))*0.3
    vel[1] = 1.5
    vel[2] = -0.
    delta = se2.exp_map(vel*0.01)
    for i in range(len(poses)):
        poses[i] = delta @ poses[i]
        trails[i].append(se2.get_t(poses[i]))

def plot_trails(ax, trail, color='k', max_opacity=0.3):
    x_coords = [point[0].item() for point in trail]
    y_coords = [point[1].item() for point in trail]
    num_points = len(trail)
    for i in range(1, num_points):
        alpha = max_opacity * (i / num_points)
        ax.plot(x_coords[i-1:i+1], y_coords[i-1:i+1], linestyle='dashed',
                color=color, alpha=alpha, linewidth=0.5)

for trail in trails:
    plot_trails(ax, trail[40:], color='darkred')

extracted_points = t.cat([se2.get_t(pose).unsqueeze(0) 
                          for pose in poses], dim=0)

# Visualize the extracted points
visualize_graph(extracted_points, edges, color='#CC1100', alpha=1.0, show_edges=True, show_labels=False, ax=ax)

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_aspect('equal', adjustable='box')

plt.savefig('images/distance_localization_green.png', dpi=700)
import os
os.system('images/distance_localization_green.png')
# plt.show()
