import se2, torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

torch.set_default_dtype(torch.float64)
# torch.set_default_device('mps')

# Use LaTeX for rendering text
rc('text', usetex=True)
rc('font', family='serif')


def add_frame(g, label=None, color='k'):
    # G is an SE(2) Matrix
    loc = g[:2, 2]
    R = g[:2, :2]
    plt.arrow(loc[0], loc[1], R[0, 0], R[1, 0], head_width=0.05, head_length=0.1, fc='r', ec='r')
    plt.arrow(loc[0], loc[1], R[0, 1], R[1, 1], head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.plot(loc[0], loc[1], 'ko')
    plt.annotate(f'{label}',
                 xy=(loc[0], loc[1]),
                 xytext=(4, -15),  # 4 points vertical offset.
                 textcoords='offset points',
                 ha='center', va='bottom',
                 color=color)


if __name__ == '__main__':
    g = []
    g.append(se2.get_pose(torch.tensor(np.radians(45)), torch.tensor([3, 4])))
    g.append(se2.get_pose(torch.tensor(np.radians(15)), torch.tensor([8, 2])))
    g.append(se2.get_pose(torch.tensor(np.radians(-25)), torch.tensor([9,7])))
    g.append(se2.get_pose(torch.tensor(np.radians(135)), torch.tensor([4, 9])))
    g.append(se2.get_pose(torch.tensor(np.radians(135)), torch.tensor([10, 12])))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 4), (3, 4)]

    plt.figure(figsize=(8, 8))
    for edge in edges:
        i, j = edge
        plt.plot([g[i][0, 2], g[j][0, 2]], [g[i][1, 2], g[j][1, 2]], 'k--', alpha = 0.7, linewidth=0.6)

    h = se2.get_pose(torch.tensor(np.radians(40)), torch.tensor([4, 4]))
    plt.axis('equal')
    plt.xlim(-2, 17)

    for i in range(len(g)):
        add_frame(g[i], r"$g_" + f"{i+1}" + r"$")
    
    for i in range(len(g)):
        add_frame(h @ g[i], r"$h g_" + f"{i+1}" + r"$", color='r')
    
    plt.show()
    print("Done")

    # g = se2.get_pose(torch.tensor(np.radians(45)), torch.tensor([5., 2.]))
    # h = se2.get_pose(torch.tensor(np.radians(15)), torch.tensor([0., 0.]))

    # plt.figure(figsize=(8, 8))
    # # plt.axis('equal')
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    
    # for pose in [g, h @ g, h@ h @ g]:
    #     add_frame(pose)
    #     measurement = torch.linalg.inv(pose) @ torch.tensor([0., 0., 1.]).unsqueeze(1)
    #     print(measurement)
    
    # plt.show()
    # print("Done")
