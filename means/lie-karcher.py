import sys, os
from tqdm import tqdm
import torch as t
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../functions')))

import se2, so3, misc
group = se2

group_dim = 3
matrix_dim = 3

num_samples = 10
# L = t.rand([3, 3])
L = t.eye(3)

samples = t.zeros([2*num_samples, matrix_dim, matrix_dim])
for i in range(num_samples):
    random_vector = t.randn(3)
    samples[2*i, :, :] = group.exp_map(t.matmul(L, random_vector))
    samples[2*i + 1, :, :] = group.exp_map(-t.matmul(L, random_vector))

dir = t.tensor([1, 1, 1])
dir = dir/t.sqrt((dir*dir).sum())

x_axis = t.linspace(-1, 1, 100)
y_axis = []
for scalar in tqdm(x_axis):
    mean_inv = group.exp_map(-scalar*dir)
    sum_dsq = 0.0
    for i in range(2*num_samples):
        error = group.log_map(t.matmul(mean_inv, samples[i]))
        sum_dsq += (error**2).sum()

    y_axis.append(sum_dsq)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3), dpi=200)
plt.rcParams.update({'font.size': 7})
# plt.subplots_adjust(bottom=0.2)
plt.plot(x_axis, y_axis)
plt.xlabel(r"$t$ (such that $\mu=\exp(t \tilde X)$)")
plt.ylabel("Sum of Squared Distances")
plt.title("Lie-Karcher Functional for Symmetric Data on $SE(2)$")
plt.tight_layout()
plt.show()
