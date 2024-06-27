import add_to_path
import matplotlib.pyplot as plt
from functions import sde, so3, misc, visualization


g = sde.SDE(group=so3, dt=0.01).simulate(misc.identity(100), 0.01)

plt.figure("SO(3) Diffusion after 0.01 seconds")
visualization.so3_sphere2D(g, scatter=True, show=False)

g = sde.SDE(group=so3, dt=0.01).simulate(g, 0.01)
plt.figure("SO(3) Diffusion after 0.02 seconds")
visualization.so3_sphere2D(g, scatter=True, show=False)

g = sde.SDE(group=so3, dt=0.01).simulate(g, 0.03)
plt.figure("SO(3) Diffusion after 0.05 seconds")
visualization.so3_sphere2D(g, scatter=True, show=False)

plt.show()

