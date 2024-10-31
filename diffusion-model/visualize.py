from functions import pickler, neural, misc, so3, sde
import constants, learning
from tqdm import tqdm

import torch
import numpy as np
from matplotlib import pyplot as plt

VIS_TYPES = ["diffusion", "score_vf", "left_invariant_vf", "losses"]
vis_type = "diffusion"


SHOW_SCATTERPLOT = False
 

def constScoreNetwork(_output):
    return lambda _input: _output.unsqueeze(0)


def orthonormalize(vector, wrt_unit_vector):
    vector = vector - np.dot(vector, wrt_unit_vector) * wrt_unit_vector
    return vector / np.linalg.norm(vector)


def get_rotation_matrix(theta, phi):
    # Gets a random rotation matrix from a given equivalence class in S^2 
    # (viewed as the homogenous space SO(3)/SO(2))
    columns = []
    columns.append(np.array(
        misc.cart_from_polar(1.0, theta, phi)))
    a_vector = np.array([1., 2., 3.])
    columns.append(orthonormalize(a_vector, wrt_unit_vector = columns[0]))            
    columns.append(np.cross(columns[0], columns[1]))
    return np.array(columns).T


def get_gradients(theta_grid, phi_grid, scoreNetwork, time):
    basis = so3.get_basis(3, dtype=constants.datatype, 
                          device=neural.get_device())
    U = np.ones_like(theta_grid)
    V = np.ones_like(phi_grid)
    dtype = scoreNetwork.layers[0].weight.dtype
    device = scoreNetwork.layers[0].weight.device
    t = torch.tensor([time], dtype=dtype, device=device)
    for i in range(phi_grid.shape[0]):
        for j in range(phi_grid.shape[1]):    
            g = torch.tensor(get_rotation_matrix(theta_grid[i, j], 
                                                phi_grid[i, j]), 
                                                dtype=dtype, device=device)
            input_tensor = learning.concatenate_input(g.unsqueeze(0), t)
            score_vector = scoreNetwork(input_tensor)[0]
            # score_vector = torch.tensor([0., 0., -1.], 
            #                             dtype=dtype, device=device)
            g_plus = g @ so3.exp_map(so3.hat(score_vector, basis)*0.001)
            g_plus = g_plus.detach().numpy()

            new_coords = misc.polar_from_cart(*g_plus[:, 0])
            U[i, j] = new_coords[1] - theta_grid[i, j]
            V[i, j] = new_coords[2] - phi_grid[i, j]  

    return U, V


if __name__ == "__main__":
    dtype = constants.datatype
    device = neural.get_device()
    if vis_type == "diffusion":
        dataset = pickler.read_all(constants.samples_filename)[:]

        plt.figure("Target pdf")
        so3.sliceVisualization(dataset, show=False, scatter=SHOW_SCATTERPLOT)

        for t in [0.01, 0.1, 1.0]:
            diffused_samples = []
            diffuser = sde.SDE(group=so3, device=device)
            g = diffuser.flow(
                torch.tensor(dataset, dtype=dtype,device=device), t
                )
            diffused_samples = g.detach().cpu().numpy()
            
            plt.figure(f"{len(diffused_samples)} Diffused Samples at t = {t}")
            so3.sliceVisualization(diffused_samples, show=False, scatter=SHOW_SCATTERPLOT)

        plt.figure("Uniform SO(3) samples")
        so3.sliceVisualization(so3.uniformSO3(len(diffused_samples)), show=False, scatter=SHOW_SCATTERPLOT)
        plt.show()

    # ---- visualizes the vector field of the score network ----    
    elif vis_type == "score_field":
        scoreNetwork = neural.Feedforward(*constants.feedforward_signature)
        scoreNetwork.load_state_dict(torch.load(constants.model_filename))
        scoreNetwork.eval()
        X = np.linspace(-np.pi, np.pi, 20)
        Y = np.linspace(0, np.pi, 20)
        X_grid, Y_grid = np.meshgrid(X, Y)

        def add_figure(time):
            plt.figure(f"Quiver at t = {time} ")
            U_grid, V_grid = get_gradients(X_grid, Y_grid,
                                           scoreNetwork=scoreNetwork, 
                                           time=time)
            norm = np.sqrt(U_grid**2 + V_grid**2)
            U_grid = U_grid / norm
            V_grid = V_grid / norm
            plt.quiver(X_grid, Y_grid, U_grid, V_grid)

        for time in [1.0, 0.25]:
            add_figure(time)
        
        plt.show()

    elif vis_type == "left_invariant_vf":
        a_tensor = torch.zeros(3, 
                                  dtype=constants.datatype, device=neural.get_device())
        a_tensor[0] = 1.
        dummyScoreNetwork = constScoreNetwork(a_tensor)
        
        X = np.linspace(-np.pi, np.pi, 20)
        Y = np.linspace(0, np.pi, 20)
        X_grid, Y_grid = np.meshgrid(X, Y)

        def add_figure(time):
            plt.figure(f"Quiver at t = {time} ")
            U_grid, V_grid = get_gradients(X_grid, Y_grid,
                                           scoreNetwork=dummyScoreNetwork, 
                                           time=time)
            norm = np.sqrt(U_grid**2 + V_grid**2)
            U_grid = U_grid / norm
            V_grid = V_grid / norm
            plt.quiver(X_grid, Y_grid, U_grid, V_grid)

        add_figure(0.0)
        plt.show()

    elif vis_type == "losses":
        scoreNetwork = neural.Feedforward(*constants.feedforward_signature)
        scoreNetwork.load_state_dict(torch.load(constants.model_filename))
        scoreNetwork.eval()
        