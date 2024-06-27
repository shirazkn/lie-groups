from functions import pickler, so, neural, misc
import constants, learning

import torch
import numpy as np
from matplotlib import pyplot as plt

PLOT_DISTRIBUTIONS = False  # Otherwise, plot the vector field of the score network
SHOW_SCATTERPLOT = True

# TODO: Fix the visualization
# Plot sample diffusions of the score network

def orthonormalize(vector, wrt_unit_vector):
    vector = vector - np.dot(vector, wrt_unit_vector) * wrt_unit_vector
    return vector / np.linalg.norm(vector)


def get_rotation_matrix(phi, theta):
    # Gets a random rotation matrix from a given equivalence class in S^2 
    # (viewed as the homogenous space SO(3)/SO(2))
    columns = []
    columns.append(np.array(
        misc.cart_from_polar(1.0, phi, theta)))
    vec = misc.random_unit_vector(3)
    columns.append(orthonormalize(vec, wrt_unit_vector = columns[0]))            
    columns.append(np.cross(columns[0], columns[1]))
    return np.array(columns).T


def get_gradients(phi_grid, theta_grid, scoreNetwork, t):
    U = np.zeros_like(phi_grid)
    V = np.zeros_like(theta_grid)
    for i in range(phi_grid.shape[0]):
        for j in range(phi_grid.shape[1]):
            g = get_rotation_matrix(phi_grid[i, j], theta_grid[i, j])
            input_vector = learning.input_from_tuple(g, t)
            input_tensor = torch.tensor(input_vector, dtype=constants.datatype)
            score_vector = scoreNetwork(input_tensor).detach().numpy()
            g_plus = g @ so.hat(so.hat(score_vector)*0.01)

            new_coords = misc.polar_from_cart(g_plus[0, 0], g_plus[1, 0], g_plus[2, 0])
            U[i, j] = new_coords[1] - phi_grid[i, j]
            V[i, j] = new_coords[2] - theta_grid[i, j]  
    return U, V


if __name__ == "__main__":
    if PLOT_DISTRIBUTIONS:
        samples = pickler.read_all(constants.samples_filename)

        plt.figure("Target pdf")
        so.sliceVisualization(samples, show=False, scatter=SHOW_SCATTERPLOT)

        training_dataset = pickler.read_all(constants.diffused_samples_filename)
        diffused_samples = []
        large_time = 0.8*constants.simulation["final_time"]
        for g, t in training_dataset:
            if t > large_time:
                diffused_samples.append(g)
        plt.figure(f"{len(diffused_samples)} Samples with t > {large_time}")
        so.sliceVisualization(diffused_samples, show=False, scatter=SHOW_SCATTERPLOT)

        plt.figure("Uniform SO(3) samples")
        so.sliceVisualization(so.uniformSO3(len(diffused_samples)), show=False, scatter=SHOW_SCATTERPLOT)
        plt.show()

    # ---- visualizes the vector field of the score network ----    
    else:
        scoreNetwork = neural.Feedforward(*constants.feedforward_signature)
        scoreNetwork.load_state_dict(torch.load(constants.model_filename))
        scoreNetwork.eval()
        X = np.linspace(-np.pi, np.pi, 10)
        Y = np.linspace(0, np.pi, 10)
        X_grid, Y_grid = np.meshgrid(X, Y)

        def add_figure(time):
            plt.figure(f"Quiver at t = {time} ")
            U_grid, V_grid = get_gradients(X_grid, Y_grid,
                                           scoreNetwork=scoreNetwork, t=time)
            plt.quiver(X_grid, Y_grid, U_grid, V_grid)

        for time in [2.0, 1.0, 0.25]:
            add_figure(time)
        
        plt.show()


