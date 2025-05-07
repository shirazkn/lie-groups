"""
Code for the paper "Parameter Estimation on Homogeneous Spaces"
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use `python landmarks.py -s <SIMULATION_NUMBER>` to run the simulation.
Simulation numbers 0-2 generate the plots shown in the paper.
Simulation number 3 verifies the Ad_H invariance property of the inner product.
"""

import torch, numpy
from tqdm import tqdm

import so3, se3, visualize, misc

from matplotlib import pyplot as plt
import argparse
import os
import pickle
from matplotlib.lines import Line2D

torch.set_default_dtype(torch.float64)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

SIMULATION_TYPES = {0: "multiple_start",
                    1: "FIM",
                    2: "fisher_scoring",
                    3: "ad-invariance"
                    }

LOAD_NAME = 'simulation_data_100000.pkl'
parser = argparse.ArgumentParser(
    description='Run landmark parameter estimation simulations.')
parser.add_argument('-s', '--simulation_type', 
                    type=int, choices=[0, 1, 2, 3], default=0, help='Type of simulation to run')
parser.add_argument('-l', '--load_data', 
                    action='store_true', help='Load data from file')

args = parser.parse_args()

LOAD_DATA = False
if args.load_data:
    LOAD_DATA = True
SIMULATION_TYPE = args.simulation_type
SAVE_FIG = True
landmarks = torch.tensor([[-1.5, 0., 0.], [1.5, 0., 0.]])


def base_plot(true_pose, landmarks):
    axis = visualize.add_frame(true_pose)
    for a in landmarks: 
        visualize.add_point(a, ax=axis, shape='cone', color='seagreen')
    
    return axis

def get_measurements(g, a, num_measurements=1):
    mean = se3.get_R(g).t() @ (a - se3.get_t(g))
    return  mean.repeat(num_measurements, 1) + torch.randn(num_measurements, 3)


def get_h_basis(landmarks):

    def velocity(skew, landmark):
            Q = so3.exp_map(skew)
            J_inv = so3.inverse_Jacobian(skew)
            return misc.matrix_vector_product(J_inv @ (torch.eye(3) - Q),
                                                landmark)
    
    if len(landmarks) == 1:  
        # The problem has a three-dimensional symmetry group.
        basis = torch.zeros(3, 6)
        for i, skew in enumerate(so3.get_basis(3)):
            vel = velocity(skew, landmarks[0])
            basis[i, :] = torch.cat((so3.vee(skew), vel), dim=0)

    elif len(landmarks) == 2:
        # The problem has a one-dimensional symmetry group, 
        # provided the landmarks are not coincident.
        basis = torch.zeros(1, 6)
        skew_vec = misc.normalize(landmarks[1] - landmarks[0])
        skew = so3.hat(skew_vec, se3.so3_basis)
        vel = velocity(skew, landmarks[0])
        basis[:] = torch.cat((skew_vec, vel), dim=0)

    else:
        # The problem does not have a symmetry,
        # provided the landmarks are not collinear.
        return torch.zeros(0, 6)

    return basis

def get_gradient_sum(g, a, x_sum, basis):
    R = se3.get_R(g)
    t = se3.get_t(g)
    grad = torch.zeros(len(basis))
    for i in range(len(basis)):
        skew = so3.hat(basis[i,:3], se3.so3_basis)
        v = basis[i,3:]
        grad[i] = misc.inner(
            misc.matrix_vector_product(skew, a) + v,
            a - t - misc.matrix_vector_product(R, x_sum)
        )
    return grad


def get_FIM(a, m_basis):
    FIM = torch.zeros(m_basis.size(0), m_basis.size(0))
    for i in range(m_basis.size(0)):        
        skew_i = so3.hat(m_basis[i,:3], se3.so3_basis)
        v_i = m_basis[i,3:]
        for j in range(m_basis.size(0)):
            skew_j = so3.hat(m_basis[j,:3], se3.so3_basis)
            v_j = m_basis[j,3:]
            FIM[i, j] = misc.inner(
                misc.matrix_vector_product(skew_i, a) + v_i,
                misc.matrix_vector_product(skew_j, a) + v_j
            )
    return FIM


def squared_error(g, estimate, basis=None):
    r_inv_error = estimate @ g.inverse()
    return se3.log_energy(r_inv_error, project_to_basis=basis)
    
def squared_errors(g, estimates, basis = None, show=False):
    sq_errors = []
    for i in range(len(estimates)):
        sq_errors.append(squared_error(g, estimates[i], basis))

    if show:
        plt.plot(sq_errors, linestyle='-.', color='pink')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()
        
    return sq_errors


def likelihoods(x_list, estimates):
    lls = []
    for i in range(len(estimates)):
        R = se3.get_R(estimates[i])
        t = se3.get_t(estimates[i])
        ll = 0.
        for j in range(len(landmarks)):
            pred = R.t() @ (landmarks[j] - t)
            for x in x_list[j]:
                ll += misc.norm_squared(x - pred)
        
        lls.append(-0.5*ll)

    return lls
    

def simulate(iterations, initial_estimate, x_avg, use_FIM=False, 
             step_size=1., step_size_decay=1.):
        if use_FIM:
            basis = m_basis
            inv_MT = inv_FIM  # inverse of the metric tensor

        else:
            basis = torch.eye(6)
            inv_MT = torch.eye(6)

        grad = torch.zeros(len(basis))
        g_hat = initial_estimate
        estimates = [g_hat]
        for _ in range(iterations):
            grad[:] = 0.
            for i in range(len(landmarks)):
                grad += get_gradient_sum(g_hat, landmarks[i], x_avg[i], basis)
            
            grad = misc.matrix_vector_product(inv_MT, grad)
            # Rewrite gradient in the standard (hat-vee) basis
            grad_vee = torch.einsum('i,ij->j', grad, basis)
            g_hat = se3.exp_map(step_size*grad_vee) @ g_hat
            step_size *= step_size_decay
            estimates.append(g_hat)
            
        return estimates

def add_circle(axis, radius=1.0):
    circle_points_above = []
    circle_points_below = []
    for theta in torch.linspace(0, 2 * torch.pi, 100):
        y = radius * torch.cos(theta)
        z = radius * torch.sin(theta)
        if z >= 0:
            circle_points_above.append((0, y, z))
        else:
            circle_points_below.append((0, y, z))
    
    circle_points_above = torch.tensor(circle_points_above).t()
    circle_points_below = torch.tensor(circle_points_below).t()
    
    axis.plot(circle_points_above[0], circle_points_above[1], 
              circle_points_above[2], linestyle=(0, (5.5, 6)), 
              linewidth=0.4, alpha=0.6, color='k', zorder=50)
    axis.plot(circle_points_below[0], circle_points_below[1], 
              circle_points_below[2], linestyle=(0, (5.5, 6)), 
              linewidth=0.4, alpha=0.6, color='k', zorder=-10)
    
def add_sphere(axis, center, radius=1.0):
    u = torch.linspace(0, 2 * torch.pi, 100)
    v = torch.linspace(0, torch.pi, 100)
    x = center[0] + radius * torch.outer(torch.cos(u), torch.sin(v))
    y = center[1] + radius * torch.outer(torch.sin(u), torch.sin(v))
    z = center[2] + radius * torch.outer(torch.ones_like(u), torch.cos(v))
    axis.plot_surface(x.numpy(), y.numpy(), z.numpy(), color='b', alpha=0.1)

    # Add great circles
    for angle in torch.linspace(0, torch.pi, 12):
        x_great = center[0] + radius * torch.cos(u) * torch.sin(angle)
        y_great = center[1] + radius * torch.sin(u) * torch.sin(angle)
        z_great = center[2] + radius * torch.cos(angle)
        axis.plot(x_great.numpy(), y_great.numpy(), z_great.numpy(), color='k', alpha=0.3, linewidth=0.5)

    for angle in torch.linspace(0, torch.pi, 12):
        x_great = center[0] + radius * torch.cos(angle) * torch.sin(v)
        y_great = center[1] + radius * torch.sin(angle) * torch.sin(v)
        z_great = center[2] + radius * torch.cos(v)
        axis.plot(x_great.numpy(), y_great.numpy(), z_great.numpy(), color='k', alpha=0.3, linewidth=0.5)

def correction_factor(error):
    error_vec = se3.log_map(error)
    skew = so3.hat(error_vec[:3], se3.so3_basis)
    vel_skew = so3.hat(error_vec[3:], se3.so3_basis)
    ad = torch.zeros(6, 6)
    ad[:3, :3] = skew
    ad[3:, 3:] = skew
    ad[3:, :3] = vel_skew
    return ad @ ad / 12.


if __name__ == '__main__':
    h_basis = get_h_basis(landmarks)
    more_vectors = torch.randn([6 - h_basis.size(0), 6])
    full_basis = torch.cat((h_basis, more_vectors), dim=0)
    full_basis = misc.gram_schmidt(full_basis)
    m_basis = full_basis[h_basis.size(0):]

    FIM = sum([get_FIM(a, m_basis) for a in landmarks])
    inv_FIM = torch.linalg.pinv(FIM)

    full_FIM = sum([get_FIM(a, full_basis) for a in landmarks])
    inv_full_FIM = torch.linalg.pinv(full_FIM)

    if SIMULATION_TYPE == 0:
        m = 10000
        true_pose = se3.exp_map(torch.tensor([0., 0., 0., 0., 0., 2.0]))
        x_avg = [get_measurements(true_pose, a, m).mean(dim=0) 
                  for a in landmarks]
        initial_estimates = [
            se3.exp_map(torch.tensor([-1., 0., 2.5, 0.5, 1.5, 1.4])),
            se3.exp_map(torch.tensor([1.5, 0.4, 0.6, 1., -0.6, 2.6])),
            se3.exp_map(torch.tensor([-0.5, 0.2, 0.5, -1.5, 0.5, 1.8])),  # top
            se3.exp_map(torch.tensor([1.3, 1.7, 2.0, 0.2, -1.5, -0.9]))  # bottom left
        ]

        color='#CC1100'
        
        axis = base_plot(true_pose, landmarks)
        if len(landmarks) == 1:
            add_sphere(axis, center=landmarks[0], radius=(landmarks[0]-se3.get_t(true_pose)).norm())
        
        elif len(landmarks) == 2:
            add_circle(axis, radius=2.0)
        for initial_estimate in initial_estimates:
            sim = simulate(15, initial_estimate.clone(), x_avg, use_FIM=True)
            visualize.plot_trail(sim, axis, color=color)
            visualize.plot_trajectory(sim, axis, 2, color=color)

        visualize.set_limits(axis, 2.0)
        axis.set_proj_type('persp')
        axis.set_box_aspect([1,1,1])

    if SIMULATION_TYPE == 1:
        use_CORRECTION = True
        show_CORRECTION = False

        use_SCALING = False
        true_pose = se3.exp_map(torch.tensor([0., 0., 0., 0., 0., 2.0]))
        final_errors = []
        final_errors_full = []
        correction_factors = []
        crbs = []
        crbs_corrected = []
        n_MC = 100000
        m_array = [2**i for i in range(0, 8)]

        if not LOAD_DATA:
            for m in m_array:
                final_error = torch.zeros(n_MC)
                final_error_full = torch.zeros(n_MC)
                delta = torch.zeros(6, 6, n_MC)
                for iter in tqdm(range(n_MC), desc=f"m = {m}"):
                    x_avg = [get_measurements(true_pose, a, m).mean(dim=0) 
                            for a in landmarks]
                    sim = simulate(5, 
                                   true_pose @ se3.standard_gaussian(1.0), x_avg, use_FIM=True)

                    final_error[iter] = squared_error(true_pose, sim[-1], m_basis)
                    final_error_full[iter] = squared_error(true_pose, sim[-1], full_basis)
                    if use_CORRECTION:
                        error = sim[-1] @ true_pose.inverse()
                        delta[:, :, iter] = correction_factor(error)  # ad^2 / 12
                        
                final_errors.append(final_error.mean())
                final_errors_full.append(final_error_full.mean())
                crbs.append(inv_FIM.trace()/m)

                if use_CORRECTION:
                    eye_delta = torch.eye(len(m_basis)) + \
                        m_basis @ delta.mean(dim=2) @ m_basis.t()
                    crbs_corrected.append(
                        (eye_delta @ inv_FIM @ eye_delta.t()).trace()/m)
                        
                if use_SCALING:
                    final_errors[-1] *= m
                    final_errors_full[-1] *= m
                    crbs[-1] *= m
                    if use_CORRECTION:
                        crbs_corrected[-1] *= m
            
            data = {
                'm_array': m_array,
                'final_errors': final_errors,
                'final_errors_full': final_errors_full,
                'crbs': crbs,
                'crbs_corrected': crbs_corrected if use_CORRECTION else None,
                'num_MC': n_MC,
            }
            
            with open('simulation_data.pkl', 'wb') as f:
                pickle.dump(data, f)
                
        elif LOAD_DATA:
            with open(LOAD_NAME, 'rb') as f:
                data = pickle.load(f)
                m_array = data['m_array']
                final_errors = data['final_errors']
                final_errors_full = data['final_errors_full']
                crbs = data['crbs']
                crbs_corrected = data['crbs_corrected']
                n_MC = data['num_MC']

                print(f"Loaded data for {len(m_array)} measurements, {n_MC} Monte Carlo runs.")

        fig = plt.figure(figsize=(4, 1.8))
        plt.plot(m_array, final_errors_full, '^-', markersize=3.5,
             color='#66BCCD', label='Variance on ' + r'$G$', linewidth=1.)
        plt.plot(m_array, final_errors, 'o-', markersize=3.5,
             color='#1e66a4', label='Variance on ' + r'$H\backslash G$', linewidth=1.)
        plt.plot(m_array, crbs_corrected, linestyle='dashed',
                 markersize=2.5, color='red', label='CRB', linewidth=1.1)
        
        if show_CORRECTION:            
            plt.plot(m_array, crbs, linestyle='dotted', markersize=2.5,
                     color='red', label='First-Order CRB', linewidth=1.1)
        
        plt.xticks(m_array)
        plt.xlabel("Number of Measurements " + r"$(m)$")
        plt.ylabel("Variance")
        if use_SCALING:
            plt.ylabel(r"$\text{Variance} \times m$")
            
        plt.tight_layout()
        plt.rcParams['patch.linewidth'] = 0.7
        
        plt.xlim(m_array[0], m_array[-1])
        
        plt.xscale('log')
        plt.yscale('log')
        # plt.xticks(m_array[0:1] + m_array[2:])
        # plt.ylim(0, 0.3)

        plt.tight_layout(rect=[0.01, 0, 1., 1])
        plt.legend(handletextpad=0.2, borderaxespad=0.15, labelspacing=0.15, fancybox=True, loc='lower left')

    if SIMULATION_TYPE == 2:
        fig = plt.figure(figsize=(4, 1.8))
        initial_estimate = se3.exp_map(torch.zeros(6))
        m_list = [2, 4, 8]
        seed = torch.randint(0, 500, (1,))
        seed = 230
        # seed = 39, 53, and 230 are interesting.
        torch.manual_seed(seed)
        print(f"Seed: {seed}")
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        for i in range(len(m_list)):
            m = m_list[i]
            color = colors[i]
            num_iter = 100
            true_pose = se3.exp_map(torch.tensor([0., 0., 0., 0., 0., 2.0]))
            x_list = [get_measurements(true_pose, a, m)
                      for a in landmarks]
            x_avg = [x.mean(dim=0) for x in x_list]
            sim = simulate(num_iter, initial_estimate.clone(), x_avg, 
                           use_FIM=True, step_size=1.0, step_size_decay=1.0)
            sim_GA = simulate(num_iter, initial_estimate.clone(), x_avg,   use_FIM=False, step_size=0.5, step_size_decay=0.9)
            
            lls = likelihoods(x_list, sim)
            lls_GA = likelihoods(x_list, sim_GA)
            plt.plot(range(num_iter+1), lls, linestyle='-', 
                     marker='o', markersize=2.5, linewidth=1., 
                     label=r'$' + f'm={m}' + r'$', color=color)
            plt.plot(range(num_iter+1), lls_GA, linestyle='dotted', linewidth=0.9, color=color)
            plt.ylabel(r"Log-Likelihood")

        # Add a second legend
        legend_elements = [
            Line2D([0], [0], color='gray', linestyle='-', label='Fisher Scoring', linewidth=1., marker='o', markersize=2.5),
            Line2D([0], [0], color='gray', linestyle='dotted', label='Gradient Ascent', linewidth=1.)
        ]
        first_legend = plt.legend(handletextpad=0.4, borderaxespad=0.15, labelspacing=0.2, loc='lower right', bbox_to_anchor=(1, 0.375),framealpha=0.95)
        plt.gca().add_artist(first_legend)
        
        second_legend = plt.legend(handles=legend_elements, loc='lower right', handletextpad=0.4, borderaxespad=0.15, labelspacing=0.2, framealpha=0.95)

        for leg in [first_legend, second_legend]:
            leg.get_frame().set_linewidth(0.5)
            leg.get_frame().set_edgecolor('darkgray')
        
        plt.xlabel("Iteration ($k$)")
        plt.tight_layout(rect=[0.01, 0, 1., 1])
        plt.xlim(0, 10)
        plt.ylim(-40, 0)
        plt.yticks([-40, -20, 0])

    if SIMULATION_TYPE == 3:
        Z = []
        for _ in range(2):
            vec = torch.randn(len(m_basis))
            z = torch.einsum('i,ij->j', vec, m_basis)
            Z.append(se3.hat(z))

        vec = torch.randn(len(h_basis))
        y = torch.einsum('i,ij->j', vec, h_basis)
        h = se3.exp_map(y)

        Adh_Z = [h @ Z[i] @ h.inverse() for i in range(2)]

        print(torch.trace(Z[0].t() @ Z[1]))
        print(torch.trace(Adh_Z[0].t() @ Adh_Z[1]))

    if SAVE_FIG:
        plt.savefig(f'images/{SIMULATION_TYPES[SIMULATION_TYPE]}.png',format='png', dpi=700)
        os.system(f'open images/{SIMULATION_TYPES[SIMULATION_TYPE]}.png')
    else:
        plt.show()
