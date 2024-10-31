import numpy as np
import matplotlib.pyplot as plt

def pd_matrix(n, eps=1e-4):
    A = np.random.randn(n, n)
    P = np.dot(A.T, A)
    P += eps * np.eye(n)
    
    return P


SigA = pd_matrix(2)
SigB = pd_matrix(2)

def get_units(length):
    thetas = np.linspace(0, 2*np.pi, length)
    return np.array([np.cos(thetas), np.sin(thetas)]).T


def draw_dM(Sig, label=None, linestyle='b-'):
    units = get_units(500)
    points = [[], []]
    for i in range(units.shape[0]):
        unit = units[i, :].reshape(2, 1)
        x = Sig @ unit / (np.sqrt(unit.T @ Sig @ unit))
        points[0].append(x[0][0])
        points[1].append(x[1][0])

    plt.plot(points[0], points[1], linestyle, label=label)

def draw_sum(SigA, SigB):
    units = get_units(500)
    points = [[], []]
    for i in range(units.shape[0]):
        unit = units[i, :].reshape(2, 1)
        x = SigA @ unit / (np.sqrt(unit.T @ SigA @ unit))
        y = SigB @ unit / (np.sqrt(unit.T @ SigB @ unit))
        z = x + y
        points[0].append(z[0][0])
        points[1].append(z[1][0])

    plt.plot(points[0], points[1], 'r--', label='Minkowski Sum')


draw_dM(SigA, linestyle='k-')
draw_dM(SigB, linestyle='k-')
draw_dM(SigA + SigB, linestyle='b--', label = 'Convolution')
draw_sum(SigA, SigB)
plt.legend()
plt.show()