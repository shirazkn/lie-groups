from tqdm import tqdm
import numpy as np
from functions import so, sde

from torch import nn, optim
from score_network import ScoreNetwork

NUM_EPOCHS = 100


def ism_loss(score, t):
    return score.norm()**2

if __name__ == "__main__":
    mean = np.eye(3)
    samples = so.testSO3(size = 1000)
    # so.sliceVisualization(samples)
    # so.sphereVisualization(samples, resolution=80)

    sde_solver = sde.SDE(bases=so.get_bases(3), dt=0.05)
    score_network = ScoreNetwork(6, 64, 3)
    score_network.train()

    mse = ism_loss
    optimizer = optim.Adam(score_network.parameters(), lr=0.01)
    
    for epoch in NUM_EPOCHS:
        for sample in tqdm(samples):
            t = np.random.uniform(0, 3.0)
            smeared_sample = sde_solver.flow(sample, t)
            score = score_network(smeared_sample[:2,:2].flatten())

    so.sphereVisualization(smeared_samples, resolution=40)
    print("Done")