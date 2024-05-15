import numpy as np
from functions.so import *

if __name__ == "__main__":
    mean = np.eye(3)
    # samples = wrappedGaussian(mean, cov = 0.01*np.eye(3), size = 1000)
    # samples = uniformSO3(size = 1000)
    samples = testSO3(size = 1000)
    sphereVisualization(samples, resolution=60)
    print("Done")