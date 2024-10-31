import numpy as np
import scipy as sp
from tqdm import tqdm

class MultivariateGaussian3D:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, num_samples=1):
        samples = np.random.multivariate_normal(self.mean, self.cov, num_samples)
        return samples.reshape(-1, 3, 1)


mean = [0, 0, 0]
A = np.array([[3.1, 1., 2.], [1., 1., 4.], [0, 2., 1.]])
cov = A @ A.T
gaussian = MultivariateGaussian3D(mean, cov)

def get_gradient(scatter, estimate):
        scinv = scatter @ sp.linalg.inv(estimate)
        return 0.25*(scinv + scinv.T - 2*np.eye(3))


MC = True
if not MC:
    num = 100
    samples = gaussian.sample(num)
    print("True Covariance:", cov)

    scatter = np.zeros((3, 3)) 
    for sample in samples:
        scatter += sample @ sample.T

    scatter /= num

    estimate = np.eye(3)
    for _ in range(1000):
        estimate = estimate @ sp.linalg.expm(0.01*get_gradient(scatter, estimate))

    print("\n\n\nMy Estimator:", estimate, "\n\nSample Estimator:", scatter)

else:
    my_errors = []
    sample_errors = []
    for sim in tqdm(range(10)):
        num = 1000
        samples = gaussian.sample(num)

        scatter = np.zeros((3, 3)) 
        for sample in samples:
            scatter += sample @ sample.T

        scatter /= num

        estimate = np.eye(3)
        for _ in range(1000):
            estimate = estimate @ sp.linalg.expm(0.01*get_gradient(scatter, estimate))

        scatter = np.cov(samples.reshape(-1, 3).T)
        my_errors.append(np.trace(estimate.T @ cov))
        sample_errors.append(np.trace(scatter.T @ cov))
        

import matplotlib.pyplot as plt
plt.plot(my_errors, label="My Estimator")
plt.plot(sample_errors, label="Sample Estimator")
plt.legend()
plt.show()