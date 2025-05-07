import numpy as np

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

# def get_gradient(scatter, estimate):
#         scinv = scatter @ sp.linalg.inv(estimate)
#         return 0.25*(scinv + scinv.T - 2*np.eye(3))


num = 100000
samples = gaussian.sample(num)
print("True Covariance:", cov)

scatter = np.zeros((3, 3)) 
for sample in samples:
    x = np.linalg.inv(A) @ sample
    scatter += x @ x.T

print("\nSample Covariance:", scatter/num)
