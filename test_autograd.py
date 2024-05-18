"""
Tests whether the autodifferentiation works as expected
"""

import torch
import numpy as np


rand_number = np.random.uniform(0, 2*np.pi)
input = torch.tensor(rand_number, requires_grad=True)

output = torch.sin(input)
output.backward()
print(f"The autograd computes {input.grad}, and the analytical value is {np.cos(rand_number)}.")