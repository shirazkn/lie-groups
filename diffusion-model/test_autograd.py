"""
Tests whether the autodifferentiation works as expected
"""

import torch
import numpy as np

TEST_NO = 1

if TEST_NO == 0:
    rand_number = np.random.uniform(0, 2*np.pi)
    input = torch.tensor(rand_number, requires_grad=True)

    output = torch.sin(input)
    output.backward()
    print(f"The autograd computes {input.grad}, and the analytical value is {np.cos(rand_number)}.")

elif TEST_NO == 1:
    a = torch.rand(3, 1, requires_grad=True)
    b = (a*a).sum()
    print("Differentiating w.r.t. a:")
    torch.autograd.grad(b, a)

    a = torch.rand(3, 1, requires_grad=True)
    b = (a*a).sum()
    print("Differentiating w.r.t. a[0]:")
    torch.autograd.grad(b, a[0])


