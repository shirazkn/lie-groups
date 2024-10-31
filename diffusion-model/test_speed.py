from timeit import timeit

import torch
from torch.nn import MSELoss
import argparse

from functions.neural import Feedforward

DEVICES = ["mps", "cpu"]

repetitions = 1000

input_dimension = 256
hidden_dimension = 512
output_dimension = 128

n_samples = 1000

if __name__ == "__main__":    
    def instantiation(_device):
        _ = torch.rand(input_dimension, device=_device).requires_grad_(True)
        _ = Feedforward(input_dimension, hidden_dimension, output_dimension, device=_device)

    print(f"Time taken to instantiate variables {repetitions} times\n") 
    for device in DEVICES:
        time = timeit(lambda: instantiation(device), number=repetitions)
        print(f" {device}: {time} seconds\n")

    # ------------------------------------------------

    def evaluation(x, F):
        _ = F(x)

    print(f"\n\nTime taken to evaluate {repetitions} times\n") 
    for device in DEVICES:
        x = torch.rand(n_samples, input_dimension, device=device).requires_grad_(True)
        F = Feedforward(input_dimension, hidden_dimension, output_dimension, device=device)
        time = timeit(lambda: evaluation(x, F), number=repetitions)
        print(f" {device}: {time} seconds\n")

    # ------------------------------------------------

    def backprop_autograd(loss):
        loss.backward(retain_graph=True)

    mse = MSELoss()
    print(f"\n\nTime taken to do 'backward()' {repetitions} times\n") 
    for device in DEVICES:
        x = torch.rand(n_samples, input_dimension, device=device).requires_grad_(True)
        F = Feedforward(input_dimension, hidden_dimension, output_dimension, device=device)
        loss = mse(F(x), torch.rand(n_samples, output_dimension).to(device))
        time = timeit(lambda: backprop_autograd(loss), number=repetitions)
        print(f" {device}: {time} seconds\n")

    # ------------------------------------------------