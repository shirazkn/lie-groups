"""
Tries to fit a step function through datapoints. The goal is to verify that I'm computing the autograd properly. It's computed twice: for backpropagation as well as for regularizing the derivative of the function.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from tqdm import tqdm


class Function(nn.Module):
    def __init__(self):
        super(Function, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("cpu")

    else:
        device = "cpu"
        print("Could not find MPS device. Using CPU.")


    model = Function().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0)
    mse = nn.MSELoss()
    
    n_points = 10
    weights = np.linspace(10, 0.001, 200)
    
    losses = []
    for epoch in tqdm(range(200)):
        optimizer.zero_grad()
        for _ in range(n_points):
            x = torch.rand(1).to(device).requires_grad_(True)
            output = model.forward(x)
            # output.backward(retain_graph=True)  # This is sending gradients to parameters
            loss = (output - torch.autograd.grad(output, x, create_graph=True)[0])**2 + weights[epoch]*(model.forward(torch.Tensor([0.0]).to(device)) - 1)**2
            loss.backward()
        
        optimizer.step()
        losses.append(loss.to("cpu").detach().numpy())
        
    print("Done training.")
    plt.plot(np.array(losses).flatten(), label="Loss")
    plt.xlabel("Epoch")
    plt.show()
    model.eval()

    plot_x = np.linspace(-1, 2, 100, dtype=np.float32)
    plt.plot(plot_x, np.exp(plot_x), label="True")
    plt.plot(plot_x, 
             model(torch.tensor(plot_x).view(-1,1).to(device)).to("cpu").detach().numpy().flatten(), 
             label="Fitted")
    plt.show()
