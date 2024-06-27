"""
Tries to fit an exponential function through datapoints. The goal is to verify that I'm computing the autograd properly. It's computed twice: for backpropagation as well as for regularizing the derivative of the function.
"""

from functions import neural
import constants

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(constants)


class Function(nn.Module):
    def __init__(self):
        super(Function, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":
    device = neural.get_device()
    model = Function().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    
    n_points = 100    
    #  x ->  MODEL  -> output -> loss 
    #        ^w,b^

    losses = []
    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        epoch_loss = 0.0
        for _ in range(n_points):
            x = torch.rand(1).to(device).requires_grad_(True)  # 0 to 1
            output = model.forward(x)

            L2_term = torch.tensor(0., requires_grad=True)
            for name, weights in model.named_parameters():
                if 'bias' not in name:
                    weights_sq_sum = torch.sum(weights**2)
                    L2_term = L2_term + weights_sq_sum

            # Note: 'output.backward(retain_graph=True)' sends gradients to parameters!
            loss = (output - torch.autograd.grad(output, x, create_graph=True)[0])**2 + (model(torch.Tensor([0.]).to(device)) - 1)**2
            + 0.001*L2_term
            loss.backward()
        
        writer.add_scalar("Loss", loss, epoch)
        optimizer.step()
        losses.append(loss.to("cpu").detach().numpy())
        scheduler.step()
        
    print("Done training.")
    plt.plot(np.array(losses).flatten(), label="Loss")
    plt.xlabel("Epoch")
    plt.show()
    model.eval()

    plot_x = np.linspace(-2, 3, 100, dtype=np.float32)
    plt.plot(plot_x, np.exp(plot_x), label="True")
    plt.plot(plot_x, 
             model(torch.tensor(plot_x).view(-1,1).to(device)).to("cpu").detach().numpy().flatten(), 
             label="Fitted")
    plt.show()
