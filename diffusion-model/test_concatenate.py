import torch

g = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
         [[-1, -2, -3],
          [-4, -5, -6],
          [-7, -8, -9]]
])

t = torch.tensor([0.1, 0.2])

from learning import concatenate_input
print(concatenate_input(g, t))
