import numpy as np
from tqdm import tqdm
import torch

from functions import pickler, neural
import constants
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 100

class MatrixDataset(Dataset):
    def __init__(self, samples):
        self.inputs = []
        for t, g in samples:
            self.inputs.append(np.concatenate(
                [g[:,:2].flatten(), np.array([t])]))
        
        self.n_samples = len(self.inputs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx]


if __name__ == "__main__":


    device =  neural.get_device()
    scoreNetwork = neural.Feedforward(6, 256, 3, depth=3, device=device)
    optimizer = torch.optim.Adam(scoreNetwork.parameters(), lr=0.001)

    samples = MatrixDataset(
        pickler.read_all(constants.diffused_samples_filename))
    dataloader = DataLoader(samples, batch_size=20, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)
    data = dataiter.next()
    t, g = data

    raise NotImplementedError("This is only necessary for larger datasets...")