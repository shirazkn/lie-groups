"""
TODO: 
- Change the weighting
- Generate samples in batch
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from functions import pickler, neural, sde, misc, so3
import constants

NUM_EPOCHS = 1000
LOAD_MODEL = True

DEBUGGING = False

device =  neural.get_device()
dtype = constants.datatype

so3_basis = so3.get_basis(3, dtype=dtype, device=device)


def immersion(g):
    return g.reshape(g.size()[0], -1)[:, :6]


def concatenate_input(g, t):
    return torch.concatenate([immersion(g), t], dim=1)


def differentials_from_matrices(g):
    pushforward_basis = []
    for i in range(3):
        tangent_vector = torch.matmul(g, so3_basis[i])
        # Being a linear map, immersion(.) is also its differential
        pushforward_vector = immersion(tangent_vector)
        # Transpose all vectors
        pushforward_basis.append(torch.unsqueeze(pushforward_vector, 2))

    differentials = torch.cat(pushforward_basis, dim=2)
    return differentials


def ism_loss(outputs, inputs, differentials):
    norm_term = 0.5*misc.inner(outputs, outputs)

    divergence_term = torch.zeros_like(norm_term)
    for i in range(3):
        component = outputs[:,i].sum()
        # Taking the gradient will 'split' this sum up due to each term depending on a different input
        gradients = torch.autograd.grad(component, inputs, create_graph=True)[0][:,:-1]
        # gradients_i = [0.01, 0.04, 0.8, ...]
        divergence_term = divergence_term + misc.inner(gradients, 
                                                       differentials[:,:,i].squeeze())

    times = inputs[:, -1]
    return (norm_term*times).mean(), (divergence_term*times).mean()


def ism_loss_sliced(outputs, inputs, differentials):
    norm_term = 0.5*misc.inner(outputs, outputs)

    vectors = torch.randn_like(outputs, dtype=dtype, device=device)
    vectors = misc.normalize(vectors)

    weighted_sum = misc.inner_sum(vectors, outputs)
    vjp = torch.autograd.grad(weighted_sum, inputs, create_graph=True)[0][:,:6]
    
    pushforwards = torch.einsum('ijk,ik->ij', differentials, vectors)
    divergence_term = misc.inner(vjp, pushforwards)

    times = inputs[:, -1]
    return (norm_term*times).mean(), (divergence_term*times).mean()


if __name__ == "__main__":
    scoreNetwork = neural.Feedforward(*constants.feedforward_signature, 
                                      device=device)

    if LOAD_MODEL and pickler.file_exists(constants.model_filename):
            scoreNetwork.load_state_dict(torch.load(constants.model_filename))
            print(f"Loaded {constants.model_filename}.")
    else:
        print("Training a new model from scratch.")
        pickler.ask_delete(constants.model_filename)
        scoreNetwork.apply(neural.scaled_init)

    writer = SummaryWriter(
        f"{constants.summary_directory}/{misc.get_datetime_string()}")
    
    writer.add_graph(scoreNetwork, torch.randn(1, 7, device=device, dtype=dtype))

    dataset = torch.tensor(
        np.array(pickler.read_all(constants.samples_filename)), 
        dtype=dtype, device=device, requires_grad=False)

    scoreNetwork.train()
    optimizer = torch.optim.Adam(scoreNetwork.parameters(), lr=constants.params["lr"], weight_decay = constants.params["decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                       gamma=(1-constants.params["lr_scheduler"]))

    if constants.params["loss_type"] == "ISM":
        loss_fn = ism_loss

    elif constants.params["loss_type"] == "ISM_Sliced":
        loss_fn = ism_loss_sliced

    print(f"Loss function: {constants.params['loss_type']}")
    diffuser = sde.SDE(group=so3, device=device)
    
    g = torch.zeros_like(dataset, requires_grad=False)
    times = torch.zeros([dataset.size()[0], 1], dtype=dtype, device=device,
                         requires_grad=False)
    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Training"):
        times = torch.rand_like(times)
        for i in range(dataset.size()[0]):
            g[i, :] = diffuser.flow(dataset[i].unsqueeze(0), times[i])

        inputs = concatenate_input(g, times)
        differentials = differentials_from_matrices(g)
        
        inputs.requires_grad = True
        outputs = scoreNetwork(inputs)
        loss_term_1, loss_term_2 = loss_fn(outputs, inputs, differentials)
        loss = loss_term_1 + loss_term_2
        inputs.requires_grad = False
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('Loss', loss.detach(), epoch)
        writer.add_scalar('Norm Term', loss_term_1.detach(), epoch)
        writer.add_scalar('Div. Term', loss_term_2.detach(), epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], epoch)
        # if epoch % 100 == 0:
        #     neural.weight_histograms(writer, epoch, scoreNetwork)

    scoreNetwork.eval()
    torch.save(scoreNetwork.state_dict(), constants.model_filename)
    writer.close()
