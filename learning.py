import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from functions import pickler, neural, so, sde, misc
import constants

NUM_EPOCHS = 1000
LOAD_MODEL = True

DEBUGGING = False
LOSS_TYPES = ["ISM", "ISM_Sliced"]
LOSS_TYPE = "ISM"

params = {
    "lr": 1e-5,
    "decay": 1e-2,
    "lr_scheduler": 1e-4
}

device =  neural.get_device()
dtype = constants.datatype
bases = torch.tensor(np.array(so.get_bases(3)), 
                     dtype=dtype, device=device)


def input_from_tuple(g, t):
    return np.concatenate([g.flatten()[:6], np.array([t])])


def differentials_from_matrices(matrices):
    pushforward_bases = []
    for i in range(3):
        tangent_vector = torch.matmul(matrices, bases[i])
        pushforward_vector = tangent_vector.view(matrices.size()[0], 9)
        # Transpose all vectors
        pushforward_bases.append(torch.unsqueeze(pushforward_vector, 2))  

    # Concatenate the columns
    differentials = torch.cat(pushforward_bases, dim=2)
    differentials = differentials[:,:6,:]

    if DEBUGGING:
        # Set DEBUGGING = True to verify what this function does
        # Use `quit()` to exit the debugger
        _ind = np.random.randint(0, len(matrices))
        print(f"Matrix: {matrices[_ind]}\n")
        for i in range(3):
            print(f"Matrix @ E_{i}:\n {matrices[_ind] @ bases[i]}\n")
        print(f"Differential/Jacobian Matrix: {differentials[_ind]}") 
        import pdb; pdb.set_trace()

    return differentials


def pre_processor(training_samples, device):
    inputs = []
    matrices = []
    for g, t in training_samples:
        inputs.append(input_from_tuple(g, t))
        matrices.append(g)
    
    inputs = torch.tensor(np.array(inputs), dtype=dtype, device=device)
    matrices = torch.tensor(np.array(matrices), dtype=dtype, device=device)
    print(f"Prepared {len(inputs)} samples.")

    return inputs, differentials_from_matrices(matrices)


def ism_loss(outputs, inputs, differentials):
    times = inputs[:,-1]
    norm_term = 0.5*misc.inner(outputs, outputs)/times

    divergence_term = torch.zeros_like(norm_term)
    for i in range(3):
        component = outputs[:,i].sum()
        #  [[0.3, 0.5, 0.1], [0.32, 0.5, 0.11], [0.2, ... ], ...]
        # component_i = 0.3 + 0.32 + 0.2 + ... = 100.00
        gradients = torch.autograd.grad(component, inputs, create_graph=True)[0][:,:-1]
        # gradients_i = [0.01, 0.04, 0.008, ...]
        divergence_term = divergence_term + misc.inner(gradients, 
                                                       differentials[:,:,i].squeeze())

    divergence_term = divergence_term/times
    return norm_term.mean(), divergence_term.mean()


def ism_loss_sliced(outputs, inputs, differentials):
    times = inputs[:,-1]
    norm_term = 0.5*misc.inner(outputs, outputs)/times

    vectors = torch.randn_like(outputs, dtype=dtype, device=device)
    vectors = misc.normalize(vectors)

    weighted_sum = misc.inner_sum(vectors, outputs)
    vjp = torch.autograd.grad(weighted_sum, inputs, create_graph=True, retain_graph=True)[0][:,:6]
    
    pushforwards = torch.einsum('ijk,ik->ij', differentials, vectors)
    divergence_term = misc.inner(vjp, pushforwards)/times
    return norm_term.mean(), divergence_term.mean()


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

    training_samples = pickler.read_all(constants.diffused_samples_filename)
    inputs, differentials = pre_processor(training_samples, device)

    scoreNetwork.train()
    optimizer = torch.optim.Adam(scoreNetwork.parameters(), lr=params["lr"], weight_decay = params["decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                       gamma=(1-params["lr_scheduler"]))

    if LOSS_TYPE == "ISM":
        loss_fn = ism_loss

    elif LOSS_TYPE == "ISM_Sliced":
        loss_fn = ism_loss_sliced
    print(f"Loss function: {LOSS_TYPE}")
    losses = {"total": [], "norm": [], "divergence": []}
    learning_rates = []
    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Training"):
        inputs.requires_grad = True
        ouputs = scoreNetwork(inputs)
        loss_term_1, loss_term_2 = loss_fn(ouputs, inputs, differentials)
        loss = loss_term_1 + loss_term_2
        inputs.requires_grad = False

        writer.add_scalar('Loss', loss.detach(), epoch)
        writer.add_scalar('Norm Term', loss_term_1.detach(), epoch)
        writer.add_scalar('Div. Term', loss_term_2.detach(), epoch)
        losses["total"].append(loss.detach().cpu().numpy())
        losses["norm"].append(loss_term_1.detach().cpu().numpy())
        losses["divergence"].append(loss_term_2.detach().cpu().numpy())
        learning_rates.append(scheduler.get_last_lr()[-1])
        
        # if losses["divergence"][-1] < -3.0:
        #     import pdb; pdb.set_trace()

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(scoreNetwork.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], epoch)
        if epoch % 100 == 0:
            neural.weight_histograms(writer, epoch, scoreNetwork)

    scoreNetwork.eval()
    torch.save(scoreNetwork.state_dict(), constants.model_filename)
    writer.close()

    from matplotlib import pyplot as plt

    plt.figure("Losses")
    plt.plot(losses["total"], label="Total Loss")
    # plt.plot(losses["norm"], label="Norm Term")
    # plt.plot(losses["divergence"], label="Divergence Term")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xlim([0, NUM_EPOCHS])
    
    plt.figure("Learning Rates")
    plt.plot(learning_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xlim([0, NUM_EPOCHS])
    plt.show()

 