import torch
from torch import nn
from constants import datatype


class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 1, device = "cpu"):
        super(Feedforward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ActivationLayer = nn.ReLU
        
        layers_list = [nn.Linear(input_dim, hidden_dim, 
                                 device=device, dtype=datatype),
                                 ActivationLayer()]
        for _ in range(depth):
                    layers_list.append(nn.Linear(hidden_dim, hidden_dim,
                                                 device=device, dtype=datatype))
                    layers_list.append(ActivationLayer())

        layers_list.append(nn.Linear(hidden_dim, output_dim, 
                                     device=device, dtype=datatype))
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)
    

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps") if datatype==torch.float32 else "cpu"
    
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Not tested

    else:
        return "cpu"


def weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights, 
                         global_step=step,bins='tensorflow')


def weight_histograms(writer, step, model):
    for layer_number in range(len(model.layers)):
        layer = model.layers[layer_number]
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, layer_number)
        else:
            pass


def scaled_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
