import torch


def get_device():
    # if torch.backends.mps.is_available():
    #     torch.set_default_dtype(torch.float32)
    #     print("Torch device was changed to float32.")
    #     return torch.device("mps")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    else:
        print("Could not detect a GPU.")
        return "cpu"
    

torch.set_default_dtype(torch.float64)

device = get_device()
print(f"Using {device} as the torch device.")
torch.set_default_device(device)