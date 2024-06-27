import config
import torch


def get_device():
    if torch.backends.mps.is_available():
        if not config.datatype == "float32":
            print("MPS only supports float32. Using CPU instead.")
            return torch.device("cpu")
        torch.set_default_dtype(torch.float32)
        return torch.device("cpu")
    
    elif torch.cuda.is_available():
        return torch.device("cuda")
    
    else:
        print("Could not detect a GPU.")
        return "cpu"
    

if config.datatype == "float32":
    torch.set_default_dtype(torch.float32)
    print("Torch datatype was set to float32.")

else:
    torch.set_default_dtype(torch.float64)
    print("Torch datatype was set to float64.")

device = get_device()
print(f"Using {device} as the torch device.")
torch.set_default_device(device)