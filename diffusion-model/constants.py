from torch import float32, float64

directory = "data"
summary_directory = f"{directory}/runs"

samples_filename = f"{directory}/samples.pkl"
model_filename = f"{directory}/scoreNetwork.pkl"

n_samples = 10000
feedforward_signature = (7, 512, 3, 2)  # Last number is the depth
datatype = float32  # Note: 
                    # MPS (Apple's equivalent of CUDA) doesn't support float64

params = {
    "lr": 1e-6,
    "decay": 0.0,
    "lr_scheduler": 1e-3,
    "loss_type": "ISM",
    "sde_dt": 0.01
}