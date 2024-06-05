from torch import float32

directory = "data"
summary_directory = f"{directory}/runs"

samples_filename = f"{directory}/samples.pkl"
diffused_samples_filename = f"{directory}/diffused_samples.pkl"
model_filename = f"{directory}/scoreNetwork.pkl"

n_samples = 200
n_training_dataset = 20000
feedforward_signature = (10, 128, 3, 2)  # Last number is the depth
datatype = float32  # MPS (Apple Silicon equivalent of CUDA) doesn't support float63

simulation = {
    "final_time": 4.0,
    "sde_dt": 0.01
}