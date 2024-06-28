from torch import float32, float64

directory = "data"
summary_directory = f"{directory}/runs"

samples_filename = f"{directory}/samples.pkl"
diffused_samples_filename = f"{directory}/diffused_samples.pkl"
model_filename = f"{directory}/scoreNetwork.pkl"

n_samples = 100000
n_training_dataset = 100000
feedforward_signature = (7, 128, 3, 2)  # Last number is the depth
datatype = float32  # Note: MPS (Apple's equivalent of CUDA) doesn't support float64

simulation = {
    "final_time": 5.0,
    "sde_dt": 0.001
}