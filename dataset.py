"""
Generates the training dataset
"""
from functions import so, sde
from functions import pickler

import constants
import numpy as np
from tqdm import tqdm

dtype = constants.datatype

if __name__ == "__main__":
    
    try:
        samples = pickler.read_all(constants.samples_filename)
        assert len(samples) == constants.n_samples
        print(f"Loaded {constants.n_samples} samples.")

    except:
        pickler.ask_delete(constants.samples_filename, description=f"Could not load {constants.n_samples} samples from the existing files, so they will be deleted.")

        print(f"Generating {constants.n_samples} samples... ")
        samples = so.testSO3(size = constants.n_samples)
        pickler.add_item(constants.samples_filename, samples)
        print("Done!\n")

    # -----------------------------------------------------------------------
    # Generating diffused samples -------------------------------------------

    pickler.ask_delete(constants.diffused_samples_filename)
    
    num_repetitions = int(constants.n_training_dataset / constants.n_samples)
    print(f"Generating {constants.n_samples} X {num_repetitions} = {constants.n_training_dataset} diffused samples... ")

    T = constants.simulation["final_time"]
    sde_solver = sde.SDE(bases=so.get_bases(3), dt=constants.simulation["sde_dt"])
    diffused_samples = []
    for i in tqdm(range(constants.n_samples), desc=f"Diffusing samples"):
        for _ in range(num_repetitions):
            t = np.random.uniform(0.01, T)
            diffused_samples.append((sde_solver.flow(samples[i], t), t))
            
    pickler.add_item(constants.diffused_samples_filename, diffused_samples)
    print("Done!")
