"""
Generates the training dataset
"""
from functions import sde, so3
from functions import pickler

import constants
import numpy as np
from tqdm import tqdm

dtype = constants.datatype

if __name__ == "__main__":
    pickler.ask_delete(constants.samples_filename, description=f"Existing samples file will be deleted.")
    #  Make a new directory `data/` if the above throws an error

    print(f"Generating {constants.n_samples} samples... ")
    samples = so3.testSO3(size = constants.n_samples)
    pickler.add_item(constants.samples_filename, samples)
    print("Done!\n")
