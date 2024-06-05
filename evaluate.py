import numpy as np
from tqdm import tqdm
import torch

from functions import neural, sde, so, pickler
import constants


if __name__ == "__main__":
    scoreNetwork = neural.Feedforward(*constants.feedforward_signature)
    scoreNetwork.load_state_dict(torch.load(constants.model_filename))
    scoreNetwork.eval()
 
    initial_values = so.uniformSO3(size = 300)

    reverseSDE = sde.ScoreSDE(bases=so.get_bases(3), 
                              dt=constants.simulation["sde_dt"], score_vector_field=scoreNetwork)

    final_values = []
    for initial_value in tqdm(initial_values):
        final_value = reverseSDE.flow_T(initial_value)
        final_values.append(final_value)
    
    # so.sliceVisualization(so.testSO3(size = constants.n_samples))
    so.sliceVisualization(final_values)
    # so.sphereVisualization(final_values, resolution=60)