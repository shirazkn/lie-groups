import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, numpy

from functions import neural, sde, pickler, so3
import constants

n_samples = 200
SHOW_SCATTERPLOT = True


if __name__ == "__main__":
    device = neural.get_device()
    scoreNetwork = neural.Feedforward(*constants.feedforward_signature).to(device)
    scoreNetwork.load_state_dict(torch.load(constants.model_filename))
    scoreNetwork.eval()
 
    initial_values = torch.tensor(numpy.array(so3.uniformSO3(size = n_samples)),
                                  dtype=constants.datatype, device=device)
    reverseSDE = sde.ScoreSDE(group=so3, device=device,
                              score_vector_field=scoreNetwork)

    final_time = torch.tensor(1., dtype=constants.datatype, device=device)
    final_values = reverseSDE.flow(initial_values, final_time).detach().cpu().numpy()
    
    plt.figure("Target pdf vs. Sampled pdf")
    dataset = pickler.read_all(constants.samples_filename)
    so3.sliceVisualization(dataset, scatter=SHOW_SCATTERPLOT, label="Dataset", show=False)
    so3.sliceVisualization(final_values, scatter=SHOW_SCATTERPLOT, show=False,
                          label="Reversed Samples", color="red", alpha=0.5)
    plt.legend()
    plt.show()
    # so.sphereVisualization(final_values, resolution=60)