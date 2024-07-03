from pf_pinn.model import PFPINN
from pf_pinn.samplings import make_flattend_grid_data, \
    make_lhs_sampling_data, \
    make_semi_circle_data, \
    make_uniform_grid_data, \
    make_uniform_grid_data_transition
    
import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "Palatino Linotype"
matplotlib.rcParams["font.size"] = 16
