from pf_pinn.model import PFPINN
from pf_pinn.samplings import make_flattend_grid_data, \
    make_lhs_sampling_data, \
    make_semi_circle_data, \
    make_uniform_grid_data, \
    make_uniform_grid_data_transition
from pf_pinn.visualization import Visualizer
from pf_pinn.loss_manager import LossManager
from pf_pinn.causal_weighting import CausalWeighter
    
# import matplotlib
# matplotlib.rcParams["font.size"] = 18
