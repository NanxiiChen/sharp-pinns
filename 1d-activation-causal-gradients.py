from pyDOE import lhs
import matplotlib.pyplot as plt
import configparser
import pandas as pd
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import pf_pinn as pfp
import numpy as np
import torch
import datetime
import matplotlib
import time
matplotlib.use("Agg")

config = configparser.ConfigParser()
config.read("config.ini")

LOG_NAME = config.get("TRAIN", "LOG_NAME").strip('"')
now = LOG_NAME
if LOG_NAME == "None":
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # now = "fourier-feather"
writer = SummaryWriter(log_dir="/root/tf-logs/" + now)


# Define the sampler
class GeoTimeSampler:
    def __init__(
        self,
        geo_span: list,  # 1d
        time_span: list,
    ):
        self.geo_span = geo_span
        self.time_span = time_span

    def resample(self, in_num, bc_num, ic_num, strateges=["lhs", "lhs", "lhs"]):
        return self.in_sample(in_num, strateges[0]), \
            self.bc_sample(bc_num, strateges[1]), \
            self.ic_sample(ic_num, strateges[2])

    def in_sample(self, in_num, strategy: str = "lhs",):

        if strategy == "lhs":
            func = pfp.make_lhs_sampling_data
        elif strategy == "grid":
            func = pfp.make_uniform_grid_data
        elif strategy == "grid_transition":
            func = pfp.make_uniform_grid_data_transition
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        geotime = func(mins=[self.geo_span[0], self.time_span[0]],
                       maxs=[self.geo_span[1], self.time_span[1]],
                       num=in_num)

        return geotime.float().requires_grad_(True)

    def bc_sample(self, bc_num, strategy: str = "lhs",):
        if strategy == "lhs":
            ts = (lhs(1, bc_num) *
                  (self.time_span[1] - self.time_span[0]) + self.time_span[0]).reshape(-1, 1)
        elif strategy == "grid":
            # ts = np.linspace(self.time_span[0], self.time_span[1], bc_num)[
            #     1:-1].reshape(-1, 1)
            ts = torch.linspace(self.time_span[0], self.time_span[1], bc_num, device="cuda")[
                1:-1].reshape(-1, 1)
        elif strategy == "grid_transition":
            # ts = np.linspace(self.time_span[0], self.time_span[1], bc_num)[
            #     1:-1].reshape(-1, 1)
            ts = torch.linspace(self.time_span[0], self.time_span[1], bc_num, device="cuda")[
                1:-1].reshape(-1, 1)
            distance = (self.time_span[1] - self.time_span[0]) / (bc_num - 1)
            # shift = np.random.uniform(-distance, distance, 1)
            # ts = np.clip(ts + shift, self.time_span[0], self.time_span[1])
            shift = torch.rand(1, device="cuda") * 2 * distance - distance
            ts = torch.clamp(ts + shift, self.time_span[0], self.time_span[1])
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        # xt_l = np.hstack([np.full(ts.shape[0], self.geo_span[0]).reshape(-1, 1),
        #                   ts.reshape(-1, 1)])
        # xt_r = np.hstack([np.full(ts.shape[0], self.geo_span[1]).reshape(-1, 1),
        #                   ts.reshape(-1, 1)])
        # xts = np.vstack([xt_l, xt_r])
        xt_l = torch.cat([torch.full((ts.shape[0], 1), self.geo_span[0], device="cuda"),
                          ts], dim=1)
        xt_r = torch.cat([torch.full((ts.shape[0], 1), self.geo_span[1], device="cuda"),
                          ts], dim=1)
        xts = torch.cat([xt_l, xt_r], dim=0)
        return xts.float().requires_grad_(True)

    def ic_sample(self, ic_num, strategy: str = "lhs", local_area=[-0.1, 0.1]):
        if strategy == "lhs":
            xs = (lhs(1, ic_num) *
                  (self.geo_span[1] - self.geo_span[0]) + self.geo_span[0]).reshape(-1, 1)
            xs_local = (lhs(1, ic_num) *
                        (local_area[1] - local_area[0]) + local_area[0]).reshape(-1, 1)
        elif strategy == "grid":
            # xs = np.linspace(self.geo_span[0], self.geo_span[1], ic_num)[
            #     1:-1].reshape(-1, 1)
            # xs_local = np.linspace(local_area[0], local_area[1], ic_num)[
            #     1:-1].reshape(-1, 1)
            xs = torch.linspace(self.geo_span[0], self.geo_span[1], ic_num, device="cuda")[
                1:-1].reshape(-1, 1)
            xs_local = torch.linspace(local_area[0], local_area[1], ic_num, device="cuda")[
                1:-1].reshape(-1, 1)
        elif strategy == "grid_transition":
            # xs = np.linspace(self.geo_span[0], self.geo_span[1], ic_num)[
            #     1:-1].reshape(-1, 1)
            # xs_local = np.linspace(local_area[0], local_area[1], ic_num)[
            #     1:-1].reshape(-1, 1)
            xs = torch.linspace(self.geo_span[0], self.geo_span[1], ic_num, device="cuda")[
                1:-1].reshape(-1, 1)
            xs_local = torch.linspace(local_area[0], local_area[1], ic_num, device="cuda")[
                1:-1].reshape(-1, 1)
            distance = (self.geo_span[1] - self.geo_span[0]) / (ic_num - 1)
            shift = torch.rand(1, device="cuda") * 2 * distance - distance
            xs = torch.clamp(xs + shift, self.geo_span[0], self.geo_span[1])
            # shift = np.random.uniform(-distance, distance, 1)
            # xs = np.clip(xs + shift, self.geo_span[0], self.geo_span[1])
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        # xs = np.vstack([xs, xs_local])
        # xts = np.hstack(
        #     [xs, np.full(xs.shape[0], self.time_span[0]).reshape(-1, 1)])
        xs = torch.cat([xs, xs_local], dim=0)
        xts = torch.cat(
            [xs, torch.full((xs.shape[0], 1), self.time_span[0], device="cuda")], dim=1)

        return xts.float().requires_grad_(True)


geo_span = eval(config.get("TRAIN", "GEO_SPAN"))
time_span = eval(config.get("TRAIN", "TIME_SPAN"))
sampler = GeoTimeSampler(geo_span, time_span)
net = pfp.PFPINN(
    sizes=eval(config.get("TRAIN", "NETWORK_SIZE")),
    act=torch.nn.Tanh
)

resume = config.get("TRAIN", "RESUME").strip('"')
try:
    net.load_state_dict(torch.load(resume))
    print("Load model successfully")
except:
    print("Can not load model")
    pass


ref_sol = pd.read_csv(config.get("TRAIN", "REF_PATH").strip('"'))

TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")
ref_sol["x"] = ref_sol["x"].apply(lambda x: x * GEO_COEF)
ref_sol["t"] = ref_sol["t"].apply(lambda t: t * TIME_COEF)


ic_weight = 1
bc_weight = 1
ac_weight = 1
ch_weight = 1

NTK_BATCH_SIZE = config.getint("TRAIN", "NTK_BATCH_SIZE")
BREAK_INTERVAL = config.getint("TRAIN", "BREAK_INTERVAL")
EPOCHS = config.getint("TRAIN", "EPOCHS")
ALPHA = config.getfloat("TRAIN", "ALPHA")
LR = config.getfloat("TRAIN", "LR")
num_seg = config.getint("TRAIN", "NUM_SEG")

ALPHA_PHI = config.getfloat("PARAM", "ALPHA_PHI")
OMEGA_PHI = config.getfloat("PARAM", "OMEGA_PHI")
DD = config.getfloat("PARAM", "DD")
AA = config.getfloat("PARAM", "AA")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))

causal_configs = {
    "eps": 1e-9,
    "min_thresh": 0.99,
    "step": 1.5,
    "mean_thresh": 0.6
}


def bc_func(xts):
    with torch.no_grad():
        val = (0.5 - xts[:, 0:1])
    return torch.cat([val, val], dim=1)


def ic_func(xts):
    with torch.no_grad():
        phi = (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                              torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * xts[:, 0:1] / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE + (1 - h_phi) * 0.0
    return torch.cat([phi, c], dim=1)


def split_temporal_coords_into_segments(ts, time_span, num_seg):
    # Split the temporal coordinates into segments
    # Return the indexes of the temporal coordinates
    ts = ts.cpu()
    min_t, max_t = time_span
    bins = torch.linspace(min_t, max_t, num_seg + 1, device=ts.device)
    indices = torch.bucketize(ts, bins)
    return [torch.where(indices-1 == i)[0] for i in range(num_seg)]


criteria = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=LR)

GEOTIME_SHAPE = eval(config.get("TRAIN", "GEOTIME_SHAPE"))
BCDATA_SHAPE = eval(config.get("TRAIN", "BCDATA_SHAPE"))
ICDATA_SHAPE = eval(config.get("TRAIN", "ICDATA_SHAPE"))
SAMPLING_STRATEGY = eval(config.get("TRAIN", "SAMPLING_STRATEGY"))
RAR_BASE_SHAPE = config.getint("TRAIN", "RAR_BASE_SHAPE")
RAR_SHAPE = config.getint("TRAIN", "RAR_SHAPE")


for epoch in range(EPOCHS):
    net.train()
    if epoch % BREAK_INTERVAL == 0:
        geotime, bcdata, icdata = sampler.resample(GEOTIME_SHAPE, BCDATA_SHAPE,
                                                   ICDATA_SHAPE, strateges=SAMPLING_STRATEGY)
        geotime = geotime.to(net.device)
        residual_base_data = sampler.in_sample(RAR_BASE_SHAPE, strategy="lhs")
        method = config.get("TRAIN", "ADAPTIVE_SAMPLING").strip('"')
        anchors = net.adaptive_sampling(RAR_SHAPE, residual_base_data,
                                        method=method)
        # net.train()
        data = torch.cat([geotime, anchors],
                         dim=0).detach().requires_grad_(True)
        # data = geotime.requires_grad_(True)

        # shuffle
        data = data[torch.randperm(len(data))]
        indices = split_temporal_coords_into_segments(data[:, -1],
                                                      time_span,
                                                      num_seg)

        bcdata = bcdata.to(net.device).detach().requires_grad_(True)
        icdata = icdata.to(net.device).detach().requires_grad_(True)

        fig, ax = net.plot_samplings(geotime, bcdata, icdata, anchors)
        # plt.savefig(f"/root/tf-logs/{now}/sampling-{epoch}.png",
        #             bbox_inches='tight', dpi=300)
        writer.add_figure("sampling", fig, epoch)

    # ac_residual, ch_residual = net.net_pde(data)
    bc_forward = net.net_u(bcdata)
    ic_forward = net.net_u(icdata)

    ac_seg_loss = torch.zeros(num_seg, device=net.device)
    ch_seg_loss = torch.zeros(num_seg, device=net.device)

    for seg_idx, data_idx in enumerate(indices):
        ac_residual, ch_residual = net.net_pde(data[data_idx])
        ac_seg_loss[seg_idx] = torch.mean(ac_residual**2)
        ch_seg_loss[seg_idx] = torch.mean(ch_residual**2)

    ac_causal_weights = torch.zeros(num_seg, device=net.device)
    ch_causal_weights = torch.zeros(num_seg, device=net.device)
    for seg_idx in range(num_seg):
        if seg_idx == 0:
            ac_causal_weights[seg_idx] = 1
            ch_causal_weights[seg_idx] = 1
        else:
            ac_causal_weights[seg_idx] = torch.exp(
                -causal_configs["eps"] * torch.sum(ac_seg_loss[:seg_idx])).item()
            ch_causal_weights[seg_idx] = torch.exp(
                -causal_configs["eps"] * torch.sum(ch_seg_loss[:seg_idx])).item()

    if ac_causal_weights[-1] > causal_configs["min_thresh"] \
            and ch_causal_weights[-1] > causal_configs["min_thresh"]:
        causal_configs["eps"] *= causal_configs["step"]
        print(f"epoch {epoch}: "
              f"increase eps to {causal_configs['eps']:.2e}")

    if torch.mean(ac_causal_weights) < causal_configs["mean_thresh"] \
            or torch.mean(ch_causal_weights) < causal_configs["mean_thresh"]:
        causal_configs["eps"] /= causal_configs["step"]
        print(f"epoch {epoch}: "
              f"decrease eps to {causal_configs['eps']:.2e}")

    ac_loss = torch.sum(ac_seg_loss * ac_causal_weights)
    ch_loss = torch.sum(ch_seg_loss * ch_causal_weights)
    bc_loss = torch.mean((bc_forward - bc_func(bcdata))**2)
    ic_loss = torch.mean((ic_forward - ic_func(icdata))**2)

    if epoch % BREAK_INTERVAL == 0:
        ac_weight, ch_weight, bc_weight, ic_weight = net.compute_gradient_weight(
            [ac_loss, ch_loss, bc_loss, ic_loss],)

        for weight in [ac_weight, ch_weight, bc_weight, ic_weight]:
            if np.isnan(weight):
                raise ValueError("NaN weight")

    losses = ac_weight * ac_loss + ch_weight * ch_loss + \
        bc_weight * bc_loss + ic_weight * ic_loss

    opt.zero_grad()
    losses.backward()
    opt.step()

    if epoch % BREAK_INTERVAL == 0:
        print(f"epoch {epoch}: ac_loss {ac_loss:.2e}, ch_loss {ch_loss:.2e}, "
              f"bc_loss {bc_loss:.2e}, ic_loss {ic_loss:.2e}, "
              f"ac_weight {ac_weight:.2e}, ch_weight {ch_weight:.2e}, "
              f"bc_weight {bc_weight:.2e}, ic_weight {ic_weight:.2e}")

        writer.add_scalar("loss/ac_loss", ac_loss, epoch)
        writer.add_scalar("loss/ch_loss", ch_loss, epoch)
        writer.add_scalar("loss/bc_loss", bc_loss, epoch)
        writer.add_scalar("loss/ic_loss", ic_loss, epoch)
        writer.add_scalar("weight/ac_weight", ac_weight, epoch)
        writer.add_scalar("weight/ch_weight", ch_weight, epoch)
        writer.add_scalar("weight/bc_weight", bc_weight, epoch)
        writer.add_scalar("weight/ic_weight", ic_weight, epoch)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(ac_causal_weights.cpu().numpy(), label="ac")
        ax.plot(ch_causal_weights.cpu().numpy(), label="ch")
        ax.set_title(f"epoch: {epoch} "
                     f"eps: {causal_configs['eps']:.2e}")
        ax.legend(loc="upper right")
        # close the figure
        plt.close(fig)
        writer.add_figure("fig/causal_weights", fig, epoch)

        fig, ax, acc = net.plot_predict(ref_sol=ref_sol, epoch=epoch)

        torch.save(net.state_dict(), f"/root/tf-logs/{now}/model-{epoch}.pt")
        writer.add_figure("fig/predict", fig, epoch)
        writer.add_scalar("acc", acc, epoch)
        plt.close(fig)
