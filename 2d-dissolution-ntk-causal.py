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
matplotlib.use("Agg")

config = configparser.ConfigParser()
config.read("config.ini")


now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter(log_dir="/root/tf-logs/" + now)


class GeoTimeSampler:
    def __init__(
        self,
        geo_span: list,  # 2d
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
        geotime = func(mins=[self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]],
                       maxs=[self.geo_span[0][1], self.geo_span[1]
                             [1], self.time_span[1]],
                       num=in_num)

        return geotime.float().requires_grad_(True)

    # TODO: bc
    def bc_sample(self, bc_num: int, strategy: str = "lhs", xspan=[-0.025, 0.025]):
        # 四条边，顺着时间变成四个面
        if strategy == "lhs":
            func = pfp.make_lhs_sampling_data
        elif strategy == "grid":
            func = pfp.make_uniform_grid_data
        elif strategy == "grid_transition":
            func = pfp.make_uniform_grid_data_transition
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        xyts = pfp.make_lhs_sampling_data(mins=[-0.025, 0, self.time_span[0]],
                                          maxs=[0.025, 0.025,
                                                self.time_span[1]],
                                          num=bc_num)
        xyts = xyts[xyts[:, 0] ** 2 + xyts[:, 1] ** 2 <= 0.025 ** 2]

        xts = func(mins=[self.geo_span[0][0], self.time_span[0]],
                   maxs=[self.geo_span[0][1], self.time_span[1]],
                   num=bc_num)
        
        top = torch.cat([xts[:, 0:1], 
                        torch.full((xts.shape[0], 1), self.geo_span[1][1], device=xts.device), 
                        xts[:, 1:2]], dim=1)  # 顶边
        
        yts = func(mins=[self.geo_span[1][0], self.time_span[0]],
                   maxs=[self.geo_span[1][1], self.time_span[1]],
                   num=bc_num)
        

        left = torch.cat([torch.full((yts.shape[0], 1), self.geo_span[0][0], device=xts.device), 
                        yts[:, 0:1], 
                        yts[:, 1:2]], dim=1)  # 左边

        right = torch.cat([torch.full((yts.shape[0], 1), self.geo_span[0][1], device=xts.device), 
                        yts[:, 0:1], 
                        yts[:, 1:2]], dim=1)  # 右边

        xyts = torch.cat([xyts, top, left, right], dim=0)

        return xyts.float().requires_grad_(True)


    def ic_sample(self, ic_num, strategy: str = "lhs", local_area=[[-0.1, 0.1], [0, 0.1]]):
        if strategy == "lhs":

            xys = pfp.make_lhs_sampling_data(mins=[self.geo_span[0][0], self.geo_span[1][0]],
                                             maxs=[self.geo_span[0][1],
                                                   self.geo_span[1][1]],
                                             num=ic_num)

        elif strategy == "grid":
            xys = pfp.make_uniform_grid_data(mins=[self.geo_span[0][0], self.geo_span[1][0]],
                                             maxs=[self.geo_span[0][1],
                                                   self.geo_span[1][1]],
                                             num=ic_num)
        elif strategy == "grid_transition":
            xys = pfp.make_uniform_grid_data_transition(mins=[self.geo_span[0][0], self.geo_span[1][0]],
                                                        maxs=[self.geo_span[0][1], self.geo_span[1][1]],
                num=ic_num)

        else:
            raise ValueError(f"Unknown strategy {strategy}")
        xys_local = pfp.make_semi_circle_data(radius=0.1, num=ic_num, center=[0, 0.])
        xys = torch.cat([xys, xys_local], dim=0)
        xyts = torch.cat([xys, torch.full((xys.shape[0], 1), self.time_span[0], device=xys.device)], dim=1)
        return xyts.float().requires_grad_(True)

geo_span = eval(config.get("TRAIN", "GEO_SPAN"))
time_span = eval(config.get("TRAIN", "TIME_SPAN"))
sampler = GeoTimeSampler(geo_span, time_span)
net = pfp.PFPINN(
    sizes=eval(config.get("TRAIN", "NETWORK_SIZE")),
    act=torch.nn.GELU
)

resume = config.get("TRAIN", "RESUME").strip('"')
try:
    net.load_state_dict(torch.load(resume))
    print("Load model successfully")
except:
    print("Can not load model")
    pass


TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")


ic_weight = 1
bc_weight = 1
ac_weight = 1
ch_weight = 1

NTK_BATCH_SIZE = config.getint("TRAIN", "NTK_BATCH_SIZE")
BREAK_INTERVAL = config.getint("TRAIN", "BREAK_INTERVAL")
EPOCHS = config.getint("TRAIN", "EPOCHS")
ALPHA = config.getfloat("TRAIN", "ALPHA")
LR = config.getfloat("TRAIN", "LR")

ALPHA_PHI = config.getfloat("PARAM", "ALPHA_PHI")
OMEGA_PHI = config.getfloat("PARAM", "OMEGA_PHI")
AA = config.getfloat("PARAM", "AA")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))
MESH_POINTS = np.load(config.get("TRAIN", "MESH_POINTS").strip('"')) * GEO_COEF

num_seg = config.getint("TRAIN", "NUM_SEG")

causal_configs = {
    "eps": 1e-18,
    "delta": 0.99
}


def ic_func(xts):
    r = torch.sqrt(xts[:, 0:1]**2 + xts[:, 1:2]**2).detach()
    with torch.no_grad():
        phi = 1 - (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                                  torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * (r-0.05) / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE
    return torch.cat([phi, c], dim=1)


def bc_func(xts):
    r = torch.sqrt(xts[:, 0:1]**2 + xts[:, 1:2]**2).detach()
    with torch.no_grad():
        phi = (r > 0.05).float()
        c = phi.detach()
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
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20000, gamma=0.9)

GEOTIME_SHAPE = eval(config.get("TRAIN", "GEOTIME_SHAPE"))
BCDATA_SHAPE = eval(config.get("TRAIN", "BCDATA_SHAPE"))
ICDATA_SHAPE = eval(config.get("TRAIN", "ICDATA_SHAPE"))
SAMPLING_STRATEGY = eval(config.get("TRAIN", "SAMPLING_STRATEGY"))
RAR_BASE_SHAPE = config.getint("TRAIN", "RAR_BASE_SHAPE")
RAR_SHAPE = config.getint("TRAIN", "RAR_SHAPE")


for epoch in range(EPOCHS):
    need_causal = causal_configs["eps"] < 1e-12
    net.train()
    if epoch % BREAK_INTERVAL == 0:
        geotime, bcdata, icdata = sampler.resample(GEOTIME_SHAPE, BCDATA_SHAPE,
                                                   ICDATA_SHAPE, strateges=SAMPLING_STRATEGY)
        geotime = geotime.to(net.device)
        residual_base_data = sampler.in_sample(RAR_BASE_SHAPE, strategy="lhs")
        method = config.get("TRAIN", "ADAPTIVE_SAMPLING").strip('"')
        anchors = net.adaptive_sampling(RAR_SHAPE, residual_base_data,
                                        method=method)
        net.train()
        data = torch.cat([geotime, anchors],
                         dim=0).detach().requires_grad_(True)
        # data = geotime.requires_grad_(True)

        # shuffle
        data = data[torch.randperm(len(data))]
        # if need_causal:
        #     indices = split_temporal_coords_into_segments(data[:, -1],
        #                                                 time_span,
        #                                                 num_seg)

        bcdata = bcdata.to(net.device).detach().requires_grad_(True)
        icdata = icdata.to(net.device).detach().requires_grad_(True)

        fig, ax = net.plot_samplings(geotime, bcdata, icdata, anchors)
        # plt.savefig(f"/root/tf-logs/{now}/sampling-{epoch}.png",
        #             bbox_inches='tight', dpi=300)
        writer.add_figure("sampling", fig, epoch)

    FORWARD_BATCH_SIZE = config.getint("TRAIN", "FORWARD_BATCH_SIZE")

    ac_residual, ch_residual = net.net_pde(data)
    bc_forward = net.net_u(bcdata)  
    ic_forward = net.net_u(icdata)


    if epoch % BREAK_INTERVAL == 0:

        ac_weight, ch_weight, bc_weight, ic_weight = \
            net.compute_ntk_weight(
                [ac_residual, ch_residual, bc_forward, ic_forward],
                method="random",
                batch_size=NTK_BATCH_SIZE
            )

        writer.add_scalar("weight/ic", ic_weight, epoch)
        writer.add_scalar("weight/bc", bc_weight, epoch)
        writer.add_scalar("weight/ac", ac_weight, epoch)
        writer.add_scalar("weight/ch", ch_weight, epoch)

        TARGET_TIMES = eval(config.get("TRAIN", "TARGET_TIMES"))

        REF_PREFIX = config.get("TRAIN", "REF_PREFIX").strip('"')

        fig, ax, acc = net.plot_predict(ts=TARGET_TIMES,
                                        mesh_points=MESH_POINTS,
                                        ref_prefix=REF_PREFIX)

        torch.save(net.state_dict(), f"/root/tf-logs/{now}/model-{epoch}.pt")

        writer.add_figure("fig/predict", fig, epoch)
        writer.add_scalar("acc", acc, epoch)
        
    # if need_causal:
    #     ac_seg_loss = torch.zeros(num_seg, device=net.device)
    #     ch_seg_loss = torch.zeros(num_seg, device=net.device)

    #     for seg_idx, data_idx in enumerate(indices):
    #         ac_residual_seg = ac_residual[data_idx]
    #         ch_residual_seg = ch_residual[data_idx]
            
    #         ac_seg_loss[seg_idx] = criteria(ac_residual_seg,
    #                                         torch.zeros_like(ac_residual_seg)) * ac_weight
    #         ch_seg_loss[seg_idx] = criteria(ch_residual_seg,
    #                                         torch.zeros_like(ch_residual_seg)) * ch_weight

    #     ac_causal_weights = torch.zeros(num_seg, device=net.device)
    #     ch_causal_weights = torch.zeros(num_seg, device=net.device)
    #     for seg_idx in range(num_seg):
    #         if seg_idx == 0:
    #             ac_causal_weights[seg_idx] = 1
    #             ch_causal_weights[seg_idx] = 1
    #         else:
    #             ac_causal_weights[seg_idx] = torch.exp(
    #                 -causal_configs["eps"] * torch.sum(ac_seg_loss[:seg_idx])).detach()
    #             ch_causal_weights[seg_idx] = torch.exp(
    #                 -causal_configs["eps"] * torch.sum(ch_seg_loss[:seg_idx])).detach()

    #     if torch.min(ac_causal_weights) > causal_configs["delta"] \
    #             and torch.min(ch_causal_weights) > causal_configs["delta"]:
    #         causal_configs["eps"] *= 10
    #         print(f"epoch: {epoch}, "
    #             f"update eps: {causal_configs['eps']}")
            
    #     if epoch % (BREAK_INTERVAL) == 0:
    #         fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #         ax.plot(ac_causal_weights.cpu().numpy(), label="ac")
    #         ax.plot(ch_causal_weights.cpu().numpy(), label="ch")
    #         ax.legend()
    #         ax.set_title(f"Eposh: {epoch} "
    #                     f"Delta: {causal_configs['delta']:.2f} "
    #                     f"eps: {causal_configs['eps']:.2e}")
    #         writer.add_figure("causal_weights", fig, epoch)
    #         # close the figure to save memory
    #         plt.close(fig)

    #     ac_loss_weighted = torch.sum(ac_causal_weights * ac_seg_loss)
    #     ch_loss_weighted = torch.sum(ch_causal_weights * ch_seg_loss)
    # else:
    ac_loss_weighted = criteria(ac_residual, torch.zeros_like(ac_residual)) * ac_weight
    ch_loss_weighted = criteria(ch_residual, torch.zeros_like(ch_residual)) * ch_weight
        
    ic_loss_weighted = criteria(bc_forward, bc_func(bcdata).detach()) * ic_weight
    bc_loss_weighted = criteria(ic_forward, ic_func(icdata).detach()) * bc_weight



    losses = ic_loss_weighted \
        + bc_loss_weighted \
        + ac_loss_weighted \
        + ch_loss_weighted

    if epoch % (BREAK_INTERVAL) == 0:
        writer.add_scalar("loss/total", losses, epoch)
        writer.add_scalar("loss/ac_loss",
                          ac_loss_weighted, epoch)
        writer.add_scalar("loss/ch_loss",
                          ch_loss_weighted, epoch)
        writer.add_scalar("loss/ic_loss",
                          ic_loss_weighted, epoch)
        writer.add_scalar("loss/bc_loss",
                          bc_loss_weighted, epoch)

        print(f"epoch: {epoch}, "
              f"ic_loss: {ic_loss_weighted.item():.4e}, "
              f"bc_loss: {bc_loss_weighted.item():.4e}, "
              f"ac_loss: {ac_loss_weighted.item():.4e}, "
              f"ch_loss: {ch_loss_weighted.item():.4e}, ")

    opt.zero_grad()
    losses.backward()
    opt.step()
    # if opt.param_groups[0]['lr'] > 1e-6:
    scheduler.step()

print("Done")
