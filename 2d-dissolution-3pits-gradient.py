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


# now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
now = "3pits-gradient-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter(log_dir="/root/tf-logs/" + now)
save_root = "/root/tf-logs"


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
        # mins = [self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]]
        # maxs = [self.geo_span[0][1], self.geo_span[1][1], np.sqrt(self.time_span[1])]
        # geotime = func(mins=mins, maxs=maxs, num=in_num)
        # geotime[:, -1] = geotime[:, -1]**2
        
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
        xyts_left = xyts.clone()
        xyts_left[:, 0:1] -= 0.15
        xyts_right = xyts.clone()
        xyts_right[:, 0:1] += 0.15
        xyts_top = xyts.clone()
        xyts_top[:, 1:2] += 0.475

        # xts = func(mins=[self.geo_span[0][0], self.time_span[0]],
        #            maxs=[self.geo_span[0][1], self.time_span[1]],
        #            num=bc_num)
        # top = torch.cat([xts[:, 0:1], torch.full(
        #             (xts.shape[0], 1), self.geo_span[1][1], device="cuda"), xts[:, 1:2]], dim=1)  # 顶边

        yts = func(mins=[self.geo_span[1][0], self.time_span[0]],
                   maxs=[self.geo_span[1][1], self.time_span[1]],
                   num=bc_num)
        # left = torch.cat([torch.full((yts.shape[0], 1), self.geo_span[0]
        #                              [0], device="cuda"), yts[:, 0:1], yts[:, 1:2]], dim=1)  # 左边
        # right = torch.cat([torch.full((yts.shape[0], 1), self.geo_span[0]
        #                               [1], device="cuda"), yts[:, 0:1], yts[:, 1:2]], dim=1)  # 右边

        xyts = torch.cat([xyts_left, xyts_right, xyts_top,], dim=0)

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
                                                        maxs=[
                self.geo_span[0][1], self.geo_span[1][1]],
                num=ic_num)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        xys_local_1 = pfp.make_lhs_sampling_data(mins=[-0.3, 0],
                                                 maxs=[0.3, 0.15],
                                                 num=ic_num*4)
        xys_local_2 = pfp.make_lhs_sampling_data(mins=[-0.15, 0.35],
                                                 maxs=[0.15, 0.50],
                                                 num=ic_num*2)
        xys = torch.cat([xys, xys_local_1, xys_local_2], dim=0)
        xyts = torch.cat(
            [xys, torch.full((xys.shape[0], 1), self.time_span[0], device="cuda")], dim=1)
        return xyts.float().requires_grad_(True)


geo_span = eval(config.get("TRAIN", "GEO_SPAN"))
time_span = eval(config.get("TRAIN", "TIME_SPAN"))
sampler = GeoTimeSampler(geo_span, time_span)
net = pfp.PFPINN(
    # sizes=eval(config.get("TRAIN", "NETWORK_SIZE")),
    act=torch.nn.Tanh
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
DD = config.getfloat("PARAM", "DD")
AA = config.getfloat("PARAM", "AA")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))
MESH_POINTS = np.load(config.get("TRAIN", "MESH_POINTS").strip('"')) * GEO_COEF

num_seg = config.getint("TRAIN", "NUM_SEG")

causal_configs = {
    "eps": 1e-12,
    "min_thresh": 0.99,
    "step": 10,
    "mean_thresh": 0.5
}


def cal_r(pts):
    # pts: x, y, t
    lower_pts_ids = torch.where(pts[:, 1] <= 0.25)[0]
    upper_pts_ids = torch.where(pts[:, 1] > 0.25)[0]
    r = torch.zeros_like(pts[:, 0])
    r[lower_pts_ids] = torch.sqrt((torch.abs(pts[lower_pts_ids, 0]) - 0.15)**2
                                  + pts[lower_pts_ids, 1]**2)
    r[upper_pts_ids] = torch.sqrt(pts[upper_pts_ids, 0]**2
                                  + (pts[upper_pts_ids, 1] - 0.50)**2)
    return r.reshape(-1, 1)


def ic_func(xts):
    r = cal_r(xts)
    with torch.no_grad():
        phi = 1 - (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                                  torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * (r-0.05) / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE
    return torch.cat([phi.reshape(-1, 1), c.reshape(-1, 1)], dim=1).detach()


def bc_func(xts):
    r = cal_r(xts)
    with torch.no_grad():
        phi = (r > 0.05).float()
        c = phi.detach()
    return torch.cat([phi, c], dim=1).detach()


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
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.9)

GEOTIME_SHAPE = eval(config.get("TRAIN", "GEOTIME_SHAPE"))
BCDATA_SHAPE = eval(config.get("TRAIN", "BCDATA_SHAPE"))
ICDATA_SHAPE = eval(config.get("TRAIN", "ICDATA_SHAPE"))
SAMPLING_STRATEGY = eval(config.get("TRAIN", "SAMPLING_STRATEGY"))
RAR_BASE_SHAPE = config.getint("TRAIN", "RAR_BASE_SHAPE")
RAR_SHAPE = config.getint("TRAIN", "RAR_SHAPE")


for epoch in range(EPOCHS):
    net.train()
    # need_causal = not (causal_configs["eps"] > 1e-10 and epoch > 12000)
    need_causal = True
    # need_causal = epoch < 15000
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
        if need_causal:
            indices = split_temporal_coords_into_segments(data[:, -1],
                                                          time_span,
                                                          num_seg)

        bcdata = bcdata.to(net.device).detach().requires_grad_(True)
        icdata = icdata.to(net.device).detach().requires_grad_(True)

        if epoch % (10*BREAK_INTERVAL) == 0:
            fig, ax = net.plot_samplings(geotime, bcdata, icdata, anchors)
            # plt.savefig(f"/root/tf-logs/{now}/sampling-{epoch}.png",
            #             bbox_inches='tight', dpi=300)
            writer.add_figure("sampling", fig, epoch)

    ac_residual, ch_residual = net.net_pde(data)
    bc_forward = net.net_u(bcdata)
    ic_forward = net.net_u(icdata)

    if need_causal:
        ac_seg_loss = torch.zeros(num_seg, device=net.device)
        ch_seg_loss = torch.zeros(num_seg, device=net.device)

        for seg_idx, data_idx in enumerate(indices):
            ac_seg_residual = ac_residual[data_idx]
            ch_seg_residual = ch_residual[data_idx]
            ac_seg_loss[seg_idx] = torch.mean(ac_seg_residual**2)
            ch_seg_loss[seg_idx] = torch.mean(ch_seg_residual**2)

        ac_causal_weights = torch.zeros(num_seg, device=net.device)
        ch_causal_weights = torch.zeros(num_seg, device=net.device)
        for seg_idx in range(num_seg):
            if seg_idx == 0:
                ac_causal_weights[seg_idx] = 1
                ch_causal_weights[seg_idx] = 1
            else:
                ac_causal_weights[seg_idx] = torch.exp(
                    -causal_configs["eps"] * torch.sum(ac_seg_loss[:seg_idx])).detach()
                ch_causal_weights[seg_idx] = torch.exp(
                    -causal_configs["eps"] * torch.sum(ch_seg_loss[:seg_idx])).detach()

        if ac_causal_weights[-1] > causal_configs["min_thresh"] \
                and ch_causal_weights[-1] > causal_configs["min_thresh"]:
            causal_configs["eps"] *= causal_configs["step"]
            print(f"epoch {epoch}: "
                  f"increase eps to {causal_configs['eps']:.2e}")
            writer.add_scalar("causal/eps", causal_configs["eps"], epoch)
        if torch.mean(ac_causal_weights) < causal_configs["mean_thresh"] \
                or torch.mean(ch_causal_weights) < causal_configs["mean_thresh"]:
            causal_configs["eps"] /= causal_configs["step"]
            print(f"epoch {epoch}: "
                  f"decrease eps to {causal_configs['eps']:.2e}")
            writer.add_scalar("causal/eps", causal_configs["eps"], epoch)

        ac_loss = torch.sum(ac_seg_loss * ac_causal_weights)
        ch_loss = torch.sum(ch_seg_loss * ch_causal_weights)

    else:
        ac_loss = torch.mean(ac_residual**2)
        ch_loss = torch.mean(ch_residual**2)

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
    scheduler.step()

    if epoch % BREAK_INTERVAL == 0:
        print(f"epoch {epoch}: ac_loss {ac_loss:.2e}, ch_loss {ch_loss:.2e}, "
              f"bc_loss {bc_loss:.2e}, ic_loss {ic_loss:.2e}, "
              f"ac_weight {ac_weight:.2e}, ch_weight {ch_weight:.2e}, "
              f"bc_weight {bc_weight:.2e}, ic_weight {ic_weight:.2e}")

        writer.add_scalar("loss/ac_loss", ac_loss, epoch)
        writer.add_scalar("loss/ch_loss", ch_loss, epoch)
        writer.add_scalar("loss/bc_loss", bc_loss, epoch)
        writer.add_scalar("loss/ic_loss", ic_loss, epoch)
        writer.add_scalar("loss/total", losses, epoch)
        writer.add_scalar("weight/ac_weight", ac_weight, epoch)
        writer.add_scalar("weight/ch_weight", ch_weight, epoch)
        writer.add_scalar("weight/bc_weight", bc_weight, epoch)
        writer.add_scalar("weight/ic_weight", ic_weight, epoch)

        if need_causal:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(ac_causal_weights.cpu().numpy(), label="ac")
            ax.plot(ch_causal_weights.cpu().numpy(), label="ch")
            ax.set_title(f"epoch: {epoch} "
                         f"eps: {causal_configs['eps']:.2e}")
            ax.legend(loc="upper right")
            # close the figure
            plt.close(fig)
            writer.add_figure("fig/causal_weights", fig, epoch)

        TARGET_TIMES = eval(config.get("TRAIN", "TARGET_TIMES"))

        REF_PREFIX = config.get("TRAIN", "REF_PREFIX").strip('"')

        fig, ax, acc = net.plot_predict(ts=TARGET_TIMES,
                                        mesh_points=MESH_POINTS,
                                        ref_prefix=REF_PREFIX)

        torch.save(net.state_dict(), f"{save_root}/{now}/model-{epoch}.pt")
        writer.add_figure("fig/predict", fig, epoch)
        writer.add_scalar("acc", acc, epoch)
        plt.close(fig)
