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
now = "causal+gradient+" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter(log_dir="/root/tf-logs/" + now)
# writer = SummaryWriter(log_dir="/root/tf-logs/" + now)
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
        # geotime = func(mins=[self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]],
        #                maxs=[self.geo_span[0][1], self.geo_span[1]
        #                      [1], self.time_span[1]],
        #                num=in_num)

        mins = [self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]]
        maxs = [self.geo_span[0][1], self.geo_span[1][1], np.sqrt(self.time_span[1])]
        geotime = func(mins=mins, maxs=maxs, num=in_num)
        geotime[:, -1] = geotime[:, -1]**2
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

        xyts = pfp.make_lhs_sampling_data(mins=[-0.05, 0, self.time_span[0]+self.time_span[1]*0.1],
                                          maxs=[0.05, 0.025,
                                                self.time_span[1]],
                                          num=bc_num)
        xyts = xyts[xyts[:, 0] ** 2 + xyts[:, 1] ** 2 <= 0.025 ** 2]

        xts = func(mins=[self.geo_span[0][0], self.time_span[0]],
                   maxs=[self.geo_span[0][1], self.time_span[1]],
                   num=bc_num)

        top = torch.cat([xts[:, 0:1],
                        torch.full(
                            (xts.shape[0], 1), self.geo_span[1][1], device=xts.device),
                        xts[:, 1:2]], dim=1)  # 顶边

        yts = func(mins=[self.geo_span[1][0], self.time_span[0]],
                   maxs=[self.geo_span[1][1], self.time_span[1]],
                   num=bc_num)

        left = torch.cat([torch.full((yts.shape[0], 1), self.geo_span[0][0], device=yts.device),
                          yts[:, 0:1],
                          yts[:, 1:2]], dim=1)  # 左边

        right = torch.cat([torch.full((yts.shape[0], 1), self.geo_span[0][1], device=yts.device),
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
                                                        maxs=[
                                                            self.geo_span[0][1], self.geo_span[1][1]],
                                                        num=ic_num)

        else:
            raise ValueError(f"Unknown strategy {strategy}")
        xys_local = pfp.make_lhs_sampling_data(mins=[-0.1, 0],
                                               maxs=[0.1, 0.1,],
                                               num=ic_num*2)
        xys = torch.cat([xys, xys_local], dim=0)
        xyts = torch.cat([xys, torch.full((xys.shape[0], 1),
                         self.time_span[0], device=xys.device)], dim=1)
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
    "eps": 1e-9,
    "min_thresh": 0.99,
    "step": 10,
    "mean_thresh": 0.6
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
    # bins = torch.linspace(min_t, max_t, num_seg + 1, device=ts.device)
    bins = torch.linspace(min_t, max_t**(1/2), num_seg + 1, device=ts.device) ** 2
    indices = torch.bucketize(ts, bins)
    return [torch.where(indices-1 == i)[0] for i in range(num_seg)]



criteria = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.8)

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
        residual_base_data = sampler.in_sample(RAR_BASE_SHAPE, strategy="lhs")
        method = config.get("TRAIN", "ADAPTIVE_SAMPLING").strip('"')
        anchors = net.adaptive_sampling(RAR_SHAPE, residual_base_data,
                                        method=method)
        
        geotime = geotime.to(net.device).detach().requires_grad_(True)
        anchors = anchors.to(net.device).detach().requires_grad_(True)
        

        if need_causal:
            indices_geotime = split_temporal_coords_into_segments(geotime[:, -1],
                                                          time_span,
                                                          num_seg)
            indices_anchors = split_temporal_coords_into_segments(anchors[:, -1],
                                                            time_span,
                                                            num_seg)

            

        bcdata = bcdata.to(net.device).detach().requires_grad_(True)
        icdata = icdata.to(net.device).detach().requires_grad_(True)

        if epoch % (10*BREAK_INTERVAL) == 0:
            fig, ax = net.plot_samplings(geotime, bcdata, icdata, anchors)
            writer.add_figure("sampling", fig, epoch)

    ac_residual_geotime, ch_residual_geotime = net.net_pde(geotime)
    ac_residual_anchors, ch_residual_anchors = net.net_pde(anchors)
    bc_forward = net.net_u(bcdata)
    ic_forward = net.net_u(icdata)


    ac_seg_loss = torch.zeros(num_seg, device=net.device)
    ch_seg_loss = torch.zeros(num_seg, device=net.device)

    # for seg_idx, data_idx in enumerate(indices):
    #     ac_seg_residual = ac_residual[data_idx]
    #     ch_seg_residual = ch_residual[data_idx]
    #     ac_seg_loss[seg_idx] = torch.mean(ac_seg_residual**2)
    #     ch_seg_loss[seg_idx] = torch.mean(ch_seg_residual**2)
    
    for seg_idx in range(num_seg):
        seg_data_idx_geotime = indices_geotime[seg_idx]
        seg_data_idx_anchors = indices_anchors[seg_idx]
        
        if len(seg_data_idx_anchors) == 0 or len(seg_data_idx_geotime) == 0:
            ac_seg_loss[seg_idx] = torch.tensor(0., device=net.device)
            ch_seg_loss[seg_idx] = torch.tensor(0., device=net.device)
            continue
        
        
        ac_seg_residual_geotime = ac_residual_geotime[seg_data_idx_geotime]
        ch_seg_residual_geotime = ch_residual_geotime[seg_data_idx_geotime]
        ac_seg_residual_anchors = ac_residual_anchors[seg_data_idx_anchors]
        ch_seg_residual_anchors = ch_residual_anchors[seg_data_idx_anchors]
        
        ac_seg_loss_geotime = torch.mean(ac_seg_residual_geotime**2)
        ch_seg_loss_geotime = torch.mean(ch_seg_residual_geotime**2)
        ac_seg_loss_anchors = torch.mean(ac_seg_residual_anchors**2)
        ch_seg_loss_anchors = torch.mean(ch_seg_residual_anchors**2)
        
        # geotime 与 anchors 自适应损失
        # ac_seg_weight_geotime, ac_seg_weight_anchors = net.compute_gradient_weight(
        #     [ac_seg_loss_geotime, ac_seg_loss_anchors])
        # ch_seg_weight_geotime, ch_seg_weight_anchors = net.compute_gradient_weight(
        #     [ch_seg_loss_geotime, ch_seg_loss_anchors])
        # ac_seg_loss[seg_idx] = ac_seg_loss_geotime +  ac_seg_loss_anchors * ac_seg_weight_anchors / ac_seg_weight_geotime
        # ch_seg_loss[seg_idx] = ch_seg_loss_geotime +  ch_seg_loss_anchors * ch_seg_weight_anchors / ch_seg_weight_geotime
        
        # geotime 与 anchors 固定权重
        ac_seg_loss[seg_idx] = ac_seg_loss_geotime + ac_seg_loss_anchors / 5
        ch_seg_loss[seg_idx] = ch_seg_loss_geotime + ch_seg_loss_anchors / 5

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
            and ch_causal_weights[-1] > causal_configs["min_thresh"] \
            and causal_configs["eps"] < 1:
        causal_configs["eps"] *= causal_configs["step"]
        print(f"epoch {epoch}: "
                f"increase eps to {causal_configs['eps']:.2e}")
        writer.add_scalar("causal/eps", causal_configs["eps"], epoch)
    # if torch.mean(ac_causal_weights) < causal_configs["mean_thresh"] \
    #         or torch.mean(ch_causal_weights) < causal_configs["mean_thresh"]:
    #     causal_configs["eps"] /= causal_configs["step"]
    #     print(f"epoch {epoch}: "
    #           f"decrease eps to {causal_configs['eps']:.2e}")
    #     writer.add_scalar("causal/eps", causal_configs["eps"], epoch)

    ac_loss = torch.sum(ac_seg_loss * ac_causal_weights)
    ch_loss = torch.sum(ch_seg_loss * ch_causal_weights)

    bc_loss = criteria(bc_forward, bc_func(bcdata).detach())
    ic_loss = criteria(ic_forward, ic_func(icdata).detach())
    # bc_loss = torch.mean((bc_forward - bc_func(bcdata))**2)
    # ic_loss = torch.mean((ic_forward - ic_func(icdata))**2)
    
    # an excepetion: `ac_loss` and `ch_loss` might be NaN or Inf
    # if this happens, we should raise an error
    if torch.isnan(ac_loss) or torch.isnan(ch_loss):
        raise ValueError("NaN loss")
    if torch.isinf(ac_loss) or torch.isinf(ch_loss):
        raise ValueError("Inf loss")


    if epoch % BREAK_INTERVAL == 0:
        if bc_loss > 1e-10:
            ac_weight, ch_weight, bc_weight, ic_weight = net.compute_gradient_weight(
                [ac_loss, ch_loss, bc_loss, ic_loss],)
        else:
            ac_weight, ch_weight, ic_weight = net.compute_gradient_weight(
                [ac_loss, ch_loss, ic_loss],)

        for weight in [ac_weight, ch_weight, bc_weight, ic_weight]:
            if np.isnan(weight):
                raise ValueError("NaN weight")
    
    eps_ic = 1.0
    losses = ac_weight * ac_loss + ch_weight * ch_loss + \
        bc_weight * bc_loss + eps_ic*ic_weight * ic_loss
        
    if epoch % BREAK_INTERVAL == 0:
        grads = net.gradient(losses)
        writer.add_scalar("grad/grads", grads.abs().mean(), epoch)

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
        
        if epoch % (10*BREAK_INTERVAL) == 0:
            if need_causal:
                bins = torch.linspace(time_span[0], time_span[1]**(1/2), num_seg + 1, device=net.device)**2
                ts = (bins[1:] + bins[:-1]) / 2 / TIME_COEF
                ts = ts.detach().cpu().numpy()
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                ax = axes[0]
                ax.plot(ts)
                ax.set_title("time segments")
                ax.set_ylabel("time (s)")
                
                ax = axes[1]
                ax.plot(ts, ac_causal_weights.cpu().numpy(), label="ac")
                ax.plot(ts, ch_causal_weights.cpu().numpy(), label="ch")
                ax.set_title(f"eps: {causal_configs['eps']:.2e}")
                ax.set_ylabel("Causal Weights")
                ax.legend(loc="upper right")

                ax = axes[2]
                ax.plot(ts, ac_seg_loss.detach().cpu().numpy(), label="ac")
                ax.set_title("AC segment loss")
                ax.set_ylabel("AC segment loss")

                ax = axes[3]
                ax.plot(ts, ch_seg_loss.detach().cpu().numpy(), label="ch")
                ax.set_title("CH segment loss")
                ax.set_ylabel("CH segment loss")

                # figure title 
                fig.suptitle(f"epoch: {epoch} ")
                # close the figure
                plt.close(fig)
                writer.add_figure("fig/causal_weights", fig, epoch)

        TARGET_TIMES = eval(config.get("TRAIN", "TARGET_TIMES"))
        REF_PREFIX = config.get("TRAIN", "REF_PREFIX").strip('"')
        
        if epoch % (10*BREAK_INTERVAL) == 0:
            fig, acc = net.plot_predict(ts=TARGET_TIMES,
                                            mesh_points=MESH_POINTS,
                                            ref_prefix=REF_PREFIX)

            torch.save(net.state_dict(), f"{save_root}/{now}/model-{epoch}.pt")
            writer.add_figure("fig/predict", fig, epoch)
            writer.add_scalar("acc", acc, epoch)
            plt.close(fig)
