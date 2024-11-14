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
now = "3d-2pits-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
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
        
        geotime = func(
            mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0], self.time_span[0]],
            maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1], self.time_span[1]],
            num=in_num
        )

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

        xyzts = pfp.make_lhs_sampling_data(mins=[-0.04, -0.04, 0, self.time_span[0]+self.time_span[1]*0.05],
                                          maxs=[0.04, 0.04, 0.04,self.time_span[1]],
                                          num=bc_num)
        xyzts = xyzts[xyzts[:, 0] ** 2 + xyzts[:, 1] ** 2 + xyzts[:, 2] ** 2 <= 0.02 ** 2]
        
        xyzts_left = xyzts.clone()
        xyzts_left[:, 0:1] -= 0.20
        xyzts_right = xyzts.clone()
        xyzts_right[:, 0:1] += 0.20 
        
        yzts = func(mins=[self.geo_span[1][0], self.geo_span[2][0], self.time_span[0]],
                    maxs=[self.geo_span[1][1], self.geo_span[2][1], self.time_span[1]],
                    num=bc_num//4)
        xmin_yzts = torch.cat([
            torch.full((yzts.shape[0], 1), self.geo_span[0][0], device=yzts.device),
            yzts
        ], dim=1)
        xmax_yzts = torch.cat([
            torch.full((yzts.shape[0], 1), self.geo_span[0][1], device=yzts.device),
            yzts
        ], dim=1)
        xzts = func(mins=[self.geo_span[0][0], self.geo_span[2][0], self.time_span[0]],
                    maxs=[self.geo_span[0][1], self.geo_span[2][1], self.time_span[1]],
                    num=bc_num//4)
        ymin_xzts = torch.cat([
            xzts[:, 0:1],
            torch.full((xzts.shape[0], 1), self.geo_span[1][0], device=xzts.device),
            xzts[:, 1:],
        ], dim=1)
        ymax_xzts = torch.cat([
            xzts[:, 0:1],
            torch.full((xzts.shape[0], 1), self.geo_span[1][1], device=xzts.device),
            xzts[:, 1:],
        ], dim=1)
        
        xyts = func(mins=[self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]],
                    maxs=[self.geo_span[0][1], self.geo_span[1][1], self.time_span[1]],
                    num=bc_num//4)

        zmax_xyts = torch.cat([
            xyts[:, 0:2],
            torch.full((xyts.shape[0], 1), self.geo_span[2][1], device=xyts.device),
            xyts[:, 2:]
        ], dim=1)
        
        xyzts = torch.cat([xyzts_left, xyzts_right,
            xmin_yzts, xmax_yzts, ymin_xzts, ymax_xzts, zmax_xyts], dim=0)

        return xyzts.float().requires_grad_(True)

    def ic_sample(self, ic_num, strategy: str = "lhs", local_area=[[-0.1, 0.1], [0, 0.1]]):
        if strategy == "lhs":

            xyzs = pfp.make_lhs_sampling_data(mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0]],
                                             maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1]],
                                             num=ic_num)

        elif strategy == "grid":
            xyzs = pfp.make_uniform_grid_data(mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0]],
                                             maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1]],
                                             num=ic_num)
        elif strategy == "grid_transition":
            xyzs = pfp.make_uniform_grid_data_transition(mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0]],
                                                        maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1]],
                                                        num=ic_num)

        else:
            raise ValueError(f"Unknown strategy {strategy}")
        
        
        xyzs_local = pfp.make_lhs_sampling_data(mins=[-0.4, -0.2, 0],
                                               maxs=[0.4, 0.2, 0.2,],
                                               num=ic_num*4)
        xyzs = torch.cat([xyzs, xyzs_local], dim=0)
        xyzs = torch.cat([xyzs, torch.full((xyzs.shape[0], 1), self.time_span[0], device=xyzs.device)], dim=1)
        return xyzs.float().requires_grad_(True)



geo_span = eval(config.get("TRAIN", "GEO_SPAN"))
time_span = eval(config.get("TRAIN", "TIME_SPAN"))
sampler = GeoTimeSampler(geo_span, time_span)
net = pfp.PFPINN(
    # sizes=eval(config.get("TRAIN", "NETWORK_SIZE")),
    act=torch.nn.Tanh
)
torch.save(net,'save.pt')

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
    "eps": 1e-4,
    "min_thresh": 0.99,
    "step": 10,
    "mean_thresh": 0.5,
    "max_thresh": 1e-3
}


def ic_func(xts):
    r = torch.sqrt((torch.abs(xts[:, 0:1]) - 0.20)**2 \
        + xts[:, 1:2]**2 + xts[:, 2:3]**2).detach()
    with torch.no_grad():
        phi = 1 - (1 - torch.tanh(torch.sqrt(torch.tensor(OMEGA_PHI)) /
                                  torch.sqrt(2 * torch.tensor(ALPHA_PHI)) * (r-0.10) / GEO_COEF)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE
    return torch.cat([phi, c], dim=1)


def bc_func(xts):
    r = torch.sqrt((torch.abs(xts[:, 0:1]) - 0.20)**2 \
        + xts[:, 1:2]**2 + xts[:, 2:3]**2).detach()
    with torch.no_grad():
        phi = (r > 0.10).float()
        c = phi.detach()
    return torch.cat([phi, c], dim=1)


def split_temporal_coords_into_segments(ts, time_span, num_seg):
    # Split the temporal coordinates into segments
    # Return the indexes of the temporal coordinates
    ts = ts.cpu()
    min_t, max_t = time_span
    # bins = torch.linspace(min_t, max_t**(1/2), num_seg + 1, device=ts.device)**2
    bins = torch.linspace(min_t, max_t, num_seg + 1, device=ts.device)
    indices = torch.bucketize(ts, bins)
    return [torch.where(indices-1 == i)[0] for i in range(num_seg)]



criteria = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.9)

GEOTIME_SHAPE = eval(config.get("TRAIN", "GEOTIME_SHAPE"))
BCDATA_SHAPE = eval(config.get("TRAIN", "BCDATA_SHAPE"))
ICDATA_SHAPE = eval(config.get("TRAIN", "ICDATA_SHAPE"))
SAMPLING_STRATEGY = eval(config.get("TRAIN", "SAMPLING_STRATEGY"))
RAR_BASE_SHAPE = config.getint("TRAIN", "RAR_BASE_SHAPE")
RAR_SHAPE = config.getint("TRAIN", "RAR_SHAPE")

cross_break = 4

for epoch in range(EPOCHS):
    net.train()
    need_causal = True
    # need_causal = epoch < 15000
    if epoch % BREAK_INTERVAL == 0:
        geotime, bcdata, icdata = sampler.resample(GEOTIME_SHAPE, BCDATA_SHAPE,
                                                   ICDATA_SHAPE, strateges=SAMPLING_STRATEGY)
        geotime = geotime.to(net.device)
        # data = geotime.requires_grad_(True)
        residual_base_data = sampler.in_sample(RAR_BASE_SHAPE, strategy="lhs")
        method = config.get("TRAIN", "ADAPTIVE_SAMPLING").strip('"')
        anchors = net.adaptive_sampling(RAR_SHAPE, residual_base_data,
                                        method=method, )

        net.train()
        data = torch.cat([geotime, anchors],
                         dim=0).detach().requires_grad_(True)


        # shuffle
        data = data[torch.randperm(len(data))]
        indices = split_temporal_coords_into_segments(data[:, -1],
                                                    time_span,
                                                    num_seg)

        bcdata = bcdata.to(net.device).detach().requires_grad_(True)
        icdata = icdata.to(net.device).detach().requires_grad_(True)

    
    residual_items = net.net_pde(data, return_dt=True)
    pde_residual = residual_items[0] \
        if epoch % BREAK_INTERVAL < (BREAK_INTERVAL // cross_break) \
        else residual_items[1]

    
    dphi_dt = residual_items[2]
    dc_dt = residual_items[3]
        
    bc_forward = net.net_u(bcdata)
    ic_forward = net.net_u(icdata)
    
    pde_seg_loss = torch.zeros(num_seg, device=net.device)
    for seg_idx, data_idx in enumerate(indices):
        pde_seg_loss[seg_idx] = torch.mean(pde_residual[data_idx]**2)
        
    pde_causal_weight = torch.zeros(num_seg, device=net.device)
    for seg_idx in range(num_seg):
        if seg_idx == 0:
            pde_causal_weight[seg_idx] = 1
        else:
            pde_causal_weight[seg_idx] = torch.exp(
                -causal_configs["eps"] * torch.sum(pde_seg_loss[:seg_idx])
            ).detach()
    
    if pde_causal_weight[-1] > causal_configs["min_thresh"] \
        and causal_configs["eps"] < causal_configs["max_thresh"]:
        causal_configs["eps"] *= causal_configs["step"]
        print(f"epoch {epoch}: "
                f"increase eps to {causal_configs['eps']:.2e}")
        
    if torch.mean(pde_causal_weight) < causal_configs["mean_thresh"]:
        causal_configs["eps"] /= causal_configs["step"]
        print(f"epoch {epoch}: "
                f"decrease eps to {causal_configs['eps']:.2e}")
 
    
    pde_loss = torch.sum(pde_causal_weight * pde_seg_loss)
    # pde_loss = torch.mean(pde_residual**2)
    
    bc_loss = torch.mean((bc_forward - bc_func(bcdata))**2)
    ic_loss = torch.mean((ic_forward - ic_func(icdata))**2)
    dev_loss = (torch.mean(torch.relu(dphi_dt)) + torch.mean(torch.relu(dc_dt))) / 2
    
    
    if epoch % (BREAK_INTERVAL // cross_break) == 0:
        pde_weight, bc_weight, ic_weight, dev_weight = net.compute_gradient_weight(
                [pde_loss, bc_loss, ic_loss, dev_loss],)
    
    losses = pde_weight * pde_loss + dev_weight * dev_loss \
             + bc_weight * bc_loss + ic_weight * ic_loss
        
    if epoch % BREAK_INTERVAL == 0:
        grads = net.gradient(losses)
        writer.add_scalar("grad/grads", grads.abs().mean(), epoch)
        

    opt.zero_grad()
    losses.backward()
    opt.step()
    scheduler.step()


    if epoch % (BREAK_INTERVAL // cross_break) == 0:
        
        print(f"epoch {epoch}: pde_loss {pde_loss:.2e}, "
              f"bc_loss {bc_loss:.2e}, ic_loss {ic_loss:.2e}, "
              f"dev_loss {dev_loss:.2e}"
              f"pde_weight {pde_weight:.2e}, "
              f"bc_weight {bc_weight:.2e}, ic_weight {ic_weight:.2e}, "
              f"dev_weight {dev_weight:.2e}")
        if epoch % BREAK_INTERVAL < BREAK_INTERVAL// cross_break :
            writer.add_scalar("loss/ac_loss", pde_loss, epoch)
            writer.add_scalar("weight/ac_weight", pde_weight, epoch)
        else:
            writer.add_scalar("loss/ch_loss", pde_loss, epoch)
            writer.add_scalar("weight/ch_weight", pde_weight, epoch)
        writer.add_scalar("loss/bc_loss", bc_loss, epoch)
        writer.add_scalar("loss/ic_loss", ic_loss, epoch)
        writer.add_scalar("loss/dev_loss", dev_loss, epoch)
        writer.add_scalar("loss/total", losses, epoch)
        
        writer.add_scalar("weight/bc_weight", bc_weight, epoch)
        writer.add_scalar("weight/ic_weight", ic_weight, epoch)
        writer.add_scalar("weight/dev_weight", dev_weight, epoch)
        
        TARGET_TIMES = eval(config.get("TRAIN", "TARGET_TIMES"))
        REF_PREFIX = config.get("TRAIN", "REF_PREFIX").strip('"')
        
        if epoch % (BREAK_INTERVAL// cross_break) == 0:
              
            # bins = torch.linspace(time_span[0], time_span[1]**(1/2), num_seg + 1, device=net.device)**2
            bins = torch.linspace(time_span[0], time_span[1], num_seg + 1, device=net.device)
            
            ts = (bins[1:] + bins[:-1]) / 2 / TIME_COEF
            ts = ts.detach().cpu().numpy()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            ax = axes[0]
            ax.plot(ts)
            ax.set_title("time segments")
            ax.set_ylabel("time (s)")
            
            
            ax = axes[1]
            if epoch % BREAK_INTERVAL < (BREAK_INTERVAL // cross_break):
                ax.plot(ts, pde_causal_weight.cpu().numpy(), label="ac")
            else:
                ax.plot(ts, pde_causal_weight.cpu().numpy(), label="ch")
            ax.set_title(f"eps: {causal_configs['eps']:.2e}")
            ax.set_ylabel("Causal Weights")
            ax.legend(loc="upper right")

            if epoch % BREAK_INTERVAL < (BREAK_INTERVAL // cross_break):
                ax = axes[2]
                ax.plot(ts, pde_seg_loss.detach().cpu().numpy(), label="ac")
                ax.set_title("AC segment loss")
                ax.set_ylabel("AC segment loss")
            else:
                ax = axes[3]
                ax.plot(ts, pde_seg_loss.detach().cpu().numpy(), label="ch")
                ax.set_title("CH segment loss")
                ax.set_ylabel("CH segment loss")

            # figure title 
            fig.suptitle(f"epoch: {epoch} ")
            # close the figure
            plt.close(fig)
            writer.add_figure("fig/causal_weights", fig, epoch)
            
            
            
            fig, acc = net.plot_3d_geo_predict(ts=TARGET_TIMES,
                                            mesh_points=MESH_POINTS,
                                            ref_prefix=REF_PREFIX,
                                            epoch=epoch,)

            torch.save(net.state_dict(), f"{save_root}/{now}/model-{epoch}.pt")
            writer.add_figure("fig/predict", fig, epoch)
            writer.add_scalar("acc", acc, epoch)
            plt.close(fig)
