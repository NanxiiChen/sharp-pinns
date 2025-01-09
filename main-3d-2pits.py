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

now = "3d-2pits-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_root = "/root/tf-logs"
writer = SummaryWriter(log_dir=f"{save_root}/" + now)



class GeoTimeSampler:
    def __init__(
        self,
        geo_span: list,
        time_span: list,
    ):
        self.geo_span = geo_span
        self.time_span = time_span

    def resample(self, in_num, bc_num, ic_num):
        return self.in_sample(in_num), \
            self.bc_sample(bc_num), \
            self.ic_sample(ic_num)

    def in_sample(self, in_num, method="grid_transition"):
        if method == "lhs":
            geotime = pfp.make_lhs_sampling_data(
                mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0], self.time_span[0]],
                maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1], self.time_span[1]],
                num=in_num)
        elif method == "grid_transition":
            geotime = pfp.make_uniform_grid_data_transition(
                mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0], self.time_span[0]],
                maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1], self.time_span[1]],
                num=in_num)

        return geotime.float().requires_grad_(True)


    def bc_sample(self, bc_num: int):

        xyzts = pfp.make_lhs_sampling_data(mins=[-0.04, -0.04, 0, self.time_span[0]+self.time_span[1]*0.05],
                                          maxs=[0.04, 0.04, 0.04,self.time_span[1]],
                                          num=bc_num)
        xyzts = xyzts[xyzts[:, 0] ** 2 + xyzts[:, 1] ** 2 + xyzts[:, 2] ** 2 <= 0.02 ** 2]
        
        xyzts_left = xyzts.clone()
        xyzts_left[:, 0:1] -= 0.20
        xyzts_right = xyzts.clone()
        xyzts_right[:, 0:1] += 0.20 
        
        yzts = pfp.make_lhs_sampling_data(
            mins=[self.geo_span[1][0], self.geo_span[2][0], self.time_span[0]],
            maxs=[self.geo_span[1][1], self.geo_span[2][1], self.time_span[1]],
            num=bc_num//5)
        xmin_yzts = torch.cat([
            torch.full((yzts.shape[0], 1), self.geo_span[0][0], device=yzts.device),
            yzts
        ], dim=1)
        xmax_yzts = torch.cat([
            torch.full((yzts.shape[0], 1), self.geo_span[0][1], device=yzts.device),
            yzts
        ], dim=1)
        xzts = pfp.make_lhs_sampling_data(
            mins=[self.geo_span[0][0], self.geo_span[2][0], self.time_span[0]],
            maxs=[self.geo_span[0][1], self.geo_span[2][1], self.time_span[1]],
            num=bc_num//5)
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
        
        xyts = pfp.make_lhs_sampling_data(
            mins=[self.geo_span[0][0], self.geo_span[1][0], self.time_span[0]],
            maxs=[self.geo_span[0][1], self.geo_span[1][1], self.time_span[1]],
            num=bc_num//5)

        zmax_xyts = torch.cat([
            xyts[:, 0:2],
            torch.full((xyts.shape[0], 1), self.geo_span[2][1], device=xyts.device),
            xyts[:, 2:]
        ], dim=1)
        
        xyzts = torch.cat([xyzts_left, xyzts_right,
            xmin_yzts, xmax_yzts, ymin_xzts, ymax_xzts, zmax_xyts], dim=0)

        return xyzts.float().requires_grad_(True)



    def ic_sample(self, ic_num):
        xyzs = pfp.make_lhs_sampling_data(
            mins=[self.geo_span[0][0], self.geo_span[1][0], self.geo_span[2][0]],
            maxs=[self.geo_span[0][1], self.geo_span[1][1], self.geo_span[2][1]],
            num=ic_num)

        xyzs_local = pfp.make_lhs_sampling_data(
            mins=[-0.4, -0.2, 0],
            maxs=[0.4, 0.2, 0.2,],
            num=ic_num*4)
        
        xyzs = torch.cat([xyzs, xyzs_local], dim=0)
        xyzs = torch.cat([xyzs, torch.full((xyzs.shape[0], 1), self.time_span[0], device=xyzs.device)], dim=1)
        return xyzs.float().requires_grad_(True)


geo_span = eval(config.get("TRAIN", "GEO_SPAN"))
time_span = eval(config.get("TRAIN", "TIME_SPAN"))
num_causal_seg = config.getint("TRAIN", "NUM_CAUSAL_SEG")
causal = eval(config.get("TRAIN", "CAUSAL_WEIGHTING"))
fourier_embedding = eval(config.get("TRAIN", "FOURIER_EMBEDDING"))
sampler = GeoTimeSampler(geo_span, time_span)
net = pfp.PFPINN(
    in_dim=config.getint("TRAIN", "IN_DIM"),
    out_dim=config.getint("TRAIN", "OUT_DIM"),
    hidden_dim=config.getint("TRAIN", "HIDDEN_DIM"),
    layers=config.getint("TRAIN", "LAYERS"),
    symmetrical_forward=eval(config.get("TRAIN", "SYMMETRIC")),
    arch=config.get("TRAIN", "ARCH").strip('"'),
    fourier_embedding=fourier_embedding,
    hard_constrain=eval(config.get("TRAIN", "HARD_CONSTRAIN")),
)
evaluator = pfp.Evaluator(net)
loss_manager = pfp.LossManager(writer, net)
if causal:
    causal_weighter = pfp.CausalWeighter(num_causal_seg)

resume = config.get("TRAIN", "RESUME").strip('"')
try:
    net.load_state_dict(torch.load(resume))
    print("Load model successfully")
except:
    print("Can not load model")
    pass


TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")

BREAK_INTERVAL = config.getint("TRAIN", "BREAK_INTERVAL")
EPOCHS = config.getint("TRAIN", "EPOCHS")
LR = config.getfloat("TRAIN", "LR")

ALPHA_PHI = config.getfloat("PARAM", "ALPHA_PHI")
OMEGA_PHI = config.getfloat("PARAM", "OMEGA_PHI")
DD = config.getfloat("PARAM", "DD")
AA = config.getfloat("PARAM", "AA")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))
MESH_POINTS = np.load(config.get("TRAIN", "MESH_POINTS").strip('"')) * GEO_COEF


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
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.9)

GEOTIME_SHAPE = eval(config.get("TRAIN", "GEOTIME_SHAPE"))
BCDATA_SHAPE = eval(config.get("TRAIN", "BCDATA_SHAPE"))
ICDATA_SHAPE = eval(config.get("TRAIN", "ICDATA_SHAPE"))
RAR_BASE_SHAPE = config.getint("TRAIN", "RAR_BASE_SHAPE")
RAR_SHAPE = config.getint("TRAIN", "RAR_SHAPE")



for epoch in range(EPOCHS):
    net.train()
    pde = "ac" if epoch % BREAK_INTERVAL < (BREAK_INTERVAL // 2) else "ch"

    if epoch % BREAK_INTERVAL == 0:
        geotime, bcdata, icdata = sampler.resample(GEOTIME_SHAPE, BCDATA_SHAPE,
                                                   ICDATA_SHAPE)
        geotime = geotime.to(net.device)
        residual_base_data = sampler.in_sample(RAR_BASE_SHAPE, method="lhs")
        anchors = net.adaptive_sampling(RAR_SHAPE, residual_base_data,
                                        method="rar", )

        data = torch.cat([geotime, anchors],
                         dim=0).detach().requires_grad_(True)

        data = data[torch.randperm(len(data))]
        indices = split_temporal_coords_into_segments(data[:, -1],
                                                    time_span,
                                                    num_causal_seg)

        bcdata = bcdata.to(net.device).detach().requires_grad_(True)
        icdata = icdata.to(net.device).detach().requires_grad_(True)
    
    
    residual_items = net.net_pde(data, return_dt=True)
    pde_residual = residual_items[0] \
        if pde == "ac" \
        else residual_items[1]

    
    dphi_dt = residual_items[2]
    dc_dt = residual_items[3]
        
    bc_forward = net.net_u(bcdata)
    ic_forward = net.net_u(icdata)
    
    if causal:
        pde_seg_loss = torch.zeros(num_causal_seg, device=net.device)
        for seg_idx, data_idx in enumerate(indices):
            pde_seg_loss[seg_idx] = torch.mean(pde_residual[data_idx]**2)
        pde_causal_weight = causal_weighter.compute_causal_weights(pde_seg_loss)
        causal_weighter.update_causal_configs(pde_causal_weight, epoch)
        pde_loss = torch.sum(pde_causal_weight * pde_seg_loss)
    else:
        pde_loss = torch.mean(pde_residual**2)
        
    bc_loss = torch.mean((bc_forward - bc_func(bcdata))**2)
    ic_loss = torch.mean((ic_forward - ic_func(icdata))**2)
    irr_loss = torch.mean(torch.relu(dphi_dt)) + torch.mean(torch.relu(dc_dt))
    
    pde_loss_name = f"{pde}_loss"
    loss_manager.register_loss([pde_loss_name, "bc_loss", "ic_loss", "irr_loss",],
                                [pde_loss, bc_loss, ic_loss, irr_loss,])
    
    if epoch % (BREAK_INTERVAL // 2) == 0:
        loss_manager.update_weights()
        
        loss_manager.write_loss(epoch)
        loss_manager.write_weight(epoch)
        loss_manager.print_loss(epoch)
        loss_manager.print_weight(epoch)
        
    losses = loss_manager.weighted_loss()

    opt.zero_grad()
    losses.backward()
    opt.step()
    scheduler.step()

    if epoch % (BREAK_INTERVAL // 2) == 0:
        
        TARGET_TIMES = eval(config.get("TRAIN", "TARGET_TIMES"))
        REF_PREFIX = config.get("TRAIN", "REF_PREFIX").strip('"')
              

        if causal:
            bins = np.linspace(time_span[0], time_span[1], num_causal_seg + 1)
            ts = (bins[1:] + bins[:-1]) / 2 / TIME_COEF
            fig = causal_weighter.plot_causal_weights(pde_seg_loss, pde_causal_weight,
                                                    pde, epoch, ts)
            writer.add_figure("fig/causal_weights", fig, epoch)
        
        
        fig, acc = evaluator.plot_3d_geo_predict(ts=TARGET_TIMES,
                                            mesh_points=MESH_POINTS,
                                            ref_prefix=REF_PREFIX,
                                            epoch=epoch,)

        torch.save(net.state_dict(), f"{save_root}/{now}/model-{epoch}.pt")
        writer.add_figure("fig/predict", fig, epoch)
        writer.add_scalar("acc", acc, epoch)
        plt.close(fig)
        writer.flush()
       
