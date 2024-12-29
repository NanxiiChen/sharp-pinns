import torch
import numpy as np
import matplotlib.pyplot as plt
import configparser

from matplotlib import gridspec
from .embeddings import SpatialTemporalFourierEmbedding
from .archs import MLP, ModifiedMLP


config = configparser.ConfigParser()
config.read("config.ini")

ALPHA_PHI = config.getfloat("PARAM", "ALPHA_PHI")
OMEGA_PHI = config.getfloat("PARAM", "OMEGA_PHI")
AA = config.getfloat("PARAM", "AA")
MM = config.getfloat("PARAM", "MM")
DD = config.getfloat("PARAM", "DD")
LP = config.getfloat("PARAM", "LP")
CSE = config.getfloat("PARAM", "CSE")
CLE = eval(config.get("PARAM", "CLE"))

TIME_COEF = config.getfloat("TRAIN", "TIME_COEF")
GEO_COEF = config.getfloat("TRAIN", "GEO_COEF")
DIM = config.getint("TRAIN", "DIM")

TIME_SPAN = eval(config.get("TRAIN", "TIME_SPAN"))
GEO_SPAN = eval(config.get("TRAIN", "GEO_SPAN"))
        

class PFEncodedPINN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, arch="modifiedmlp"):
        super().__init__()
        self.model = ModifiedMLP(in_dim, hidden_dim, out_dim, layers)\
            if arch == "modifiedmlp" \
            else MLP(in_dim, hidden_dim, out_dim, layers)
        
    def forward(self, x):
        sol = torch.tanh(self.model(x)) / 2 + 1/2
        phi, cl = torch.split(sol, 1, dim=1)
        cl = cl * (1 - CSE + CLE)
        c = (CSE - CLE) * (-2*phi**3 + 3*phi**2) + cl
        return torch.cat([phi, c], dim=1)


class PFEncodedPINNTwoNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.model_cl = ModifiedMLP(in_dim, hidden_dim//2, out_dim//2, layers)
        self.model_phi = ModifiedMLP(in_dim, hidden_dim//2, out_dim//2, layers)
        
    def forward(self, x):
        phi = torch.tanh(self.model_phi(x)[:, 0:1]) / 2 + 1/2
        cl = torch.tanh(self.model_cl(x)[:, 0:1]) / 2 + 1/2
        cl = cl * (1 - CSE + CLE)
        c = (CSE - CLE) * (-2*phi**3 + 3*phi**2) + cl
        return torch.cat([phi, c], dim=1)

MODELDICT = {
    "mlp": MLP,
    "modifiedmlp": ModifiedMLP
}
        
class PFPINN(torch.nn.Module):
    def __init__(
        self,
        in_dim=256, hidden_dim=200, out_dim=2, layers=6,
        embedding_features=64,
        symmetrical_forward=True,
        arch="modifiedmlp",
        fourier_embedding=True,
        hard_constrain=True
    ):
        super().__init__()
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.embedding_features = embedding_features
        self.embedding = SpatialTemporalFourierEmbedding(DIM+1, embedding_features, scale=2).to(self.device)\
            if fourier_embedding \
            else torch.nn.Linear(DIM+1, embedding_features*4).to(self.device)
        if hard_constrain:
            self.model = PFEncodedPINN(in_dim, hidden_dim, out_dim, layers, arch=arch).to(self.device)
        else:
            self.model = MODELDICT[arch](in_dim, hidden_dim, out_dim, layers, norm=True).to(self.device)
        self.symmetrical_forward = symmetrical_forward


    def auto_grad(self, up, down):
        return torch.autograd.grad(inputs=down, outputs=up,
                                   grad_outputs=torch.ones_like(up),
                                   create_graph=True, retain_graph=True)[0]
    
    def forward(self, x):
        if self.symmetrical_forward:
            x_embedded = self.embedding(x)
            x_neg_embedded = self.embedding(x * torch.tensor([-1,] + [1]*(x.shape[1]-1),  
                                            dtype=x.dtype, device=x.device))
            
            output_pos = self.model(x_embedded)
            output_neg = self.model(x_neg_embedded)
            
            return (output_pos + output_neg) / 2
        else:
            return self.model(self.embedding(x))

    def net_u(self, x):
        # compute the pde solution `u`: [phi, c]
        x = x.to(self.device)
        return self.forward(x)
    
    
    def net_dev(self, x, on:int=0):
        # compute the derivative of the pde solution `u` w.r.t. x: [dphi/dx, dc/dx] or [dphi/dy, dc/dy]
        x = x.to(self.device)
        out = self.forward(x)
        dev_phi = self.auto_grad(out[:, 0:1], x)
        dev_c = self.auto_grad(out[:, 1:2], x)
        return torch.cat([dev_phi[:, on:on+1], dev_c[:, on:on+1]], dim=1)


    def net_pde(self, geotime, return_dt=False):
        # compute the pde residual
        # geo: x/y, t
        # sol: phi, c

        AC1 = 2 * AA * LP / TIME_COEF
        AC2 = LP * OMEGA_PHI / TIME_COEF
        AC3 = LP * ALPHA_PHI * GEO_COEF**2 / TIME_COEF
        CH1 = 2 * AA * MM * GEO_COEF**2 / TIME_COEF

        geotime = geotime.detach().requires_grad_(True).to(self.device)
        sol = self.net_u(geotime)

        dphi_dgeotime = self.auto_grad(sol[:, 0:1], geotime)
        dc_dgeotime = self.auto_grad(sol[:, 1:2], geotime)

        dphi_dt = dphi_dgeotime[:, -1:]
        dc_dt = dc_dgeotime[:, -1:]

        dphi_dgeo = dphi_dgeotime[:, :-1]
        dc_dgeo = dc_dgeotime[:, :-1]

        nabla2phi = torch.zeros_like(dphi_dgeo[:, 0:1])
        for i in range(geotime.shape[1]-1):
            nabla2phi += self.auto_grad(dphi_dgeo[:, i:i+1],
                                        geotime)[:, i:i+1]

        nabla2c = torch.zeros_like(dphi_dgeo[:, 0:1])
        for i in range(geotime.shape[1]-1):
            nabla2c += self.auto_grad(dc_dgeo[:, i:i+1],
                                      geotime)[:, i:i+1]

        h_phi = -2 * sol[:, 0:1]**3 + 3 * sol[:, 0:1]**2
        dh_dphi = -6 * sol[:, 0:1]**2 + 6 * sol[:, 0:1]
        dg_dphi = 4 * sol[:, 0:1]**3 - 6 * sol[:, 0:1]**2 + 2 * sol[:, 0:1]
        nabla2_h_phi = 6 * (
            sol[:, 0:1] * (1 - sol[:, 0:1]) * nabla2phi
            + (1 - 2 * sol[:, 0:1]) *
            torch.sum(dphi_dgeo**2, dim=1, keepdim=True)
        )

        ch = dc_dt - CH1 * nabla2c + CH1 * (CSE - CLE) * nabla2_h_phi
        ac = dphi_dt - AC1 * (sol[:, 1:2] - h_phi*(CSE-CLE) - CLE) * (CSE - CLE) * dh_dphi \
            + AC2 * dg_dphi - AC3 * nabla2phi 

        if return_dt:
            return [ac/1e6, ch, dphi_dt, dc_dt]
        
        return [ac/1e6, ch]
        
    def gradient(self, loss):
        # compute gradient of loss w.r.t. model parameters
        loss.backward(retain_graph=True)
        return torch.cat([g.grad.view(-1) for g in self.model.parameters() if g.grad is not None])

    def adaptive_sampling(self, num, base_data, method):
        # adaptive sampling based on various criteria
        base_data = base_data.to(self.device)
        self.eval()
        if method == "rar":
            ac_residual, ch_residual = self.net_pde(base_data)
            ac_residual = ac_residual.view(-1).detach()
            ch_residual = ch_residual.view(-1).detach()
            _, ac_idx = torch.topk(ac_residual.abs(), num)
            _, ch_idx = torch.topk(ch_residual.abs(), num)

            idxs = torch.cat([ac_idx, ch_idx])
            idxs = torch.unique(torch.cat([ac_idx, ch_idx]))
        elif method == "gar":
            sol = self.net_u(base_data)
            dphi_dgeotime = self.auto_grad(sol[:, 0:1], base_data)
            idxs = []
            for i in range(dphi_dgeotime.shape[1]):
                _, idx = torch.topk(dphi_dgeotime[:, i].abs(), num)
                idxs.append(idx)
            idxs = torch.unique(torch.cat(idxs))
        else:
            raise ValueError("method must be one of 'rar' or 'gar'")
        return base_data[idxs].to(self.device)
