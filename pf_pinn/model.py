import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
# from allen_cahn.sampler import GeoTimeSampler
import time
import configparser

# from .efficient_kan import KAN


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


class FourierEmbedding(torch.nn.Module):
    def __init__(self, in_features, embedding_features, std=1, method="trig"):
        super().__init__()
        self.method = method
        self.linear = torch.nn.Linear(in_features, embedding_features)
        if self.method == "trig":
            self.linear.weight.data = \
                torch.randn(embedding_features, in_features) * std * np.pi
            self.linear.bias.data.zero_()
        elif self.method == "linear":
            torch.nn.init.xavier_normal_(self.linear.weight)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.linear(x)
        method = self.method
        if method == "trig":
            return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        elif method == "linear":
            return x
        else:
            raise ValueError("Ivalid method.")


class SpatialTemporalFourierEmbedding(torch.nn.Module):

    def __init__(self, in_features, embedding_features, std=2):
        super().__init__()
        self.spatial_embedding = FourierEmbedding(in_features-1,
                                                  embedding_features, std)
        self.temporal_embedding = FourierEmbedding(1, embedding_features, std)

    def forward(self, x):
        y_spatial = self.spatial_embedding(x[:, :-1])
        y_temporal = self.temporal_embedding(x[:, -1:])
        return torch.cat([y_spatial, y_temporal], dim=1)


class MultiScaleFourierEmbedding(torch.nn.Module):

    def __init__(self, in_features, embedding_features=8, std=1):
        super().__init__()
        self.spatial_low_embedding = FourierEmbedding(
            in_features-1,
            embedding_features,
            std/10
        )
        self.spatial_high_embedding = FourierEmbedding(
            in_features-1,
            embedding_features,
            std*10
        )
        self.temporal_embedding = FourierEmbedding(
            1, embedding_features,
            std*5,
        )

    def forward(self, x):
        y_low = self.spatial_low_embedding(x[:, :-1])
        y_high = self.spatial_high_embedding(x[:, :-1])
        y_temporal = self.temporal_embedding(x[:, -1:])
        return torch.cat([y_low, y_high, y_temporal], dim=1)
    
    
class ModifiedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.gate_layer_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.gate_layer_2 = torch.nn.Linear(in_dim, hidden_dim)
        
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_dim if idx == 0 else hidden_dim, hidden_dim) for idx in range(layers)
        ])
        
        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)

        self.act = torch.nn.Tanh()
        
    def forward(self, x):
        u = self.act(self.gate_layer_1(x))
        v = self.act(self.gate_layer_2(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
            x = x * u + (1 - x) * v
        return self.out_layer(x)
    
    
# if __name__ == "__main__":
#     model = ModifiedMLP(3, 32, 2, 4, alpha=0.8)
#     x = torch.randn(10, 3)
#     y = model(x)
#     print(y)
        




# class FourierEmbedding(torch.nn.Module):
#     def __init__(self, input_dim, embedding_dim, ):
#         super().__init__()
#         self.omega = torch.arange(1, embedding_dim+1, device="cuda").float().view(1, -1) * np.pi / 10
#         self.omega_mat = torch.cat([self.omega] * input_dim, dim=0).requires_grad_(False)


#     def forward(self, geotime):
#         # geotime: [x, y, t]
#         # only on x, y
#         return torch.cat([
#             torch.cos(geotime[:, :-1] @ self.omega_mat),
#             torch.sin(geotime[:, :-1] @ self.omega_mat),
#             geotime[:, -1:] @ self.omega
#         ], dim=1)


class PFPINN(torch.nn.Module):
    def __init__(
        self,
        # sizes: list,
        act=torch.nn.Tanh,
        embedding_features=20,
    ):
        super().__init__()
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        # self.sizes = sizes
        self.act = act
        self.embedding_features = embedding_features
        # self.model = torch.nn.Sequential(self.make_layers()).to(self.device)
        self.model = self.make_modified_mlp_layers().to(self.device)
        

        # self.embedding = FourierEmbedding(DIM, embedding_features)
        # self.embedding = MultiScaleFourierEmbedding(DIM+1, embedding_features).to(self.device)
        # self.spatial_embedding = FourierEmbedding(DIM, embedding_features, std=1, method="trig").to(self.device)
        # self.temporal_embedding = FourierEmbedding(1, embedding_features, std=1, method="trig").to(self.device)
        # self.out_layer = torch.nn.Linear(sizes[-1], 2).to(self.device)
        # self.out_layer = torch.nn.Linear(self.sizes[-1], 2).to(self.device)

    def auto_grad(self, up, down):
        return torch.autograd.grad(inputs=down, outputs=up,
                                   grad_outputs=torch.ones_like(up),
                                   create_graph=True, retain_graph=True)[0]

    def make_layers(self):
        layers = []
        for i in range(len(self.sizes) - 1):

            linear_layer = torch.nn.Linear(self.sizes[i], self.sizes[i + 1])
            torch.nn.init.xavier_normal_(linear_layer.weight)
            layers.append((f"linear{i}", linear_layer))
            if i != len(self.sizes) - 2:
                layers.append((f"act{i}", self.act()))
        return OrderedDict(layers)
    
    # def make_kan_layers(self):
    #     kan_layer = KAN([2, 32, 32, 32, 2])
    #     return kan_layer
    
    def make_modified_mlp_layers(self):
        modified_mlp = ModifiedMLP(3, 64, 2, 4)
        return modified_mlp

    def forward(self, x):
        # y_spatial = self.spatial_embedding(x[:, :-1])
        # y_temporal = self.temporal_embedding(x[:, -1:])
        # out_spatial = self.model(y_spatial)
        # out_temporal = self.model(y_temporal)
        # # merge by pointwise multiplication
        # out = self.out_layer(out_spatial * out_temporal)
        # return out
        # x = self.embedding(x)
        return self.model(x)

    def net_u(self, x):
        # compute the pde solution `u`: [phi, c]
        x = x.to(self.device)
        return self.forward(x)

    def net_dev(self, x, on="y"):
        # compute the derivative of the pde solution `u` w.r.t. x: [dphi/dx, dc/dx] or [dphi/dy, dc/dy]
        x = x.to(self.device)
        out = self.forward(x)
        dev_phi = self.auto_grad(out[:, 0:1], x)
        dev_c = self.auto_grad(out[:, 1:2], x)
        if on == "y":
            return torch.cat([dev_phi[:, 1:2], dev_c[:, 1:2]], dim=1)
        elif on == "x":
            return torch.cat([dev_phi[:, 0:1], dev_c[:, 0:1]], dim=1)

    # def net_pde(self, geotime):
    #     # compute the pde residual
    #     # geo: x, y, t
    #     # sol: phi, c
    #     geotime = geotime.detach().requires_grad_(True).to(self.device)
    #     sol = self.net_u(geotime)

    #     dphi_dgeotime = self.auto_grad(sol[:, 0:1], geotime)
    #     dc_dgeotime = self.auto_grad(sol[:, 1:2], geotime)

    #     dphi_dt = dphi_dgeotime[:, -1:] * TIME_COEF
    #     dc_dt = dc_dgeotime[:, -1:] * TIME_COEF

    #     dphi_dgeo = dphi_dgeotime[:, :-1] * GEO_COEF
    #     dc_dgeo = dc_dgeotime[:, :-1] * GEO_COEF

    #     nabla2phi = torch.zeros_like(dphi_dgeo[:, 0:1])
    #     for i in range(geotime.shape[1]-1):
    #         nabla2phi += self.auto_grad(dphi_dgeo[:, i:i+1],
    #                                     geotime)[:, i:i+1] * GEO_COEF

    #     nabla2c = torch.zeros_like(dphi_dgeo[:, 0:1])
    #     for i in range(geotime.shape[1]-1):
    #         nabla2c += self.auto_grad(dc_dgeo[:, i:i+1],
    #                                   geotime)[:, i:i+1] * GEO_COEF

    #     df_dphi = 12 * AA * (CSE - CLE) * sol[:, 0:1] * (sol[:, 0:1] - 1) * \
    #         (sol[:, 1:2] - (CSE - CLE) * (-2 * sol[:, 0:1]**3 + 3 * sol[:, 0:1]**2) - CLE) \
    #         + 2*OMEGA_PHI*sol[:, 0:1]*(sol[:, 0:1] - 1)*(2 * sol[:, 0:1] - 1)

    #     nabla2_df_dc = 2 * AA * (
    #         nabla2c
    #         + 6 * (CSE - CLE) * (
    #             sol[:, 0:1] * (sol[:, 0:1] - 1) * nabla2phi
    #             + (2*sol[:, 0:1] - 1) *
    #             torch.sum(dphi_dgeo**2, dim=1, keepdim=True)
    #         )
    #     )

    #     ac = dphi_dt + LP * (df_dphi - ALPHA_PHI * nabla2phi)
    #     ch = dc_dt - DD / 2 / AA * nabla2_df_dc

    #     return [ac, ch]

    def net_pde(self, geotime):
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
        # g_phi = sol[:, 0:1]**2 * (sol[:, 0:1] - 1)**2
        dh_dphi = -6 * sol[:, 0:1]**2 + 6 * sol[:, 0:1]
        # d2h_dphi2 = -12 * sol[:, 0:1] + 6
        dg_dphi = 4 * sol[:, 0:1]**3 - 6 * sol[:, 0:1]**2 + 2 * sol[:, 0:1]
        # nabla_h_phi = dh_dphi * dphi_dgeo
        # nabla2_h_phi = dh_dphi * nabla2phi + d2h_dphi2 * torch.sum(dphi_dgeo**2, dim=1, keepdim=True)
        nabla2_h_phi = 6 * (
            sol[:, 0:1] * (1 - sol[:, 0:1]) * nabla2phi
            + (1 - 2 * sol[:, 0:1]) *
            torch.sum(dphi_dgeo**2, dim=1, keepdim=True)
        )

        ch = dc_dt - CH1 * nabla2c + CH1 * (CSE - CLE) * nabla2_h_phi
        ac = dphi_dt - AC1 * (sol[:, 1:2] - h_phi*(CSE-CLE) - CLE) * (CSE - CLE) * dh_dphi \
            + AC2 * dg_dphi - AC3 * nabla2phi 

        return [ac/1e6, ch*1e3]
        # return [ac, ch]

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

            # ac_anchors = base_data[ac_idx].to(self.device)
            # ch_anchors = base_data[ch_idx].to(self.device)
            # return [ac_anchors, ch_anchors]
        elif method == "gar":
            sol = self.net_u(base_data)
            dphi_dgeotime = self.auto_grad(sol[:, 0:1], base_data)
            idxs = []
            for i in range(dphi_dgeotime.shape[1]):
                _, idx = torch.topk(dphi_dgeotime[:, i].abs(), num)
                idxs.append(idx)
                # anchors.append(base_data[idx].to(self.device))
            # return anchors
            idxs = torch.unique(torch.cat(idxs))
        else:
            raise ValueError("method must be one of 'rar' or 'gar'")
        return base_data[idxs].to(self.device)

    def compute_jacobian(self, output, mini_batch=False):
        params = [p for p in list(self.parameters())[:-1] if p.requires_grad]
        output = output.reshape(-1)

        if not mini_batch:
            grads = torch.autograd.grad(output, params,
                                        (torch.eye(output.shape[0])
                                            .to(self.device),),
                                        is_grads_batched=True, retain_graph=True)

            return torch.cat([grad.flatten().reshape(len(output), -1) for grad in grads], 1)
        else:
            batch_size = 100  # Adjust this value to fit your GPU memory
            grads = []
            for i in range(0, output.shape[0], batch_size):
                output_batch = output[i:min(i + batch_size, output.shape[0])]
                grad_batch = torch.autograd.grad(output_batch, params,
                                                 (torch.eye(output_batch.shape[0])
                                                  .to(self.device),),
                                                 is_grads_batched=True, retain_graph=True)
                grads.append(torch.cat([grad.flatten().reshape(
                    len(output_batch), -1) for grad in grad_batch], 1))
            return torch.cat(grads)

    # def compute_jacobian(self, output):
    #     output = output.reshape(-1)

    #     grads = torch.autograd.grad(output, self.params,
    #                                 grad_outputs=torch.ones_like(output),
    #                                 create_graph=True, retain_graph=True)

    #     return torch.cat([grad.flatten().reshape(len(output), -1) for grad in grads], 1)

    def compute_ntk(self, jac, compute='trace'):
        if compute == 'full':
            return torch.einsum('Na,Ma->NM', jac, jac)
        elif compute == 'diag':
            return torch.einsum('Na,Na->N', jac, jac)
        elif compute == 'trace':
            return torch.einsum('Na,Na->', jac, jac)
        else:
            raise ValueError('compute must be one of "full",'
                             + '"diag", or "trace"')

    def plot_predict(self, ref_sol=None, epoch=None, ts=None,
                     mesh_points=None, ref_prefix=None):
        # plot the prediction of the model
        geo_label_suffix = f" [{1/GEO_COEF:.0e}m]"
        time_label_suffix = f" [{1/TIME_COEF:.0e}s]"

        if mesh_points is None:
            geotime = np.vstack([ref_sol["x"].values, ref_sol["t"].values]).T
            geotime = torch.from_numpy(geotime).float().to(self.device)
            sol = self.net_u(geotime).detach().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()
            axes[0].scatter(ref_sol["x"], ref_sol["t"],
                            c=sol[:, 0], cmap="coolwarm", label="phi",
                            vmin=0, vmax=1)
            axes[0].set(xlim=GEO_SPAN, ylim=TIME_SPAN,
                        xlabel="x" + geo_label_suffix, ylabel="t" + time_label_suffix)

            diff = np.abs(sol[:, 0] - ref_sol["phi"].values)
            axes[1].scatter(ref_sol["x"], ref_sol["t"], c=diff, cmap="coolwarm",
                            vmin=0, vmax=1)
            axes[1].set(xlim=GEO_SPAN, ylim=TIME_SPAN,
                        xlabel="x" + geo_label_suffix, ylabel="t" + time_label_suffix, )

            axes[0].set_title(r"Solution $\hat\phi$"
                              + f" at epoch {epoch}")
            axes[1].set_title(r"Error $|\hat \phi - \phi_{ref}|$"
                              + f" at epoch {epoch}")
            # fig.legend()

            acc = 1 - np.sqrt(np.mean(diff**2))

        else:

            fig, axes = plt.subplots(len(ts), 2, figsize=(15, 5*len(ts)))
            diffs = []
            for idx, tic in enumerate(ts):
                tic_tensor = torch.ones(mesh_points.shape[0], 1)\
                    .view(-1, 1) * tic * TIME_COEF
                mesh_tensor = torch.from_numpy(mesh_points).float()
                geotime = torch.cat([mesh_tensor, tic_tensor],
                                    dim=1).to(self.device)
                with torch.no_grad():
                    sol = self.net_u(geotime).detach().cpu().numpy()
                axes[idx, 0].scatter(mesh_points[:, 0], mesh_points[:, 1], c=sol[:, 0],
                                     cmap="coolwarm", label="phi", vmin=0, vmax=1)
                axes[idx, 0].set(xlim=GEO_SPAN[0], ylim=GEO_SPAN[1], aspect="equal",
                                 xlabel="x" + geo_label_suffix, ylabel="y" + geo_label_suffix,
                                 title="pred t = " + str(round(tic, 2)))

                truth = np.load(ref_prefix + f"{tic:.2f}" + ".npy")
                diff = np.abs(sol[:, 0] - truth[:, 0])
                axes[idx, 1].scatter(mesh_points[:, 0], mesh_points[:, 1], c=diff,
                                     cmap="coolwarm", label="error", vmin=0, vmax=1)
                axes[idx, 1].set(xlim=GEO_SPAN[0], ylim=GEO_SPAN[1], aspect="equal",
                                 xlabel="x" + geo_label_suffix, ylabel="y" + geo_label_suffix,
                                 title="error t = " + str(round(tic, 2)))
                diffs.append(diff)
            acc = 1 - np.sqrt(np.mean(np.array(diffs)**2))

        return fig, axes, acc

    def plot_samplings(self, geotime, bcdata, icdata, anchors):
        # plot the sampling points
        geotime = geotime.detach().cpu().numpy()
        bcdata = bcdata.detach().cpu().numpy()
        icdata = icdata.detach().cpu().numpy()
        anchors = anchors.detach().cpu().numpy()

        geo_label_suffix = f" [{1/GEO_COEF:.0e}m]"
        time_label_suffix = f" [{1/TIME_COEF:.0e}s]"

        if geotime.shape[1] == 2:

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(geotime[:, 0], geotime[:, 1],
                       c="blue", label="collcations")
            ax.scatter(anchors[:, 0], anchors[:, 1],
                       c="green", label="anchors", marker="x")
            ax.scatter(bcdata[:, 0], bcdata[:, 1],
                       c="red", label="boundary condition")
            ax.scatter(icdata[:, 0], icdata[:, 1],
                       c="orange", label="initial condition")
            ax.set(xlim=GEO_SPAN, ylim=TIME_SPAN,
                   xlabel="x" + geo_label_suffix, ylabel="t" + time_label_suffix,)
            ax.legend(bbox_to_anchor=(1.02, 1.00),
                      loc='upper left', borderaxespad=0.)
        elif geotime.shape[1] == 3:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5),
                                   subplot_kw={"aspect": "auto",
                                               "xlim": GEO_SPAN[0],
                                               "ylim": GEO_SPAN[1],
                                               "zlim": TIME_SPAN,
                                               "xlabel": "x" + geo_label_suffix,
                                               "ylabel": "y" + geo_label_suffix,
                                               "zlabel": "t" + time_label_suffix,
                                               "projection": "3d"})
            ax.scatter(geotime[:, 0], geotime[:, 1], geotime[:, 2],
                       c="blue", label="collcations", alpha=0.1, s=1)
            ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2],
                       c="black", label="anchors", s=1, marker="x", alpha=0.5)
            ax.scatter(bcdata[:, 0], bcdata[:, 1], bcdata[:, 2],
                       c="red", label="boundary condition", s=1, marker="x", alpha=0.5)
            ax.scatter(icdata[:, 0], icdata[:, 1], icdata[:, 2],
                       c="orange", label="initial condition", s=1, marker="x", alpha=0.5)
            ax.legend(bbox_to_anchor=(1.02, 1.00),
                      loc='upper left', borderaxespad=0.)

        else:
            raise ValueError("Only 2 or 3 dimensional data is supported")
        return fig, ax

    def compute_ntk_diag(self, residuals, batch_size):
        diags = []
        for res in residuals:
            if batch_size < len(res):
                jac = self.compute_jacobian(res[
                    np.random.randint(0, len(res), batch_size)
                ])
                diag = self.compute_ntk(jac, compute='diag')
            else:
                jac = self.compute_jacobian(res)
                diag = self.compute_ntk(jac, compute='diag')
            diags.append(diag)
        return diags

    def compute_ntk_weight(self, residuals, method, batch_size, return_ntk_info=False):
        # compute the weight of each loss term using ntk-based method
        traces = []
        jacs = []
        for res in residuals:
            if method == "random":  # random-batch technique
                if batch_size < len(res):
                    jac = self.compute_jacobian(res[
                        np.random.randint(0, len(res), batch_size)
                    ])
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / batch_size)
                else:
                    jac = self.compute_jacobian(res)
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / len(res))

            elif method == "topres":  # compute the weight using the points with top residuals
                if batch_size < len(res):
                    jac = self.compute_jacobian(res[:batch_size])
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / batch_size)
                else:
                    jac = self.compute_jacobian(res)
                    trace = self.compute_ntk(jac, compute='trace').item()
                    traces.append(trace / len(res))

            elif method == "mini":  # compute the weight using mini-batch technique
                trace = 0
                for i in range(0, len(res), batch_size):
                    jac = self.compute_jacobian(res[
                        i: min(i + batch_size, len(res))
                    ])
                    trace += self.compute_ntk(jac, compute='trace').item()
                traces.append(trace / len(res))
            elif method == "full":
                jac = self.compute_jacobian(res)
                trace = self.compute_ntk(jac, compute='trace').item()
                traces.append(trace / len(res))

            else:
                raise ValueError("method must be one of 'random', 'topres'"
                                 " 'mini', or 'full'")

            if return_ntk_info:
                jacs.append(jac)

        traces = np.array(traces)
        if return_ntk_info:
            return traces.sum() / traces, jacs, traces
        # return traces.sum() / traces / np.sqrt(np.sum(traces**2) * len(traces))
        return traces.sum() / traces

    def compute_gradient_weight(self, losses):

        grads = np.zeros(len(losses))

        for idx, loss in enumerate(losses):
            # zero_grad
            self.zero_grad()
            grad = self.gradient(loss)
            grads[idx] = torch.sqrt(torch.sum(grad**2)).item()
            # grads[idx] = torch.mean(torch.abs(grad)).item()
            # grads[idx] = torch.max(torch.abs(grad)).item()

        return np.sum(grads) / grads


# class CausalWeightor:
#     def split_temporal_coords_into_segments(self,
#                                             time_coords: torch.Tensor,
#                                             num_segments: int = 10,
#                                             time_span: tuple[float, float] = TIME_SPAN,) -> torch.Tensor:
#         min_time, max_time = time_span
#         bins = torch.linspace(min_time, max_time,
#                               num_segments+1, device=time_coords.device)
#         indices = torch.searchsorted(bins, time_coords)

#         return indices

#     def compute_causal_weight(self, )

# if __name__ == "__main__":
#     causal_weightor = CausalWeightor()
#     time_coords = torch.rand(100, )
#     segments = causal_weightor.split_temporal_coords_into_segments(
#         time_coords, time_span=(0, 1))
#     print(segments)


# if __name__ == "__main__":
#     model = PFPINN()
#     x = torch.randn(10, 3)
#     y = model(x)
#     print(y)
#     # print(model)