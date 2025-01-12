import torch
import numpy as np
import matplotlib.pyplot as plt
import configparser

from matplotlib import gridspec

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


class Evaluator:
    def __init__(self, pinn):
        self.pinn = pinn

    def plot_predict(self, ref_sol=None, epoch=None, ts=None,
                     mesh_points=None, ref_prefix=None):
        # plot the prediction of the model
        geo_label_suffix = f" [{1/GEO_COEF:.0e}m]"
        time_label_suffix = f" [{1/TIME_COEF:.0e}s]"

        if mesh_points is None:
            geotime = np.vstack([ref_sol["x"].values, ref_sol["t"].values]).T
            geotime = torch.from_numpy(geotime).float().to(self.device)
            sol = self.pinn.net_u(geotime).detach().cpu().numpy()

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
            # use gs to plot the prediction
            fig = plt.figure(figsize=(18, 5*len(ts)))
            gs = gridspec.GridSpec(len(ts), 3, figure=fig, width_ratios=[1, 1, 0.05])
            mesh_tensor = torch.from_numpy(mesh_points).float()
            diffs = []
            for idx, tic in enumerate(ts):
                tic_tensor = torch.ones(mesh_tensor.shape[0], 1)\
                    .view(-1, 1) * tic * TIME_COEF
                geotime = torch.cat([mesh_tensor, tic_tensor],
                                    dim=1).to(self.pinn.device)
                with torch.no_grad():
                    sol = self.pinn.net_u(geotime).detach().cpu().numpy()
                    
                ax = fig.add_subplot(gs[idx, 0])
                ax.scatter(mesh_points[:, 0], mesh_points[:, 1], c=sol[:, 0],
                                     cmap="coolwarm", label="phi", vmin=0, vmax=1)
                ax.set(xlim=GEO_SPAN[0], ylim=GEO_SPAN[1], aspect="equal",
                                 xlabel="x" + geo_label_suffix, ylabel="y" + geo_label_suffix,
                                 title="pred t = " + str(round(tic, 3)))

                truth = np.load(ref_prefix + f"{tic:.3f}" + ".npy")
                diff = np.abs(sol[:, 0] - truth[:, 0])
                
                ax = fig.add_subplot(gs[idx, 1])
                error = ax.scatter(mesh_points[:, 0], mesh_points[:, 1], c=diff,
                                     cmap="coolwarm", label="error")
                ax.set(xlim=GEO_SPAN[0], ylim=GEO_SPAN[1], aspect="equal",
                                 xlabel="x" + geo_label_suffix, ylabel="y" + geo_label_suffix,
                                 title="error t = " + str(round(tic, 3)))
                # add a colorbar to show the scale of the error
                cbar_ax = fig.add_subplot(gs[idx, 2])
                fig.colorbar(error, cax=cbar_ax)

                diffs.append(diff)
            acc = np.mean(np.array(diffs)**2)

        return fig, acc
    
    
    def plot_3d_geo_predict(self, ref_prefix, epoch, ts, mesh_points):
        geo_label_suffix = f" [{1/GEO_COEF:.0e}m]"
        time_label_suffix = f" [{1/TIME_COEF:.0e}s]"
        
        filter_ = np.where(
            (mesh_points[:, 0] >= GEO_SPAN[0][0]) &
            (mesh_points[:, 0] <= GEO_SPAN[0][1]) &
            (mesh_points[:, 1] >= GEO_SPAN[1][0]) &
            (mesh_points[:, 1] <= GEO_SPAN[1][1]) &
            (mesh_points[:, 2] >= GEO_SPAN[2][0]) &
            (mesh_points[:, 2] <= GEO_SPAN[2][1])
        )[0]
        mesh_points = mesh_points[filter_]
        
        fig, axes = plt.subplots(2, len(ts), figsize=(5*len(ts), 10), 
                                 subplot_kw={"projection": "3d",
                                             "xlim": GEO_SPAN[0],
                                             "ylim": GEO_SPAN[1],
                                             "zlim": (GEO_SPAN[2][1], GEO_SPAN[2][0]),
                                             "xlabel": "x" + geo_label_suffix,
                                             "ylabel": "y" + geo_label_suffix,
                                             "zlabel": "z" + geo_label_suffix,
                                             "aspect": "auto",
                                             "box_aspect": (2,2,1),})

        mesh_tensor = torch.from_numpy(mesh_points).float()        
        diffs = []
        for idx, tic in enumerate(ts):
            tic_tensor = torch.ones(mesh_tensor.shape[0], 1)\
                .view(-1, 1) * tic * TIME_COEF
            geotime = torch.cat([mesh_tensor, tic_tensor],
                                dim=1).to(self.pinn.device)
            with torch.no_grad():
                sol = self.pinn.net_u(geotime).detach().cpu().numpy()
                
            
            ax = axes[0, idx]
            idx_interface_sol = np.where((sol[:, 0] > 0.05) & (sol[:, 0] < 0.95))[0]
            ax.scatter(mesh_points[idx_interface_sol, 0], mesh_points[idx_interface_sol, 1],
                       mesh_points[idx_interface_sol, 2], c=sol[idx_interface_sol, 0],
                       cmap="coolwarm", label="phi", vmin=0, vmax=1)
            ax.set_title(f"pred t = {tic:.3f} s\nat epoch {epoch}")
            ax.set_axis_off()
            ax.view_init(elev=30, azim=45)
            
            ax = axes[1, idx]
            truth = np.load(ref_prefix + f"{tic:.3f}" + ".npy")[filter_]
            diff = np.abs(sol[:, 0] - truth[:, 0])
            
            # idx_interface_truth = np.where((truth[:, 0] > 0.05) & (truth[:, 0] < 0.95))[0]
            # ax.scatter(mesh_points[idx_interface_truth, 0], mesh_points[idx_interface_truth, 1],
            #               mesh_points[idx_interface_truth, 2], c=truth[idx_interface_truth, 0],
            #               cmap="coolwarm", label="phi", vmin=0, vmax=1)
            idx_interface_diff = np.where(diff > 0.05)[0]
            ax.scatter(mesh_points[idx_interface_diff, 0], mesh_points[idx_interface_diff, 1],
                          mesh_points[idx_interface_diff, 2], c=diff[idx_interface_diff],
                          cmap="coolwarm", label="error")
            
            ax.set_title(f"error t = {tic:.3f} s\nat epoch {epoch}")
            # colorbar 
            cbar = fig.colorbar(ax.collections[0], ax=ax, orientation="horizontal")
            cbar.set_label("error")
            ax.set_axis_off()
            ax.view_init(elev=30, azim=45)
            diffs.append(np.mean(diff**2))
        acc = np.mean(np.array(diffs))
        return fig, acc
            

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