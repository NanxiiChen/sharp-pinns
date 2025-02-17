import torch
import numpy as np 
import matplotlib.pyplot as plt


class CausalWeighter:
    def __init__(self, num_causal_seg) -> None:
        self.causal_configs = {
                "eps": 1e-4,
                "min_thresh": 0.99,
                "step": 10,
                "mean_thresh": 0.5,
                "max_eps": 1e-3
            }
        self.num_causal_seg = num_causal_seg
    
    
    def compute_causal_weights(self, seg_loss:torch.tensor):
        causal_weights = torch.zeros(self.num_causal_seg, device=seg_loss.device)
        for seg_idx in range(self.num_causal_seg):
            if seg_idx == 0:
                causal_weights[seg_idx] = 1
            else:
                causal_weights[seg_idx] = torch.exp(
                    -self.causal_configs["eps"] * torch.sum(seg_loss[:seg_idx])
                ).detach()
        return causal_weights
                
    def update_causal_configs(self, causal_weights, epoch):
        if causal_weights[-1] > self.causal_configs["min_thresh"] \
            and self.causal_configs["eps"] < self.causal_configs["max_eps"]:
            self.causal_configs["eps"] *= self.causal_configs["step"]
            print(f"Epoch {epoch}: "
                    f"increase eps to {self.causal_configs['eps']:.2e}")
            
        if torch.mean(causal_weights) < self.causal_configs["mean_thresh"]:
            self.causal_configs["eps"] /= self.causal_configs["step"]
            print(f"Epoch {epoch}: "
                    f"decrease eps to {self.causal_configs['eps']:.2e}")
            
            
    def plot_causal_weights(self, seg_loss, causal_weight, pde, epoch, ts):

        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        ax = axes[0]
        ax.plot(ts)
        ax.set_title("time segments")
        ax.set_ylabel("time (s)")
        
        
        ax = axes[1]
        if pde == "ac":
            ax.plot(ts, causal_weight.cpu().numpy(), label="ac")
        else:
            ax.plot(ts, causal_weight.cpu().numpy(), label="ch")
        ax.set_title(f"eps: {self.causal_configs['eps']:.2e}")
        ax.set_ylabel("Causal Weights")
        ax.legend(loc="upper right")

        if pde == "ac":
            ax = axes[2]
            ax.plot(ts, seg_loss.detach().cpu().numpy(), label="ac")
            ax.set_title("AC segment loss")
            ax.set_ylabel("AC segment loss")
        else:
            ax = axes[3]
            ax.plot(ts, seg_loss.detach().cpu().numpy(), label="ch")
            ax.set_title("CH segment loss")
            ax.set_ylabel("CH segment loss")

        fig.suptitle(f"epoch: {epoch} ")
        return fig