import torch 
import numpy as np


class LossManager:
    def __init__(self, writer, pinn:torch.nn.Module):
        self.writer = writer
        self.pinn = pinn
        self.loss_panel = {}
        self.weight_panel = {}
        self.loss_format = "{:.2e}"
        self.alpha = 0.0
        
    def register_loss(self, names, losses):
        self.loss_panel = {}
        for name, loss in zip(names, losses):
            self.loss_panel[name] = loss
            
        
    def update_weights(self):
        grads = np.zeros(len(self.loss_panel))
        # for i, loss in enumerate(self.loss_panel.values()):
        for idx, (name, loss) in enumerate(self.loss_panel.items()):
            self.pinn.zero_grad()
            grad = self.pinn.gradient(loss)
            grads[idx] = torch.sqrt(torch.sum(grad**2)).item()
            
        grads = np.clip(grads, 1e-8, 1e8)
        weights = np.mean(grads) / grads
        weights = np.clip(weights, 1e-8, 1e8)
        self.weight_panel = {}
        for i, name in enumerate(self.loss_panel.keys()):
            self.weight_panel[name] = weights[i]
            
    
        # for i, name in enumerate(self.loss_panel.keys()):
        #     if self.weight_panel.get(name) is None:
        #         self.weight_panel[name] = weights[i]
        #     else:
        #         # moving average
        #         self.weight_panel[name] = self.alpha * self.weight_panel[name] \
        #                                     + (1-self.alpha) * weights[i]
                
        # return weights
    
    def weighted_loss(self):
        losses = torch.zeros(1, device=self.pinn.device)
        for name, loss in self.loss_panel.items():
            losses += self.weight_panel[name] * loss
        return losses
        
    
    def write_loss(self, epoch,):
        for name, loss in self.loss_panel.items():
            self.writer.add_scalar(f"loss/{name}", loss.item(), epoch)
        self.writer.flush()
        
    def write_weight(self, epoch):
        for name, weight in self.weight_panel.items():
            self.writer.add_scalar(f"weight/{name}", weight, epoch)
        self.writer.flush()
        
    def print_loss(self, epoch):
        print(f"Loss at epoch {epoch}:", end=" ")
        for name, loss in self.loss_panel.items():
            print(f"{name}: {self.loss_format.format(loss.item())}", end=", ")
        print()
        
    def print_weight(self, epoch):
        print(f"Weights at epoch {epoch}:", end=" ")
        for name, weight in self.weight_panel.items():
            print(f"{name}: {self.loss_format.format(weight)}", end=", ")
        print()
    
