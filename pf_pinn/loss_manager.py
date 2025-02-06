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
            
    def check_loss(self):
        if len(self.loss_panel) == 0:
            raise ValueError("No loss is registered.")
            
        
    def update_weights(self):
        self.check_loss()
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
    
    def compute_similarity(self, name1, name2):
        self.check_loss()
        loss1 = self.loss_panel[name1] * self.weight_panel[name1]
        loss2 = self.loss_panel[name2] * self.weight_panel[name2]
        
        self.pinn.zero_grad()
        grad1 = self.pinn.gradient(loss1)
        self.pinn.zero_grad()
        grad2 = self.pinn.gradient(loss2)
        
        cos_sim = torch.sum(grad1 * grad2) / \
            (torch.sqrt(torch.sum(grad1**2)) * torch.sqrt(torch.sum(grad1**2)))
        grad_sim = 2*torch.sqrt(torch.sum(grad1**2)) * torch.sqrt(torch.sum(grad1**2)) / \
            (torch.sum(grad1**2) + torch.sum(grad2**2))
        
        return cos_sim.item(), grad_sim.item()
    
    def compute_grad_align_score(self):
        # gradient alignment score
        # 2 \| \frac{\sum_{i=1}^n  \frac{g_i}{\|g_i\|} }{n} \|^2 - 1
        # where n is the number of losses
        self.check_loss()
        self.pinn.zero_grad()
        grads = torch.stack([
            self.pinn.gradient(loss) 
            for loss in self.loss_panel.values()
        ])
        grad_norm = torch.sqrt(torch.sum(grads**2, dim=1))
        grad_unit = grads / grad_norm[:, None]
        grad_align = torch.sum(grad_unit, dim=0) / len(grad_unit)
        return 2*torch.sum(grad_align**2) - 1
        
        
        
