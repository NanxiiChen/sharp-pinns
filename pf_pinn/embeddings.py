import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_features, out_features, scale=1):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features)*np.pi*scale, requires_grad=True)

    
    def forward(self, x):
        x = torch.matmul(x, self.weights)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)



class MultiscaleFourierEmbedding(nn.Module):
    # output shape is 4 * out_features
    def __init__(self, in_features, out_features, scale=1):
        super().__init__()
        self.high_embedding = FourierFeatureEmbedding(in_features, out_features, scale*5)
        self.low_embedding = FourierFeatureEmbedding(in_features, out_features, scale)


    def forward(self, x):
        y_high = self.high_embedding(x)
        y_low = self.low_embedding(x)
        return torch.cat([y_high, y_low], dim=-1)

    
class SpatialTemporalFourierEmbedding(nn.Module):
    # output shape is 4 * out_features
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        self.spatial_weight = nn.Parameter(torch.randn(in_features-1, out_features)\
                                            * np.pi * scale, requires_grad=True)
        self.temporal_weight = nn.Parameter(torch.randn(1, out_features)\
                                            * np.pi * scale/6, requires_grad=True)
        
    def forward(self, x):
        y_spatial = x[:, :-1]
        y_temporal = x[:, -1:]
        y_spatial = torch.matmul(y_spatial, self.spatial_weight)
        y_temporal = torch.matmul(y_temporal, self.temporal_weight)
        return torch.cat([torch.sin(y_spatial), \
                            torch.cos(y_spatial), \
                            torch.sin(y_temporal), \
                            torch.cos(y_temporal)], dim=-1)
