import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
# from allen_cahn.sampler import GeoTimeSampler
import time
import configparser

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding(torch.nn.Module):
    def __init__(self, in_features, embedding_features, std=2, method="trig"):
        super().__init__()
        self.method = method
        self.linear = torch.nn.Linear(in_features, embedding_features)
        if self.method == "trig":
            self.linear.weight.data = \
                torch.randn(embedding_features, in_features) * std * np.pi
            self.linear.bias.data = torch.randn(embedding_features) * std * np.pi
            # self.linear.bias.data.zero_()
            for param in self.linear.parameters():
                param.requires_grad = False
        elif self.method == "linear":
            # torch.nn.init.xavier_normal_(self.linear.weight)
            self.linear.weight.data = \
                torch.randn(embedding_features, in_features) * std
            for param in self.linear.parameters():
                param.requires_grad = False
            pass


    def forward(self, x):
        x = self.linear(x)
        method = self.method
        if method == "trig":
            return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        elif method == "linear":
            return x
        else:
            raise ValueError("Ivalid method.")


# class SpatialTemporalFourierEmbedding(torch.nn.Module):

#     def __init__(self, in_features, embedding_features, std=2):
#         super().__init__()
#         self.spatial_embedding = FourierEmbedding(in_features-1,
#                                                   embedding_features, std,
#                                                   method="trig")
#         self.spatial_embedding_back = torch.nn.Linear(embedding_features*2, in_features-1)
#         # self.temporal_embedding = FourierEmbedding(1, embedding_features, std*3,
#         #                                            method="trig")

#     def forward(self, x):
#         y_spatial = self.spatial_embedding(x[:, :-1])
#         y_spatial = self.spatial_embedding_back(y_spatial)
#         # y_temporal = self.temporal_embedding(x[:, -1:])
        
#         return torch.cat([y_spatial, x[:, -1:]], dim=1)


# class MultiScaleFourierEmbedding(torch.nn.Module):

#     def __init__(self, in_features, embedding_features=8, std=1):
#         super().__init__()
#         self.spatial_low_embedding = FourierEmbedding(
#             in_features-1,
#             embedding_features,
#             std/2
#         )
#         self.spatial_high_embedding = FourierEmbedding(
#             in_features-1,
#             embedding_features,
#             std*5
#         )
#         self.temporal_embedding = FourierEmbedding(
#             1, embedding_features,
#             std,
#         )

#     def forward(self, x):
#         y_low = self.spatial_low_embedding(x[:, :-1])
#         y_high = self.spatial_high_embedding(x[:, :-1])
#         y_temporal = self.temporal_embedding(x[:, -1:])
#         return torch.cat([y_low, y_high, y_temporal], dim=1)


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_features, out_features, scale=1):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features)*np.pi*scale, requires_grad=True)

    
    def forward(self, x):
        x = torch.matmul(x, self.weights)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)



class MiltiscaleFourierEmbedding(nn.Module):
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
    def __init__(self, in_features, out_features, scale=2):
        super().__init__()
        self.spatial_weight = nn.Parameter(torch.randn(in_features-1, out_features) * np.pi * scale, requires_grad=True)
        self.temporal_weight = nn.Parameter(torch.linspace(1/2, 8, 2*out_features).reshape(1, -1), requires_grad=True)
        
        
    def forward(self, x):
        y_spatial = x[:, :-1]
        y_temporal = x[:, -1:]
        y_spatial = torch.matmul(y_spatial, self.spatial_weight)
        y_temporal = torch.matmul(y_temporal, torch.ones_like(self.temporal_weight)) ** self.temporal_weight
        return torch.cat([torch.sin(y_spatial), torch.cos(y_spatial), y_temporal], dim=-1)
        