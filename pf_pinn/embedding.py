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


# class FourierEmbedding(torch.nn.Module):
#     def __init__(self, in_features, embedding_features, std=1, method="trig"):
#         super().__init__()
#         self.method = method
#         self.linear = torch.nn.Linear(in_features, embedding_features)
#         if self.method == "trig":
#             self.linear.weight.data = \
#                 torch.randn(embedding_features, in_features) * std * np.pi
#             self.linear.bias.data.zero_()
#             for param in self.linear.parameters():
#                 param.requires_grad = False
#         elif self.method == "linear":
#             # torch.nn.init.xavier_normal_(self.linear.weight)
#             self.linear.weight.data = \
#                 torch.randn(embedding_features, in_features) * std
#             for param in self.linear.parameters():
#                 param.requires_grad = False
#             pass



#     def forward(self, x):
#         x = self.linear(x)
#         method = self.method
#         if method == "trig":
#             return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
#         elif method == "linear":
#             return x
#         else:
#             raise ValueError("Ivalid method.")


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
    def __init__(self, in_features, out_features, scale=1, method="trig"):
        super().__init__()
        if method == "trig":
            self.weights = nn.Parameter(torch.randn(in_features, out_features) * np.pi * scale, requires_grad=False)
        elif method == "linear":
            self.weights = nn.Parameter(torch.randn(in_features, 2*out_features) * np.pi * scale, requires_grad=False)
        self.method = method
    
    def forward(self, x):
        x = torch.matmul(x, self.weights)
        if self.method == "trig":
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        elif self.method == "linear":
            return x
        
    
class SpatialTemporalFourierEmbedding(nn.Module):
    # output shape is 4 * out_features
    def __init__(self, in_features, out_features, scale=1):
        super().__init__()
        self.spatial_embedding = FourierFeatureEmbedding(in_features-1, out_features, scale)
        self.temporal_embedding = FourierFeatureEmbedding(1, out_features, scale, method="linear")
        # self.temporal_embedding = nn.Sequential(
        #     nn.Linear(1, out_features),
        #     nn.Tanh()
        # )
        # self.temporal_embedding[0].weight.data = torch.randn(out_features, 1) * scale * np.pi

    def forward(self, x):
        y_spatial = self.spatial_embedding(x[:, :-1])
        y_temporal = self.temporal_embedding(x[:, -1:])
        return torch.cat([y_spatial, y_temporal], dim=-1)
        