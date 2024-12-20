import numpy as np
from pyDOE import lhs
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def make_flattend_grid_data(spans, nums):
    series = [np.random.uniform(*span, num) for span, num in zip(spans, nums)]
    grid = np.meshgrid(*series)
    flatten = np.vstack([g.flatten() for g in grid]).T
    return flatten


def make_lhs_sampling_data(mins, maxs, num):
    lb = np.array(mins)
    ub = np.array(maxs)
    if not len(lb) == len(ub):
        raise ValueError(f"mins and maxs should have the same length.")
    ret = lhs(len(lb), int(num)) * (ub - lb) + lb
    return torch.tensor(ret, dtype=torch.float32, device=DEVICE)


def make_semi_circle_data(radius, num, center=[0, 0]):
    square = make_lhs_sampling_data(mins=[center[0] - radius, center[1]],
                                    maxs=[center[0] + radius,
                                          center[1] + radius],
                                    num=num)
    semi_circle = square[square[:, 0] ** 2 + square[:, 1] ** 2 <= radius ** 2]
    return semi_circle


def make_uniform_grid_data(mins, maxs, num):
    if not len(mins) == len(maxs) == len(num):
        raise ValueError(f"mins, maxs, num should have the same length.")
    each_col = [torch.linspace(mins[i], maxs[i], num[i], device=DEVICE)[1:-1]
                for i in range(len(mins))]
    return torch.stack(torch.meshgrid(*each_col, indexing='ij'), axis=-1).reshape(-1, len(mins))


def make_uniform_grid_data_transition(mins, maxs, num):
    if not len(mins) == len(maxs) == len(num):
        raise ValueError(f"mins, maxs, num should have the same length.")
    
    each_col = [torch.linspace(mins[i], maxs[i], num[i], device=DEVICE)[1:-1]
                for i in range(len(mins))]
    distances = [(maxs[i] - mins[i]) / (num[i] - 1) for i in range(len(mins))]
    shift = [torch.tensor(np.random.uniform(-distances[i], distances[i], 1), device=DEVICE)
             for i in range(len(distances))]
    shift = torch.cat(shift, dim=0)
    each_col = [each_col[i] + shift[i] for i in range(len(each_col))]

    return torch.stack(torch.meshgrid(*each_col, indexing="ij"), axis=-1).reshape(-1, len(mins))

