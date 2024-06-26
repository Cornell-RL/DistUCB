import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import torch
# from sklearn.linear_model import LinearRegression

def get_model(cfg, num_actions):
    if cfg.alg == "distributional":
        num_actions = num_actions * cfg.task.num_atoms

    if cfg.task.task == "cifar100":
        model = resnet18()
        in_features = model.fc.in_features
        out_features = num_actions
        model.fc = torch.nn.Linear(in_features, out_features)
    elif cfg.task.task == "prudential":
        input_size = 1887
        hidden_sizes = [3000, 4000]

        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

    elif cfg.task.task == "housing":
        input_size = 88
        hidden_sizes = [32768, 16384]

        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )
    else:
        raise NotImplementedError("Task doesn't have model")




    return model


prudential_cost_to_idx = {
    0: 0,
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    50: 5,
    60: 6,
    70: 7,
    100: 8,
}

cifar_cost_to_idx = {
    0: 0,
    50: 1,
    100: 2,
}

fine_to_coarse = {
    0: 4,
    1: 1,
    2: 14,
    3: 8,
    4: 0,
    5: 6,
    6: 7,
    7: 7,
    8: 18,
    9: 3,
    10: 3,
    11: 14,
    12: 9,
    13: 18,
    14: 7,
    15: 11,
    16: 3,
    17: 9,
    18: 7,
    19: 11,
    20: 6,
    21: 11,
    22: 5,
    23: 10,
    24: 7,
    25: 6,
    26: 13,
    27: 15,
    28: 3,
    29: 15,
    30: 0,
    31: 11,
    32: 1,
    33: 10,
    34: 12,
    35: 14,
    36: 16,
    37: 9,
    38: 11,
    39: 5,
    40: 5,
    41: 19,
    42: 8,
    43: 8,
    44: 15,
    45: 13,
    46: 14,
    47: 17,
    48: 18,
    49: 10,
    50: 16,
    51: 4,
    52: 17,
    53: 4,
    54: 2,
    55: 0,
    56: 17,
    57: 4,
    58: 18,
    59: 17,
    60: 10,
    61: 3,
    62: 2,
    63: 12,
    64: 12,
    65: 16,
    66: 12,
    67: 1,
    68: 9,
    69: 19,
    70: 2,
    71: 10,
    72: 0,
    73: 1,
    74: 16,
    75: 12,
    76: 9,
    77: 13,
    78: 15,
    79: 13,
    80: 16,
    81: 19,
    82: 2,
    83: 4,
    84: 6,
    85: 19,
    86: 5,
    87: 5,
    88: 8,
    89: 19,
    90: 18,
    91: 1,
    92: 2,
    93: 15,
    94: 6,
    95: 0,
    96: 17,
    97: 8,
    98: 14,
    99: 13,
}
