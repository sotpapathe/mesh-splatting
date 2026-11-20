# SPDX-FileCopyrightText: 2026 Sotiris Papatheodorou
# SPDX-License-Identifier: MIT

import torch


class FuvMLP(torch.nn.Module):
    """UV mapping (X → UV)"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Tanh(),
        ).to(dtype=self.dtype)

    def forward(self, x):
        # x:          (N, 3)
        # forward(x): (N, 2)
        return self.layers.forward(x)


class FuviMLP(torch.nn.Module):
    """Inverse UV mapping (UV → X)"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3),
            torch.nn.Tanh(),
        ).to(dtype=self.dtype)

    def forward(self, x):
        # x:          (N, 2)
        # forward(x): (N, 3)
        return self.layers.forward(x)
