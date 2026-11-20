# SPDX-FileCopyrightText: 2026 Sotiris Papatheodorou
# SPDX-License-Identifier: MIT

import torch

from .texture.ngp import FtexNGP as Ftex
from .uv.mlp import FuvMLP as Fuv, FuviMLP as Fuvi

class INeuTex(torch.nn.Module):
    """NeuTex using Instant-NGP ((X,D) → (RGB))"""
    def __init__(self, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        with self.device:
            self.Fuv = Fuv().to(dtype=self.dtype)
            # XXX: only instantiate Fuvi when training
            self.Fuvi = Fuvi().to(dtype=self.dtype)
            self.Ftex = Ftex().to(dtype=self.dtype)

    def forward(self, x, train=False):
        # x:          (N, 3)
        # forward(x): (N, 3)
        # forward(x): (N, 6) train=True
        pos = x[:,0:3]
        #dir = x[:,3:5]
        dir = None
        uv = self.Fuv.forward(pos)
        rgb = self.Ftex.forward(uv, dir)
        y = rgb
        if train:
            # Also compute the inverse UV mapping when training.
            y = torch.cat((y, self.Fuvi(uv)), -1)
        return y
