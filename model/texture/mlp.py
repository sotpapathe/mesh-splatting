# SPDX-FileCopyrightText: 2026 Sotiris Papatheodorou
# SPDX-License-Identifier: MIT

import torch

from ..common import PositionalEncodingPlus


# TODO: add the direction-dependent layers
class FtexMLP(torch.nn.Module):
    # XXX: NeuTex uses 3 input dimensions for UV. How are they computed?
    DIMS_IN = 2 #3
    DIMS_OUT = 3
    NUM_FREQUENCIES = 10
    LEAKYRELU_SLOPE = 0.2

    def __init__(self, dtype=torch.float32):
        super().__init__()
        encoder_out_dims = FtexMLP.DIMS_IN * (1 + 2 * FtexMLP.NUM_FREQUENCIES)
        self.dtype = dtype
        self.layers = torch.nn.Sequential(
            PositionalEncodingPlus(FtexMLP.NUM_FREQUENCIES),
            torch.nn.Linear(encoder_out_dims, 256),
            torch.nn.LeakyReLU(FtexMLP.LEAKYRELU_SLOPE),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(FtexMLP.LEAKYRELU_SLOPE),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(FtexMLP.LEAKYRELU_SLOPE),
            torch.nn.Linear(256, 256),
            torch.nn.Softplus(),
            torch.nn.Linear(256, 3)
        ).to(dtype=self.dtype)

    def forward(self, uv, dir):
        if __debug__:
            print('input', uv)
            for i, l in enumerate(self.layers):
                uv = l.forward(uv)
                print('LAYER', i, l)
                if hasattr(l, 'weight'):
                    print('WEIGHTS', l.weight)
                print('OUTPUT', uv)
            return uv
        else:
            return self.layers.forward(uv)
