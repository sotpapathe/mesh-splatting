# SPDX-FileCopyrightText: 2026 Sotiris Papatheodorou
# SPDX-License-Identifier: MIT

import math
import torch

class PositionalEncoding(torch.nn.Module):
    """The NeRF one."""
    def __init__(self, num_frequencies):
        super().__init__()
        # f = [1π, 2π, 4π, 8π, ... ]
        f = math.pi * 2 ** torch.arange(num_frequencies, dtype=torch.get_default_dtype())
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f.to(device=x.device, dtype=x.dtype)
        input_dim = x.size()[-1]
        output_dim = 2 * len(self.f) * input_dim
        # fx = [1π x0, 2π x0, 4π x0, ..., 1π x2, 2π x1, 4π x1, ...]
        fx = x.repeat_interleave(len(self.f), -1) * f.repeat(input_dim)
        s = torch.sin(fx)
        c = torch.cos(fx)
        # Interleave the individual elements of the sin and cos tensors.
        return torch.stack((s, c), dim=-2).transpose(-1, -2).reshape((-1, output_dim))


class PositionalEncodingPlus(PositionalEncoding):
    def __init__(self, num_frequencies):
        super().__init__(num_frequencies)

    def forward(self, x):
        # Concatenate the input as-is with its positional encoding.
        return torch.cat((x, super().forward(x)), -1)
