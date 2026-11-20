# SPDX-FileCopyrightText: 2025 Hrushikesh Budhale
# SPDX-FileCopyrightText: 2026 Sotiris Papatheodorou
# SPDX-License-Identifier: MIT

import numpy as np
import torch


# TODO: add the view direction as an auxiliary input
class FtexNGP(torch.nn.Module):
    """
    2D Instant-NGP
    https://github.com/HrushikeshBudhale/NGP_from_scratch/
    """
    D = 2       # grid dimension (hardcoded)
    L = 16      # number of levels
    F = 2       # features per hash table entry
    T = 2**19   # hash table size (entries per level) [2¹⁴,2²⁴]
    NMIN = 16   # coarsest resolution
    NMAX = 2048 # finest resolution

    FV_SZ = L * F

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.levels = np.geomspace(self.NMIN, self.NMAX, self.L, dtype=int)
        self.primes = torch.tensor([1, 2_654_435_761])
        self.lookup_tables = torch.nn.ParameterDict(
            {
                str(i): torch.nn.Parameter(
                       (torch.randn((self.T, self.F), dtype=self.dtype) * 2 - 1) * 1e-4
                    ) for i in range(self.L)
            }
        ) # {L: (T, F)}
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.FV_SZ, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
            torch.nn.Sigmoid(),
        ).to(dtype=self.dtype)

    def _get_grid_vertices(self, x):
        # x: (N, D)
        floor = torch.floor(x) # (N, D)
        ceil  = torch.ceil(x)  # (N, D)
        vertices = torch.empty_like(x, dtype=torch.int64)
        vertices = vertices.unsqueeze(1).repeat(1, 2**self.D, 1) # (N, 2**D, D)
        vertices[:, 0] = floor
        vertices[:, 1] = torch.cat([ ceil[:,[0]], floor[:,[1]]], dim=-1)
        vertices[:, 2] = torch.cat([floor[:,[0]],  ceil[:,[1]]], dim=-1)
        vertices[:, 3] = ceil
        return vertices

    def _spatial_hashing(self, x):
        # x: (N, 2**D, D)
        y = x * self.primes                         # (N, 2**D, D)
        z = torch.bitwise_xor(y[..., 0], y[..., 1]) # (N, 2**D)
        return z % self.T                           # (N, 2**D)

    def _get_interpolated_features_from_table(self, x, grid_vertices, corner_features):
        # x: (N, D)
        # grid_vertices: (N, 2**D, D)
        # corner_features: (N, 2**D, F)
        corner_features = corner_features.swapaxes(1, 2) # (N, F, 2**D)
        # points' position within each D dimensional unit cube centered at origin
        sample_position = x - grid_vertices[:, 0] - 0.5 # (N, D)
        # interpolation
        corner_features = corner_features.reshape(corner_features.shape[0], self.F, 2, 2) # (N, F, 2, 2)
        sample_position = sample_position.unsqueeze(1).unsqueeze(1)                       # (N, 1, 1, D)
        features = torch.nn.functional.grid_sample(corner_features, sample_position,
                                                   mode='bilinear', align_corners=False)  # (N, F, 1, 1)
        return features.squeeze() # (N, F)

    def _features(self, x):
        # x: (N, D)
        features = torch.empty((x.shape[0], self.FV_SZ), device=x.device, dtype=self.dtype) # (N, F*L)
        for i, level in enumerate(self.levels):
            level_scaled_pos = x * level
            grid_vertices = self._get_grid_vertices(level_scaled_pos) # (N, 2**D, D)
            hashes = self._spatial_hashing(grid_vertices) # (N, 2**D)
            corner_features = self.lookup_tables[str(i)][hashes] # (N, 2**D, F)
            features[:, i * self.F : (i+1) * self.F] = \
                self._get_interpolated_features_from_table(level_scaled_pos, grid_vertices, corner_features) # (N, F)
        return features

    def forward(self, uv, dir):
        # uv:          (N, 2)
        # dir:         (N, 2)
        # forward(uv): (N, 3)
        # XXX: NeuTex uses 3 dimensions for UV (cartesian coordinates on the
        # unit sphere?). Why?
        return self.mlp.forward(self._features(uv))
