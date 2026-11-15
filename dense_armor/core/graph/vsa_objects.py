"""VSA encoding of ARC objects (colour, centre, shape)."""

from __future__ import annotations

import numpy as np

from dense_armor.core.graph.vsa import (
    N_DIMENSIONS,
    SSPSpace,
    bind,
    bundle,
    random_unit_vector,
    similarity,
    unbind,
)

MAX_GRID_SIZE = 30
N_COLOURS = 10


class ObjectEncoder:
    def __init__(self, n=N_DIMENSIONS, length_scale=0.25, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.colour_sps = [random_unit_vector(self.rng, n) for _ in range(N_COLOURS)]
        self.role_colour = random_unit_vector(self.rng, n)
        self.role_centre = random_unit_vector(self.rng, n)
        self.role_shape = random_unit_vector(self.rng, n)
        self.ssp = SSPSpace(n=n, length_scale=length_scale, seed=seed)
        coords = np.arange(-MAX_GRID_SIZE, MAX_GRID_SIZE + 1, dtype=float)
        xx, yy = np.meshgrid(coords, coords)
        self.centre_index = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        self.centre_samples = self.ssp.encode(self.centre_index)

    def encode_object(self, cells, colour, grid_h, grid_w):
        r_min = min(r for r, _ in cells)
        r_max = max(r for r, _ in cells)
        c_min = min(c for _, c in cells)
        c_max = max(c for _, c in cells)
        cx = (r_min + r_max) / 2.0
        cy = (c_min + c_max) / 2.0
        rel_x = cx - (grid_h - 1) / 2.0
        rel_y = cy - (grid_w - 1) / 2.0
        centre_sp = self.ssp.encode(np.array([rel_x, rel_y]))
        rel_positions = np.array([(r - cx, c - cy) for r, c in cells], dtype=float)
        pixel_sps = self.ssp.encode(rel_positions)
        shape_sp = bundle(*pixel_sps)
        return bundle(
            bind(self.role_colour, self.colour_sps[colour]),
            bind(self.role_centre, centre_sp),
            bind(self.role_shape, shape_sp),
        )

    def decode_colour(self, obj_sp):
        colour_sp = unbind(obj_sp, self.role_colour)
        sims = [similarity(colour_sp, c) for c in self.colour_sps]
        return int(np.argmax(sims))

    def decode_centre(self, obj_sp):
        centre_sp = unbind(obj_sp, self.role_centre)
        sims = self.centre_samples @ centre_sp
        idx = int(np.argmax(sims))
        return tuple(int(v) for v in self.centre_index[idx])

    def decode_shape(self, obj_sp):
        return unbind(obj_sp, self.role_shape)
