"""Block-average (bin) native satellite fields onto a regular Grid."""

import numpy as np
from scipy import sparse


def _bin_matrix(coord, lo_edge, dx, n):
    """Sparse (n, len(coord)) indicator mapping native points to grid cells."""
    idx = np.floor((np.asarray(coord, dtype="float64") - lo_edge) / dx).astype(int)
    ok = (idx >= 0) & (idx < n)
    cols = np.flatnonzero(ok)
    return sparse.csr_matrix((np.ones(cols.size, dtype="float32"),
                              (idx[ok], cols)), shape=(n, len(coord)))


def bin_to_grid(lon, lat, field, grid, min_frac=0.5):
    """Average field(lat, lon) into grid cells.

    lon must already be in the grid's longitude convention. Cells need at
    least min_frac of the largest per-cell sample count to be valid, which
    suppresses cells that are mostly land/cloud.
    """
    ny, nx = grid.shape
    h = grid.dx / 2
    Mlat = _bin_matrix(lat, grid.lat[0] - h, grid.dx, ny)
    Mlon = _bin_matrix(lon, grid.lon[0] - h, grid.dx, nx)
    f = np.asarray(field, dtype="float32")
    good = np.isfinite(f)
    S = (Mlon @ (Mlat @ np.where(good, f, 0.0)).T).T
    C = (Mlon @ (Mlat @ good.astype("float32")).T).T
    cmax = C.max()
    out = np.full((ny, nx), np.nan, dtype="float32")
    if cmax <= 0:
        return out
    ok = C >= max(1.0, min_frac * cmax)
    out[ok] = S[ok] / C[ok]
    return out
