"""SCHISM mesh handling: triangulation (quads split) and regular-grid weights."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def split_quads(face_nodes):
    """Return an (n, 3) 0-based triangle array from SCHISM face_nodes.

    face_nodes is (nface, 3 or 4), 1-based, with a NaN or non-positive 4th
    entry for triangles. Each quad (0,1,2,3) becomes (0,1,2) + (0,2,3).
    """
    fn = np.asarray(face_nodes, dtype="float64")
    if fn.ndim != 2 or fn.shape[1] < 3:
        raise ValueError(f"bad face_nodes shape {fn.shape}")
    if fn.shape[1] == 3:
        return (fn - 1).astype("int32")
    col4 = fn[:, 3]
    is_quad = np.isfinite(col4) & (col4 > 0)
    tris = fn[:, :3]
    quads = fn[is_quad]
    extra = quads[:, [0, 2, 3]]
    return (np.vstack([tris, extra]) - 1).astype("int32")


def load_mesh(out2d_path: Path):
    """Return (x, y, depth, triangles) from an out2d_N.nc file."""
    import xarray as xr
    keep = ["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y",
            "SCHISM_hgrid_face_nodes", "depth"]
    ds = xr.open_dataset(str(out2d_path), decode_times=False)
    try:
        x = np.asarray(ds[keep[0]].values, dtype="float64")
        y = np.asarray(ds[keep[1]].values, dtype="float64")
        face = np.asarray(ds[keep[2]].values)
        depth = np.asarray(ds[keep[3]].values, dtype="float32")
    finally:
        ds.close()
    return x, y, depth, split_quads(face)


@dataclass
class Weights:
    """Barycentric interpolation from mesh nodes to a regular grid."""
    nodes: np.ndarray    # (npts, 3) int32, node indices of the enclosing triangle
    w: np.ndarray        # (npts, 3) float32 barycentric weights
    valid: np.ndarray    # (npts,) bool, point lies inside the mesh
    shape: tuple         # (ny, nx)

    def apply(self, nodal):
        """Interpolate a nodal field; NaN outside the mesh or next to NaN nodes."""
        v = np.asarray(nodal, dtype="float32")
        out = np.full(self.nodes.shape[0], np.nan, dtype="float32")
        idx = np.flatnonzero(self.valid)
        vals = v[self.nodes[idx]]                       # (m, 3)
        res = (vals * self.w[idx]).sum(axis=1)
        res[~np.isfinite(vals).all(axis=1)] = np.nan
        out[idx] = res
        return out.reshape(self.shape)

    def save(self, path: Path):
        np.savez(path, nodes=self.nodes, w=self.w, valid=self.valid,
                 shape=np.array(self.shape))

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        return cls(z["nodes"], z["w"], z["valid"], tuple(int(s) for s in z["shape"]))


def build_trifinder(x, y, tris):
    import matplotlib.tri as mtri
    return mtri.Triangulation(x, y, tris).get_trifinder()


def build_weights(finder, x, y, tris, grid) -> Weights:
    """Find the enclosing triangle of every grid cell centre and its weights."""
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    px, py = lon2d.ravel(), lat2d.ravel()
    t = np.asarray(finder(px, py))
    valid = t >= 0
    nodes = np.zeros((px.size, 3), dtype="int32")
    w = np.zeros((px.size, 3), dtype="float32")
    tv = t[valid]
    n = tris[tv]
    xa, xb, xc = x[n[:, 0]], x[n[:, 1]], x[n[:, 2]]
    ya, yb, yc = y[n[:, 0]], y[n[:, 1]], y[n[:, 2]]
    det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
    qx, qy = px[valid], py[valid]
    wa = ((yb - yc) * (qx - xc) + (xc - xb) * (qy - yc)) / det
    wb = ((yc - ya) * (qx - xc) + (xa - xc) * (qy - yc)) / det
    nodes[valid] = n
    w[valid] = np.stack([wa, wb, 1.0 - wa - wb], axis=1)
    return Weights(nodes, w, valid, grid.shape)
