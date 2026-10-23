"""Map of the zoom area: mesh bathymetry, the mesh triangles and nearby cities.

Reads the mesh (nodes, faces, depth) from the first out2d file, keeps only what lies in the
zoom box and draws the triangles as thin white lines so the mesh resolution is visible.
"""

import time

import numpy as np

from . import mesh
from .config import Config
from .map_domain import map_opts

LAND = "0.85"

DEFAULT_CITIES = [          # name, lat, lon (degrees, east positive)
    ("Tijuana", 32.53, -117.04),
    ("San Diego", 32.72, -117.16),
    ("Oceanside", 33.20, -117.38),
    ("Long Beach", 33.77, -118.19),
    ("Los Angeles", 34.05, -118.24),
    ("Ventura", 34.27, -119.23),
    ("Santa Barbara", 34.42, -119.70),
    ("Point Conception", 34.45, -120.47),
    ("Santa Maria", 34.95, -120.44),
    ("San Luis Obispo", 35.28, -120.66),
    ("Morro Bay", 35.37, -120.85),
    ("Big Sur", 36.27, -121.81),
    ("Carmel-by-the-Sea", 36.55, -121.92),
]


def unique_edges(tris):
    """Unique undirected edges (m, 2) of an (n, 3) triangle array."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]).astype("int64")
    e.sort(axis=1)
    n = int(e.max()) + 1
    key = np.unique(e[:, 0] * n + e[:, 1])
    return np.stack([key // n, key % n], axis=1)


def edge_lengths_km(x, y, edges):
    lat = 0.5 * (y[edges[:, 0]] + y[edges[:, 1]])
    dx = (x[edges[:, 0]] - x[edges[:, 1]]) * np.cos(np.radians(lat)) * 111.195
    dy = (y[edges[:, 0]] - y[edges[:, 1]]) * 111.195
    return np.hypot(dx, dy)


def select_box(x, y, tris, box, pad):
    """Triangles with at least one node in the padded box, re-indexed. Returns (index map, tris)."""
    inb = ((x >= box["lon_min"] - pad) & (x <= box["lon_max"] + pad) &
           (y >= box["lat_min"] - pad) & (y <= box["lat_max"] + pad))
    sel = tris[inb[tris].any(axis=1)]
    nodes, inv = np.unique(sel, return_inverse=True)
    return nodes, inv.reshape(sel.shape).astype("int32")


def run_zoom_map(cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import FuncFormatter

    from .plotting import fmt_lat, fmt_lon

    opts = map_opts(cfg)
    z = cfg.zoom
    src = cfg.outputs_dir / f"out2d_{cfg.stack_for_day(cfg.plot_start)}.nc"
    if not src.exists():
        raise FileNotFoundError(src)
    t0 = time.time()
    print(f"Loading mesh from {src}", flush=True)
    x, y, depth, tris = mesh.load_mesh(src)
    nodes, t = select_box(x, y, tris, z, pad=0.1)
    xs, ys, ds = x[nodes], y[nodes], depth[nodes].astype("float64")
    del x, y, depth, tris
    print(f"  {nodes.size} nodes, {t.shape[0]} triangles in the zoom box [{time.time() - t0:.0f}s]",
          flush=True)
    if t.shape[0] == 0:
        raise RuntimeError("no mesh triangles inside the zoom box")

    edges = unique_edges(t)
    inside = ((xs[edges] >= z["lon_min"]) & (xs[edges] <= z["lon_max"]) &
              (ys[edges] >= z["lat_min"]) & (ys[edges] <= z["lat_max"])).all(axis=1)
    L = edge_lengths_km(xs, ys, edges[inside] if inside.any() else edges)
    print(f"  element edge length in the box: min {L.min():.2f}  median {np.median(L):.2f}  "
          f"max {L.max():.2f} km", flush=True)

    levels = opts["depth_levels"]
    norm = mcolors.BoundaryNorm(levels, plt.get_cmap(opts["cmap"]).N, extend="max")
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    ax.set_facecolor(LAND)

    tri = mtri.Triangulation(xs, ys, t)
    dtri = ds[t].mean(axis=1)                       # one depth per triangle (flat shading)
    tri.set_mask(dtri <= 0)                         # land triangles are left to the base colour
    pm = ax.tripcolor(tri, facecolors=dtri, cmap=opts["cmap"], norm=norm, rasterized=True)
    ax.tricontour(tri, ds, levels=[0.0], colors="k", linewidths=0.8)

    if opts.get("mesh_lines", True):
        segs = np.stack([np.column_stack([xs[edges[:, 0]], ys[edges[:, 0]]]),
                         np.column_stack([xs[edges[:, 1]], ys[edges[:, 1]]])], axis=1)
        ax.add_collection(LineCollection(segs, colors="white", linewidths=0.12, alpha=0.7,
                                         rasterized=True))

    cities = opts.get("cities") or DEFAULT_CITIES
    for name, lat, lon in cities:
        lon360 = lon % 360.0
        if not (z["lon_min"] <= lon360 <= z["lon_max"] and z["lat_min"] <= lat <= z["lat_max"]):
            continue
        ax.plot(lon360, lat, "o", color="red", markersize=5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=6)
        east = lon360 > z["lon_max"] - 0.18 * (z["lon_max"] - z["lon_min"])
        ax.annotate(name, (lon360, lat), xytext=(-6 if east else 6, 4), textcoords="offset points",
                    ha="right" if east else "left", fontsize=9, fontweight="bold", color="k",
                    path_effects=halo, zorder=7)

    ax.set_xlim(z["lon_min"], z["lon_max"])
    ax.set_ylim(z["lat_min"], z["lat_max"])
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (z["lat_min"] + z["lat_max"]))))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.grid(color="k", alpha=0.15, linewidth=0.5)
    ax.set_title("STOFS-3D Pacific mesh and bathymetry (zoom area)", pad=10)
    ax.text(0.015, 0.012, f"element edge length in view: median {np.median(L):.1f} km "
            f"(min {L.min():.1f}, max {L.max():.1f})", transform=ax.transAxes, fontsize=8.5,
            va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, pad=0.3))
    cb = fig.colorbar(pm, ax=ax, fraction=0.04, pad=0.02, extend="max")
    cb.set_label("Model bathymetry (m)")
    fig.tight_layout()

    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.fig_dir / "zoom_map.png"
    fig.savefig(out, dpi=opts["dpi"])
    plt.close(fig)
    print(f"wrote {out}  [{time.time() - t0:.0f}s]", flush=True)
    return out
