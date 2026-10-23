"""Map of the whole model domain: bathymetry with the Niño boxes (blue) and the zoom box (red).

Uses the bathymetry already regridded by `prepare` (cache/grid_fields.npz), so it is quick
and needs neither the model output nor the satellite data.
"""

import numpy as np

from .config import Config

LAND = "0.85"
# one blue per Niño box so the nested boxes can be told apart
NINO_STYLE = {
    "nino34": dict(color="#08306b", ls="-", label="Niño 3.4  (5°S–5°N, 170°W–120°W)"),
    "nino3":  dict(color="#2171b5", ls="--", label="Niño 3  (5°S–5°N, 150°W–90°W)"),
    "nino12": dict(color="#6baed6", ls="-", label="Niño 1+2  (10°S–0°, 90°W–80°W)"),
}
# (dx, dy) in degrees of the text label relative to the box corner, and alignment
LABEL_POS = {
    "nino34": ("lon_min", "lat_min", 0.0, -1.2, "left", "top"),
    "nino3":  ("lon_max", "lat_max", 0.0, 1.0, "right", "bottom"),
    "nino12": ("lon_min", "lat_min", 0.0, -1.2, "left", "top"),
}
NAME = {"nino34": "Niño 3.4", "nino3": "Niño 3", "nino12": "Niño 1+2"}


def map_opts(cfg: Config):
    o = dict(cmap="viridis_r", depth_levels=[0, 50, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
             dpi=200)
    o.update(cfg.raw.get("maps") or {})
    return o


def box_xy(b):
    return b["lon_min"], b["lat_min"], b["lon_max"] - b["lon_min"], b["lat_max"] - b["lat_min"]


def run_domain_map(cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    from .plotting import fmt_lat, fmt_lon
    from .prepare import load_cache

    opts = map_opts(cfg)
    _, depth, _ = load_cache(cfg)                    # refuses a stale cache
    g = cfg.grid("full")
    d = np.ma.masked_where(~np.isfinite(depth["full"]) | (depth["full"] <= 0), depth["full"])

    levels = opts["depth_levels"]
    norm = mcolors.BoundaryNorm(levels, plt.get_cmap(opts["cmap"]).N, extend="max")
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]

    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_facecolor(LAND)
    e = g.extent
    pm = ax.pcolormesh(np.append(g.lon - g.dx / 2, g.lon[-1] + g.dx / 2),
                       np.append(g.lat - g.dx / 2, g.lat[-1] + g.dx / 2),
                       d, cmap=opts["cmap"], norm=norm, rasterized=True)
    ax.contour(g.lon, g.lat, np.where(np.isfinite(depth["full"]), depth["full"], -1.0),
               levels=[0.0], colors="k", linewidths=0.5)

    handles = []
    for name, st in NINO_STYLE.items():
        b = cfg.regions.get(name)
        if not b:
            continue
        x, y, w, h = box_xy(b)
        r = Rectangle((x, y), w, h, fill=False, edgecolor=st["color"], linestyle=st["ls"],
                      linewidth=2.0, path_effects=halo, zorder=5)
        ax.add_patch(r)
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=st["color"],
                                 linestyle=st["ls"], linewidth=2.0, label=st["label"]))
        kx, ky, dx, dy, ha, va = LABEL_POS[name]
        ax.text(b[kx] + dx, b[ky] + dy, NAME[name], color=st["color"], fontsize=10,
                fontweight="bold", ha=ha, va=va, path_effects=halo, zorder=6)

    z = cfg.zoom
    x, y, w, h = box_xy(z)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2.2,
                           path_effects=halo, zorder=5))
    handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2.2,
                             label=f"Zoom box  ({z['lat_min']:g}–{z['lat_max']:g}°N, "
                                   f"{360 - z['lon_max']:g}–{360 - z['lon_min']:g}°W)"))
    ax.annotate("Zoom", xy=(z["lon_min"], z["lat_max"]),
                xytext=(z["lon_min"] - 6.0, z["lat_max"] + 7.0), color="red", fontsize=10,
                fontweight="bold", ha="right", path_effects=halo, zorder=6,
                arrowprops=dict(arrowstyle="-", color="red", lw=1.2))

    ax.set_xlim(e[0], e[1])
    ax.set_ylim(e[2], e[3])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.grid(color="k", alpha=0.15, linewidth=0.5)
    ax.set_title("STOFS-3D Pacific model domain: bathymetry, Niño regions and zoom box")
    cb = fig.colorbar(pm, ax=ax, fraction=0.025, pad=0.015, extend="max")
    cb.set_label("Model bathymetry (m)")
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92)
    fig.tight_layout()

    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.fig_dir / "domain_map.png"
    fig.savefig(out, dpi=opts["dpi"])
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out
