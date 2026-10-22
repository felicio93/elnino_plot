"""Side-by-side model vs satellite SST frames (optionally with currents)."""

from datetime import date
from pathlib import Path

import numpy as np

from .config import PLOT_TYPES

LAND = "0.80"


def fmt_lon(v, _pos=None):
    v = ((v + 180.0) % 360.0) - 180.0
    if abs(v) < 1e-9 or abs(abs(v) - 180) < 1e-9:
        return f"{abs(v):g}°"
    return f"{abs(v):g}°{'E' if v > 0 else 'W'}"


def fmt_lat(v, _pos=None):
    if abs(v) < 1e-9:
        return "0°"
    return f"{abs(v):g}°{'N' if v > 0 else 'S'}"


def _even_px(inches, dpi):
    px = int(round(inches * dpi))
    return px + (px % 2)


def make_layout(panel_w, panel_h, dpi, left=0.85, gap=0.6, top=0.95, bottom=0.7,
                cbar_pad=0.3, cbar_w=0.22, cbar_label=0.95):
    """Figure with two equal panels and one shared colorbar; fixed pixel size."""
    import matplotlib.pyplot as plt
    W = left + 2 * panel_w + gap + cbar_pad + cbar_w + cbar_label
    H = bottom + panel_h + top
    Wpx, Hpx = _even_px(W, dpi), _even_px(H, dpi)
    fig = plt.figure(figsize=(Wpx / dpi, Hpx / dpi), dpi=dpi)
    Wi, Hi = Wpx / dpi, Hpx / dpi

    def rect(x, y, w, h):
        return [x / Wi, y / Hi, w / Wi, h / Hi]

    ax1 = fig.add_axes(rect(left, bottom, panel_w, panel_h))
    ax2 = fig.add_axes(rect(left + panel_w + gap, bottom, panel_w, panel_h))
    cax = fig.add_axes(rect(left + 2 * panel_w + gap + cbar_pad, bottom, cbar_w, panel_h))
    return fig, ax1, ax2, cax, (Wi, Hi, top)


def render_frame(cfg, ptype, d: date, grid, model, sat, depth=None,
                 arrows=None, out_path: Path = None):
    """Render one frame.

    model, sat : (ny, nx) SST on `grid`.
    depth      : (ny, nx) bathymetry on `grid` (for isobaths) or None.
    arrows     : (grid_a, u, v) current vectors on the arrow grid, or None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    region, with_cur = PLOT_TYPES[ptype]
    s = cfg.sst
    lim = s[region]
    dpi = int(s.get("dpi", 100))
    box = cfg.domain if region == "full" else cfg.zoom
    dlon = box["lon_max"] - box["lon_min"]
    dlat = box["lat_max"] - box["lat_min"]
    pw = float(s["panel_width_full" if region == "full" else "panel_width_zoom"])
    ph = pw * dlat / dlon
    fig, ax1, ax2, cax, (Wi, Hi, top) = make_layout(pw, ph, dpi)

    cmap = plt.get_cmap(s.get("cmap", "jet")).with_extremes(bad=LAND)
    ext = grid.extent
    isob = s.get("isobaths_full" if region == "full" else "isobaths_zoom") or []

    ims = []
    for ax, field in ((ax1, model), (ax2, sat)):
        ax.set_facecolor(LAND)
        im = ax.imshow(np.ma.masked_invalid(field), origin="lower", extent=ext,
                       cmap=cmap, vmin=lim["vmin"], vmax=lim["vmax"],
                       aspect="auto", interpolation="nearest")
        ims.append(im)
        if isob and depth is not None:
            ax.contour(grid.lon, grid.lat, depth, levels=[float(v) for v in isob],
                       colors="k", linewidths=0.5, alpha=0.6)
        ax.set_xlim(box["lon_min"], box["lon_max"])
        ax.set_ylim(box["lat_min"], box["lat_max"])
        ax.xaxis.set_major_locator(MaxNLocator(6 if region == "full" else 5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(6 if region == "full" else 5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
        ax.tick_params(labelsize=9)
    ax2.tick_params(labelleft=False)

    if with_cur and arrows is not None:
        ga, u, v = arrows
        cur = cfg.currents
        scale = float(cur["scale_full" if region == "full" else "scale_zoom"])
        X, Y = np.meshgrid(ga.lon, ga.lat)
        spd = np.hypot(u, v)
        m = np.isfinite(spd) & (spd > float(cur.get("min_speed", 0.0)))
        if m.any():
            q = ax1.quiver(X[m], Y[m], u[m], v[m], angles="xy", scale_units="xy",
                           scale=scale, width=float(cur.get("width", 0.0014)),
                           color=cur.get("color", "k"), pivot="mid", zorder=5)
            ref = float(cur.get("ref_speed", 0.5))
            ax1.quiverkey(q, 0.88, 1.035, ref, f"{ref:g} m/s", labelpos="E",
                          coordinates="axes", fontproperties={"size": 9})

    ax1.set_title("SCHISM surface temperature"
                  + (" + surface currents" if with_cur else ""), fontsize=11)
    ax2.set_title("ACSPO L3S-LEO-DY SST", fontsize=11)
    cb = fig.colorbar(ims[0], cax=cax, extend="both")
    cb.set_label("SST (°C)", fontsize=10)
    fig.text(0.5, 1 - 0.18 / Hi, f"{d:%Y-%m-%d}", ha="center", va="top",
             fontsize=14, fontweight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.jpg")
    fig.savefig(str(tmp), dpi=dpi, format="jpeg",
                pil_kwargs={"quality": int(s.get("jpeg_quality", 90))})
    plt.close(fig)
    tmp.replace(out_path)
    return out_path


def frame_path(cfg, ptype, d: date) -> Path:
    return cfg.frames_dir(ptype) / f"{ptype}_{d:%Y%m%d}.jpg"
