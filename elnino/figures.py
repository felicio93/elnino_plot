"""Summary figures built from the per-day stats files."""

from datetime import timedelta

import numpy as np

from .config import Config

MODEL_C, SAT_C = "#d62728", "k"


def load_stats(cfg: Config):
    """Return (dates, dict name -> array over all plot days; NaN where missing)."""
    days = cfg.days()
    first = None
    series = {}
    for i, d in enumerate(days):
        f = cfg.stats_dir / f"stats_{d:%Y%m%d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if first is None:
            first = {k: z[k] for k in z.files}
            for k, v in first.items():
                series[k] = np.full((len(days),) + np.shape(v), np.nan, dtype="float64")
        for k in z.files:
            series[k][i] = z[k]
    return days, series


def _dates(days):
    return np.array([np.datetime64(d.isoformat()) for d in days])


def write_csv(cfg, days, S):
    cols = ["date"]
    data = []
    for name in cfg.regions:
        cols += [f"{name}_model", f"{name}_sat"]
        data += [S[f"{name}_model"], S[f"{name}_sat"]]
    for reg in ("full", "zoom"):
        cols += [f"{reg}_bias", f"{reg}_rmse", f"{reg}_corr", f"{reg}_ncells"]
        data += [S[f"skill_{reg}"][:, i] for i in range(4)]
    out = cfg.fig_dir / "daily_stats.csv"
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for i, d in enumerate(days):
            f.write(f"{d.isoformat()}," + ",".join(
                "" if not np.isfinite(c[i]) else f"{c[i]:.4f}" for c in data) + "\n")
    return out


def nino_figure(cfg, days, S, plt):
    names = [("nino34", "Niño 3.4  (5°S–5°N, 170°W–120°W)"),
             ("nino3", "Niño 3  (5°S–5°N, 150°W–90°W)"),
             ("nino12", "Niño 1+2  (10°S–0°, 90°W–80°W)")]
    names = [n for n in names if n[0] in cfg.regions and f"{n[0]}_model" in S]
    t = _dates(days)
    nb = int(cfg_fig(cfg).get("baseline_days", 7))
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 3.2 * len(names)), sharex=True,
                             squeeze=False)
    for r, (key, title) in enumerate(names):
        mo, sa = S[f"{key}_model"], S[f"{key}_sat"]
        ax = axes[r, 0]
        ax.plot(t, sa, color=SAT_C, lw=1.6, label="ACSPO L3S-LEO-DY")
        ax.plot(t, mo, color=MODEL_C, lw=1.6, label="SCHISM")
        ax.set_ylabel("SST (°C)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        ax = axes[r, 1]
        for series, c, lab in ((sa, SAT_C, "ACSPO"), (mo, MODEL_C, "SCHISM")):
            base = np.nanmean(series[:nb])
            ax.plot(t, series - base, color=c, lw=1.6, label=lab)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_ylabel("ΔSST (°C)")
        ax.set_title(f"change since mean of first {nb} days", fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        if r == 0:
            axes[0, 0].legend(loc="best", fontsize=9)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("El Niño regions: area-mean SST, model vs satellite (2023)", fontsize=13)
    fig.tight_layout()
    return fig


def cfg_fig(cfg):
    return cfg.raw.get("figures", {})


def hovmoller(cfg, plt, days, mo, sa, x, xlabel, vlim, title, xfmt, time_on_y):
    """Two-panel Hovmoller (model | satellite) with one shared colourbar."""
    t = _dates(days)
    f = cfg_fig(cfg)
    cmap = plt.get_cmap(f.get("cmap", "jet")).with_extremes(bad="0.85")
    if time_on_y:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        mk = lambda a: np.ma.masked_invalid(a)                      # (time, x)
        mesh = [ax.pcolormesh(x, t, mk(d), cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                              shading="nearest", rasterized=True)
                for ax, d in zip(axes, (mo, sa))]
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.xaxis.set_major_formatter(xfmt)
        axes[0].set_ylabel("Date")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        mesh = [ax.pcolormesh(t, x, np.ma.masked_invalid(d.T), cmap=cmap, vmin=vlim[0],
                              vmax=vlim[1], shading="nearest", rasterized=True)
                for ax, d in zip(axes, (mo, sa))]
        for ax in axes:
            ax.set_ylabel(xlabel)
            ax.yaxis.set_major_formatter(xfmt)
        axes[-1].tick_params(axis="x", rotation=30)
    axes[0].set_title("SCHISM", fontsize=11)
    axes[1].set_title("ACSPO L3S-LEO-DY", fontsize=11)
    fig.suptitle(title, fontsize=13)
    cb = fig.colorbar(mesh[0], ax=list(axes), extend="both", pad=0.02, fraction=0.04)
    cb.set_label("SST (°C)")
    return fig


def skill_figure(cfg, days, S, plt):
    t = _dates(days)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    labels = [("Bias (model − satellite, °C)", 0), ("RMSE (°C)", 1), ("Correlation", 2)]
    for c, (reg, title) in enumerate((("full", "Full domain"), ("zoom", "California zoom box"))):
        for r, (lab, idx) in enumerate(labels):
            ax = axes[r, c]
            ax.plot(t, S[f"skill_{reg}"][:, idx], color=MODEL_C, lw=1.4)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.axhline(0, color="0.5", lw=0.8)
            if r == 0:
                ax.set_title(title, fontsize=11)
        axes[-1, c].tick_params(axis="x", rotation=30)
    fig.suptitle("SCHISM vs ACSPO L3S-LEO-DY daily SST skill (area-weighted)", fontsize=13)
    fig.tight_layout()
    return fig


def run_figures(cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from .plotting import fmt_lat, fmt_lon

    days, S = load_stats(cfg)
    if not S:
        print("figures: no stats files found; run the per-day jobs first.")
        return
    have = int(np.isfinite(S["skill_full"][:, 0]).sum())
    print(f"figures: {have}/{len(days)} days with stats", flush=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg_fig(cfg).get("dpi", 150))
    f = cfg_fig(cfg)

    def save(fig, name):
        out = cfg.fig_dir / name
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  wrote {out}", flush=True)

    print(f"  wrote {write_csv(cfg, days, S)}", flush=True)
    save(nino_figure(cfg, days, S, plt), "nino_indices.png")
    gfull = cfg.grid("full")
    save(hovmoller(cfg, plt, days, S["eq_model"], S["eq_sat"], gfull.lon, "Longitude",
                   f.get("eq_vlim", [20, 31]),
                   f"Equatorial SST Hovmöller ({cfg.eq['lat_min']:g}° to {cfg.eq['lat_max']:g}° mean)",
                   FuncFormatter(fmt_lon), True), "hovmoller_equator.png")
    rows = np.isfinite(S["coast_model"]).any(axis=0) | np.isfinite(S["coast_sat"]).any(axis=0)
    save(hovmoller(cfg, plt, days, S["coast_model"][:, rows], S["coast_sat"][:, rows],
                   gfull.lat[rows], "Latitude", f.get("coast_vlim", [10, 26]),
                   f"US West Coast SST Hovmöller (coastal band, {cfg.coast['band_width_deg']:g}° wide)",
                   FuncFormatter(fmt_lat), False), "hovmoller_coast.png")
    save(skill_figure(cfg, days, S, plt), "skill_timeseries.png")
