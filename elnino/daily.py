"""Per-day task: regrid model, load satellite, write stats and frames."""

from datetime import date

import numpy as np

from . import model, plotting, stats
from .config import PLOT_TYPES, Config
from .prepare import load_cache
from .satellite import load_day


def stats_path(cfg: Config, d: date):
    return cfg.stats_dir / f"stats_{d:%Y%m%d}.npz"


def compute_stats(cfg, d, grids, mfull, sfull, mzoom, szoom, coast_mask):
    """Dict of arrays/scalars written to stats_YYYYMMDD.npz."""
    out = {}
    for name, box in cfg.regions.items():
        mm, ss = stats.region_means(mfull, sfull, grids["full"], box)
        out[f"{name}_model"], out[f"{name}_sat"] = mm, ss
    out["skill_full"] = np.array(stats.skill(mfull, sfull, grids["full"]), dtype="float64")
    out["skill_zoom"] = np.array(stats.skill(mzoom, szoom, grids["zoom"]), dtype="float64")
    eq = cfg.eq
    out["eq_model"], out["eq_sat"] = stats.lat_band_profile(
        mfull, sfull, grids["full"], eq["lat_min"], eq["lat_max"])
    out["coast_model"], out["coast_sat"] = stats.coast_profile(mfull, sfull, coast_mask)
    return out


def run_day(cfg: Config, d: date, force: bool = False, types=None):
    """Process one day. Returns True if anything was written."""
    types = list(types or PLOT_TYPES)
    todo = [t for t in types if force or not plotting.frame_path(cfg, t, d).exists()]
    need_stats = force or not stats_path(cfg, d).exists()
    if not todo and not need_stats:
        print(f"{d:%Y%m%d}: all outputs exist, skipping.", flush=True)
        return False

    sat = {r: load_day(cfg, r, d) for r in ("full", "zoom")}
    if sat["full"] is None or sat["zoom"] is None:
        print(f"{d:%Y%m%d}: no satellite data, skipping.", flush=True)
        return False

    W, depth, coast_mask = load_cache(cfg)
    grids = {n: cfg.grid(n) for n in ("full", "zoom")}
    match = cfg.sst.get("match", "daily_mean")

    sst_nodal, desc = model.surface_daily_mean(cfg, "temperature", "temperature", d, match)
    if sst_nodal is None:
        print(f"{d:%Y%m%d}: no model SST ({desc}), skipping.", flush=True)
        return False
    mod = {n: W[n].apply(sst_nodal) for n in ("full", "zoom")}
    print(f"{d:%Y%m%d}: model SST {desc}", flush=True)

    if need_stats:
        cfg.stats_dir.mkdir(parents=True, exist_ok=True)
        s = compute_stats(cfg, d, grids, mod["full"], sat["full"], mod["zoom"],
                          sat["zoom"], coast_mask)
        tmp = stats_path(cfg, d).with_suffix(".tmp.npz")
        np.savez(tmp, **s)
        tmp.replace(stats_path(cfg, d))

    arrows = {}
    if any(PLOT_TYPES[t][1] for t in todo):
        u_n, du = model.surface_daily_mean(cfg, "horizontalVelX", "horizontalVelX", d, match)
        v_n, dv = model.surface_daily_mean(cfg, "horizontalVelY", "horizontalVelY", d, match)
        if u_n is None or v_n is None:
            print(f"{d:%Y%m%d}: no model currents ({du}; {dv}); skipping current frames.",
                  flush=True)
            todo = [t for t in todo if not PLOT_TYPES[t][1]]
        else:
            for region in ("full", "zoom"):
                k = int(round(cfg.grid("arrows_" + region).dx / grids[region].dx))
                u = stats.block_mean(W[region].apply(u_n), k)
                v = stats.block_mean(W[region].apply(v_n), k)
                ga = cfg.grid("arrows_" + region)
                if u.shape != ga.shape:
                    raise RuntimeError(f"arrow grid {ga.shape} != block-mean shape {u.shape}")
                arrows[region] = (ga, u, v)

    for t in todo:
        region, with_cur = PLOT_TYPES[t]
        plotting.render_frame(
            cfg, t, d, grids[region], mod[region], sat[region],
            depth=depth[region],
            arrows=arrows.get(region) if with_cur else None,
            out_path=plotting.frame_path(cfg, t, d))
        print(f"{d:%Y%m%d}: wrote {t}", flush=True)
    return True
