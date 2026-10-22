"""Read SCHISM standalone (new I/O) output: surface layer, daily mean."""

from datetime import date
from pathlib import Path

import numpy as np

FILL_HI = 1.0e30
FILL_LO = -9990.0


def stack_path(cfg, prefix: str, stack: int) -> Path:
    return cfg.outputs_dir / f"{prefix}_{stack}.nc"


def _clean(a):
    a = np.asarray(a, dtype="float32")
    return np.where((a > FILL_HI) | (a < FILL_LO), np.nan, a)


def _time_indices(times, d: date, match: str):
    """Indices of the hourly outputs belonging to calendar day d.

    Stack N holds 01:00 ... 24:00 of model day N-1, so the day's samples are
    (d 00:00, d+1 00:00]. match='noon' keeps only the output nearest 12Z.
    """
    day0 = np.datetime64(d.isoformat())
    day1 = day0 + np.timedelta64(1, "D")
    sel = np.flatnonzero((times > day0) & (times <= day1))
    if match == "noon" and sel.size:
        target = day0 + np.timedelta64(12, "h")
        sel = sel[[np.argmin(np.abs(times[sel] - target))]]
    return sel


def surface_daily_mean(cfg, prefix: str, var: str, d: date, match: str = "daily_mean"):
    """Daily mean of the surface layer of `var` for day d, as a nodal array.

    Reads one timestep and the top layer at a time so memory stays at one
    (nnode, nlayer) slab. Returns (values, description) or (None, reason).
    """
    import xarray as xr
    path = stack_path(cfg, prefix, cfg.stack_for_day(d))
    if not path.exists():
        return None, f"missing {path.name}"
    ds = xr.open_dataset(str(path), decode_times=True)
    try:
        if var not in ds:
            return None, f"{var} not in {path.name}"
        sel = _time_indices(ds["time"].values, d, match)
        if sel.size == 0:
            return None, f"no timesteps for {d} in {path.name}"
        da = ds[var]
        lay = next(dim for dim in da.dims if "layer" in dim.lower() or "vgrid" in dim.lower())
        acc, n = None, 0
        for i in sel:
            v = _clean(da.isel(time=int(i), **{lay: -1}).values)
            if acc is None:
                acc = np.zeros(v.shape, dtype="float64")
                cnt = np.zeros(v.shape, dtype="int16")
            ok = np.isfinite(v)
            acc[ok] += v[ok]
            cnt += ok
            n += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype("float32")
        return mean, f"{match}, {n} step(s) of {path.name}"
    finally:
        ds.close()
