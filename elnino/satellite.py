"""NOAA/STAR ACSPO L3S-LEO-DY satellite SST: download (DTN) and load.

Two sources (config satellite.source): "erddap" (default; server-side subsetting
of dataset noaacwLEOACSPOSSTL3SCDaily, ~100 MB/day) or "files" (global 433 MB files).
Both return identical values.

Each global daily file (0.02 deg, -180..180 lon, Kelvin) is cropped and
converted to degC on the plotting grids, and two small files are written:

    obs/sst_leo/leosst_full_YYYYMMDD.nc   domain, binned to grid_dx_full
    obs/sst_leo/leosst_zoom_YYYYMMDD.nc   zoom box, binned to grid_dx_zoom

Both files hold lon/lat (cell centres, 0-360) and sst (degC). Days already
present are skipped, so the download is resume-safe.
"""

import argparse
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from .config import load_config
from .satellite_grid import bin_to_grid

VAR = "sea_surface_temperature"

# The netCDF4/HDF5 library is not thread-safe: downloads run in parallel but
# every read/write of a netCDF file is serialized through this lock.
_HDF_LOCK = threading.Lock()

RATE_LIMIT_TRIES = 12      # max attempts for a request that keeps getting HTTP 429/503 (file server)
ERDDAP_RATE_TRIES = 2      # ERDDAP: give up quickly on 429 and switch to the file route
_state = {"rate_limited": 0, "erddap_block_until": 0.0}


def sat_file(cfg, region: str, d: date) -> Path:
    return cfg.obs_dir / f"leosst_{region}_{d:%Y%m%d}.nc"


def day_url(cfg, d: date, template: str) -> str:
    base = cfg.sat["base_url"].rstrip("/")
    return f"{base}/{d.year}/{d:%j}/{template.format(ymd=f'{d:%Y%m%d}')}"


def _fetch(url: str, dest: Path, retries: int, resume: bool = True,
           rate_tries: int = None) -> bool:
    """Download url to dest. With resume, a partial file is continued between attempts
    (file server only; ERDDAP responses are generated on the fly and always restart)."""
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            if not resume:
                dest.unlink(missing_ok=True)
            have = dest.stat().st_size if dest.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                resumed = have > 0 and r.status == 206
                size = int(r.headers.get("Content-Length", -1))
                with open(dest, "ab" if resumed else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            expect = size + (have if resumed else 0)
            if size >= 0 and dest.stat().st_size != expect:
                raise IOError(f"truncated download: {dest.stat().st_size} of {expect} bytes")
            return dest.stat().st_size > 0
        except Exception as exc:                      # 404, timeouts, resets
            code = getattr(exc, "code", None)
            if resume and code == 416:                # range past EOF: already complete
                return dest.exists() and dest.stat().st_size > 0
            detail = ""
            if hasattr(exc, "read"):
                try:
                    detail = exc.read(300).decode("utf-8", "replace").replace("\n", " ")
                except Exception:
                    pass
            print(f"    fetch attempt {attempt}/{retries} failed: {exc!r} {detail}\n"
                  f"      {url[:140]}", flush=True)
            if code == 404:
                dest.unlink(missing_ok=True)
                return False
            if code in (429, 503):                    # rate limited / busy: back off, don't give up early
                ra = getattr(exc, "headers", None) and exc.headers.get("Retry-After")
                wait = int(ra) if ra and str(ra).isdigit() else min(600, 60 * attempt)
                wait = min(600, wait) * (0.8 + 0.4 * random.random())
                _state["rate_limited"] += 1
                if attempt >= (rate_tries or RATE_LIMIT_TRIES):
                    break
                print(f"      rate limited; waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                retries = max(retries, attempt + 1)   # waiting is not a failure: keep trying
                continue
            if attempt < retries:
                time.sleep(5 * attempt)
    dest.unlink(missing_ok=True)
    return False


def _write(path: Path, grid, sst):
    import xarray as xr
    ds = xr.Dataset(
        {"sst": (("lat", "lon"), sst.astype("float32"))},
        coords={"lat": grid.lat.astype("float32"),
                "lon": grid.lon.astype("float32")})
    ds["sst"].attrs.update(units="degC", long_name="sea surface temperature")
    ds["lon"].attrs.update(units="degrees_east")
    ds["lat"].attrs.update(units="degrees_north")
    enc = {"sst": {"zlib": True, "complevel": 4}}
    tmp = path.with_suffix(".tmp")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)


def subset_raw(raw: Path, cfg, out_full: Path, out_zoom: Path):
    """Crop the global file to the full and zoom grids; write degC files."""
    import xarray as xr
    gfull, gzoom = cfg.grid("full"), cfg.grid("zoom")
    ds = xr.open_dataset(str(raw))
    try:
        lat = ds["lat"].values
        lon = ds["lon"].values
        lon360 = np.where(lon < 0, lon + 360.0, lon)
        order = np.argsort(lon360)
        lon_s = lon360[order]
        for grid, out in ((gfull, out_full), (gzoom, out_zoom)):
            h = grid.dx / 2
            lat_i = np.flatnonzero((lat >= grid.lat[0] - h) & (lat <= grid.lat[-1] + h))
            lon_i = np.flatnonzero((lon_s >= grid.lon[0] - h) & (lon_s <= grid.lon[-1] + h))
            if lat_i.size == 0 or lon_i.size == 0:
                raise RuntimeError(f"{grid.name} grid outside the satellite file")
            sub = ds[VAR].isel(time=0, lat=slice(lat_i[0], lat_i[-1] + 1))
            arr = sub.values.astype("float32")                # (lat, lon) K, NaN = missing
            arr = arr[:, order][:, lon_i]
            sst_c = arr - 273.15
            binned = bin_to_grid(lon_s[lon_i], lat[lat_i[0]:lat_i[-1] + 1], sst_c, grid)
            _write(out, grid, binned)
    finally:
        ds.close()


def _erddap_segments(grid, pad=0.05):
    """(lon_lo, lon_hi) segments in -180..180 covering a 0-360 grid, plus lat range."""
    h = grid.dx / 2 + pad
    lo, hi = grid.lon[0] - h, grid.lon[-1] + h
    segs = []
    if lo < 180.0:
        segs.append((max(lo, 0.0) if lo < 0 else lo, min(hi, 179.99)))
    if hi > 180.0:
        segs.append((max(lo, 180.0) - 360.0, hi - 360.0))
    lat_lo = max(grid.lat[0] - h, -89.99)
    lat_hi = min(grid.lat[-1] + h, 89.99)
    return segs, lat_lo, lat_hi


def erddap_url(cfg, d: date, lat_lo, lat_hi, lon_lo, lon_hi) -> str:
    base = cfg.sat.get("erddap_url", "https://coastwatch.noaa.gov/erddap/griddap").rstrip("/")
    ds = cfg.sat.get("erddap_dataset", "noaacwLEOACSPOSSTL3SCDaily")
    q = (f"sea_surface_temperature%5B({d:%Y-%m-%d}T12:00:00Z)%5D"
         f"%5B({lat_hi:.2f}):1:({lat_lo:.2f})%5D%5B({lon_lo:.2f}):1:({lon_hi:.2f})%5D")
    return f"{base}/{ds}.nc?{q}"


def _read_erddap_segment(path: Path):
    """Return (lon360, lat, sst_degC) from one ERDDAP subset file."""
    import xarray as xr
    ds = xr.open_dataset(str(path))
    try:
        v = ds["sea_surface_temperature"]
        arr = v.isel(time=0).values.astype("float32")
        if np.nanmean(arr) > 200.0:                      # Kelvin
            arr = arr - 273.15
        lon = ds["longitude"].values.astype("float64")
        lat = ds["latitude"].values.astype("float64")
    finally:
        ds.close()
    return np.where(lon < 0, lon + 360.0, lon), lat, arr


def download_day_erddap(cfg, d: date) -> str:
    """Download only the needed sub-rectangles from ERDDAP and bin them."""
    out_full, out_zoom = sat_file(cfg, "full", d), sat_file(cfg, "zoom", d)
    tmp_dir = cfg.obs_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    retries = int(cfg.sat.get("retries", 3))
    hits0 = _state["rate_limited"]
    for grid, out in ((cfg.grid("full"), out_full), (cfg.grid("zoom"), out_zoom)):
        segs, lat_lo, lat_hi = _erddap_segments(grid)
        files = []
        try:
            for k, (lo, hi) in enumerate(segs):
                f = tmp_dir / f"erddap_{grid.name}_{d:%Y%m%d}_{k}.nc"
                if not _fetch(erddap_url(cfg, d, lat_lo, lat_hi, lo, hi), f, retries, resume=False,
                               rate_tries=ERDDAP_RATE_TRIES):
                    if _state["rate_limited"] > hits0:
                        cool = float(cfg.sat.get("erddap_cooldown_min", 30))
                        _state["erddap_block_until"] = time.time() + 60 * cool
                        print(f"  ERDDAP is rate limiting this host: using the file server "
                              f"for the next {cool:g} min.", flush=True)
                    return "fail"
                files.append(f)
            with _HDF_LOCK:
                parts = [_read_erddap_segment(f) for f in files]
                lon = np.concatenate([p[0] for p in parts])
                lat = parts[0][1]
                sst = np.concatenate([p[2] for p in parts], axis=1)
                _write(out, grid, bin_to_grid(lon, lat, sst, grid))
        except Exception as exc:
            print(f"  {d:%Y%m%d}: erddap {grid.name} failed: {exc}", flush=True)
            return "fail"
        finally:
            for f in files:
                f.unlink(missing_ok=True)
    return "ok"


def download_day(cfg, d: date) -> str:
    """Download + subset one day. Returns 'ok', 'skip' or 'fail'."""
    out_full, out_zoom = sat_file(cfg, "full", d), sat_file(cfg, "zoom", d)
    if (out_full.exists() and out_full.stat().st_size > 0
            and out_zoom.exists() and out_zoom.stat().st_size > 0):
        return "skip"
    if (str(cfg.sat.get("source", "erddap")).lower() == "erddap"
            and time.time() >= _state["erddap_block_until"]):
        r = download_day_erddap(cfg, d)
        if r == "ok" or not cfg.sat.get("fallback_to_files", True):
            return r
        print(f"  {d:%Y%m%d}: ERDDAP failed, falling back to the global file.", flush=True)
    tmp_dir = cfg.obs_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw = tmp_dir / f"raw_{d:%Y%m%d}.nc"
    got = False
    for tmpl in cfg.sat["file_templates"]:
        if _fetch(day_url(cfg, d, tmpl), raw, int(cfg.sat.get("retries", 3))):
            got = True
            break
    if not got:
        return "fail"
    try:
        with _HDF_LOCK:
            subset_raw(raw, cfg, out_full, out_zoom)
    except Exception as exc:
        print(f"  {d:%Y%m%d}: subset failed: {exc}", flush=True)
        return "fail"
    finally:
        raw.unlink(missing_ok=True)
    return "ok"


def run_download(cfg):
    cfg.obs_dir.mkdir(parents=True, exist_ok=True)
    days = cfg.days()
    workers = int(cfg.sat.get("workers", 4))
    print(f"Satellite SST (ACSPO L3S-LEO-DY): {days[0]} -> {days[-1]}  "
          f"({len(days)} days, {workers} workers)\n  output: {cfg.obs_dir}", flush=True)

    def work(d):
        try:
            r = download_day(cfg, d)
        except Exception as exc:
            print(f"  {d:%Y%m%d}: ERROR {exc}", flush=True)
            r = "fail"
        print(f"  {d:%Y%m%d}: {r}", flush=True)
        return d, r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, days))
    failed = [f"{d:%Y%m%d}" for d, r in results if r == "fail"]
    print(f"\nDone. ok/skip: {len(results) - len(failed)}  failed: {len(failed)}")
    if failed:
        print("  missing days: " + ", ".join(failed))
    return failed


def load_day(cfg, region: str, d: date):
    """Return the (ny, nx) satellite SST array for a day, or None if absent."""
    import xarray as xr
    f = sat_file(cfg, region, d)
    if not (f.exists() and f.stat().st_size > 0):
        return None
    ds = xr.open_dataset(str(f))
    try:
        return ds["sst"].values.astype("float32")
    finally:
        ds.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download satellite SST (run on the DTN)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    failed = run_download(load_config(args.config))
    sys.exit(0)


if __name__ == "__main__":
    main()
