"""Build a small fake SCHISM run + satellite files for end-to-end tests."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from elnino.config import Config
from elnino.satellite import _write


def true_sst(lon, lat, doy):
    """Smooth 'El Nino-like' SST: warm pool, cold tongue, seasonal wobble."""
    base = 29.0 - 0.012 * (lat - 0.0) ** 2 / 1.0 * 0.6 - 0.15 * np.maximum(lat - 25, 0)
    tongue = -4.0 * np.exp(-((lat) / 6.0) ** 2) * np.exp(-((lon - 250) / 35.0) ** 2)
    return base + tongue + 1.5 * np.sin(2 * np.pi * (doy - 80) / 365) * lat / 60


def coast_lon(lat):
    return 245.0 - 0.3 * (lat - 30.0)


def make_mesh(dx=0.5):
    lon = np.arange(93.0, 290.0 + 1e-9, dx)
    lat = np.arange(-33.0, 67.0 + 1e-9, dx)
    LON, LAT = np.meshgrid(lon, lat)
    nlat, nlon = LON.shape
    x, y = LON.ravel(), LAT.ravel()
    land_node = (LON > coast_lon(LAT)) & (LAT > 20) & (LAT < 55)
    depth = np.where(land_node, -5.0, 100.0 + 40.0 * (coast_lon(LAT) - LON).clip(0)).ravel()
    n = lambda j, i: j * nlon + i
    faces = []
    for j in range(nlat - 1):
        for i in range(nlon - 1):
            q = [n(j, i), n(j, i + 1), n(j + 1, i + 1), n(j + 1, i)]
            if depth[q].min() <= 0:
                continue
            if (i + j) % 3 == 0:
                faces.append([q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1])      # quad
            else:
                faces.append([q[0] + 1, q[1] + 1, q[2] + 1, np.nan])        # triangle
                faces.append([q[0] + 1, q[2] + 1, q[3] + 1, np.nan])
    return x, y, depth.astype("float32"), np.array(faces, dtype="float64")


def make_run(cfg_raw, tmp: Path, days, nlayer=4):
    """Write out2d/temperature/horizontalVel{X,Y} stacks for the given days."""
    outputs = tmp / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    x, y, depth, faces = make_mesh()
    start = date.fromisoformat(str(cfg_raw["model_start_date"]))
    rng = np.random.default_rng(1)
    for d in days:
        N = (d - start).days + 1
        t = (N - 1) * 86400.0 + 3600.0 * np.arange(1, 25)
        tunits = "seconds since 2022-12-01T00:00:00+00"
        doy = d.timetuple().tm_yday
        sst = true_sst(x, y, doy) + 0.5 + 0.2 * np.sin(x / 7.0)
        sst = np.where(depth > 0, sst, np.nan).astype("float32")
        T = np.full((24, x.size, nlayer), 9.96921e36, dtype="float32")
        T[:, :, -1] = sst[None] + 0.3 * np.sin(np.arange(24) / 24 * 2 * np.pi)[:, None]
        T[:, :, 0] = sst[None] - 5
        U = np.zeros_like(T); V = np.zeros_like(T)
        U[:, :, -1] = 0.4 * np.cos(np.deg2rad(y))[None]
        V[:, :, -1] = 0.2 * np.sin(np.deg2rad(x) * 3)[None]
        dims = ("time", "nSCHISM_hgrid_node", "nSCHISM_vgrid_layers")
        for name, arr in (("temperature", T), ("horizontalVelX", U), ("horizontalVelY", V)):
            ds = xr.Dataset({name: (dims, arr)}, coords={"time": ("time", t, {"units": tunits})})
            ds[name].attrs["missing_value"] = np.float32(9.96921e36)
            ds.to_netcdf(outputs / f"{name}_{N}.nc")
        if not (outputs / f"out2d_{N}.nc").exists():
            m = xr.Dataset({
                "SCHISM_hgrid_node_x": (("nSCHISM_hgrid_node",), x),
                "SCHISM_hgrid_node_y": (("nSCHISM_hgrid_node",), y),
                "SCHISM_hgrid_face_nodes": (("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
                                            np.where(np.isnan(faces), -1, faces).astype("int32")),
                "depth": (("nSCHISM_hgrid_node",), depth)})
            m["SCHISM_hgrid_face_nodes"].attrs["_FillValue"] = np.int32(-1)
            m["SCHISM_hgrid_face_nodes"].attrs["start_index"] = 1
            m.to_netcdf(outputs / f"out2d_{N}.nc")
    return outputs


def make_satellite(cfg: Config, days):
    cfg.obs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    for d in days:
        for region in ("full", "zoom"):
            g = cfg.grid(region)
            LON, LAT = np.meshgrid(g.lon, g.lat)
            sst = true_sst(LON, LAT, d.timetuple().tm_yday) + rng.normal(0, 0.2, LON.shape)
            sst = np.where((LON > coast_lon(LAT)) & (LAT > 20) & (LAT < 55), np.nan, sst)
            sst[rng.random(sst.shape) < 0.05] = np.nan                       # cloud gaps
            _write(cfg.obs_dir / f"leosst_{region}_{d:%Y%m%d}.nc", g, sst)
