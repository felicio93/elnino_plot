"""Configuration loading and the regular grids shared by every stage."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

# Plot types: name -> (region, with_currents)
PLOT_TYPES = {
    "full":     ("full", False),
    "zoom":     ("zoom", False),
    "full_cur": ("full", True),
    "zoom_cur": ("zoom", True),
}


@dataclass(frozen=True)
class Grid:
    """Regular lon/lat grid described by cell centres."""
    name: str
    lon: np.ndarray
    lat: np.ndarray
    dx: float

    @property
    def shape(self):
        return (self.lat.size, self.lon.size)

    @property
    def lon2d(self):
        return np.meshgrid(self.lon, self.lat)[0]

    @property
    def lat2d(self):
        return np.meshgrid(self.lon, self.lat)[1]

    @property
    def extent(self):
        h = self.dx / 2
        return (self.lon[0] - h, self.lon[-1] + h,
                self.lat[0] - h, self.lat[-1] + h)


def make_grid(name, box, dx):
    """Cell-centred grid covering box = {lon_min, lon_max, lat_min, lat_max}."""
    nx = int(round((box["lon_max"] - box["lon_min"]) / dx))
    ny = int(round((box["lat_max"] - box["lat_min"]) / dx))
    lon = box["lon_min"] + dx * (np.arange(nx) + 0.5)
    lat = box["lat_min"] + dx * (np.arange(ny) + 0.5)
    return Grid(name, lon, lat, float(dx))


def _to_date(v):
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


class Config:
    """Thin wrapper over the YAML dict with resolved paths and grids."""

    def __init__(self, raw: dict, config_path: Path = None):
        self.raw = raw
        self.config_path = Path(config_path).resolve() if config_path else None
        self.outputs_dir = Path(raw["outputs_dir"])
        self.out_dir = Path(raw["out_dir"])
        self.model_start = _to_date(raw["model_start_date"])
        self.plot_start = _to_date(raw["plot_start"])
        self.plot_end = _to_date(raw["plot_end"])
        self.domain = dict(raw["domain"])
        self.zoom = dict(raw["zoom"])
        self.sst = raw["sst"]
        self.currents = raw["currents"]
        self.regions = raw["regions"]
        self.eq = raw["equator_hovmoller"]
        self.coast = raw["coast_hovmoller"]
        self.anim = raw["animation"]
        self.sat = raw["satellite"]
        self.slurm = raw["slurm"]
        self.python = raw["python"]
        repo = raw.get("repo_dir")
        self.repo_dir = (Path(repo) if repo
                         else Path(__file__).resolve().parent.parent)

    # -- directories ---------------------------------------------------
    def sub(self, *parts) -> Path:
        return self.out_dir.joinpath(*parts)

    @property
    def obs_dir(self):    return self.sub("obs", "sst_leo")
    @property
    def cache_dir(self):  return self.sub("cache")
    @property
    def stats_dir(self):  return self.sub("stats")
    @property
    def fig_dir(self):    return self.sub("figures")
    @property
    def anim_dir(self):   return self.sub("animations")
    @property
    def log_dir(self):    return self.sub("logs")

    def frames_dir(self, ptype: str) -> Path:
        return self.sub("frames", ptype)

    # -- dates -------------------------------------------------------------
    def days(self):
        out, d = [], self.plot_start
        while d <= self.plot_end:
            out.append(d)
            d += timedelta(days=1)
        return out

    def stack_for_day(self, d: date) -> int:
        """Stack number holding model day d (stack N = start+N-1 d, 01:00..24:00)."""
        return (d - self.model_start).days + 1

    # -- grids -------------------------------------------------------------------
    def grid(self, name: str) -> Grid:
        """name: full | zoom | arrows_full | arrows_zoom"""
        if name == "full":
            return make_grid(name, self.domain, self.raw["grid_dx_full"])
        if name == "zoom":
            return make_grid(name, self.zoom, self.raw["grid_dx_zoom"])
        if name == "arrows_full":
            return make_grid(name, self.domain, self.currents["dx_full"])
        if name == "arrows_zoom":
            return make_grid(name, self.zoom, self.currents["dx_zoom"])
        raise KeyError(name)

    def all_grid_names(self):
        return ["full", "zoom", "arrows_full", "arrows_zoom"]


def load_config(path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw, path)
