"""One-off preparation: mesh -> regular-grid interpolation weights and masks.

Writes under <out_dir>/cache/:
    weights_full.npz, weights_zoom.npz   barycentric weights per plotting grid
    grid_fields.npz                      bathymetry on each grid + coastal band mask
Run once (SLURM job "prepare") before the per-day jobs.
"""

import time
from pathlib import Path

import numpy as np

from . import mesh, stats
from .config import Config


def weights_path(cfg: Config, name: str) -> Path:
    return cfg.cache_dir / f"weights_{name}.npz"


def fields_path(cfg: Config) -> Path:
    return cfg.cache_dir / "grid_fields.npz"


def check_arrow_ratios(cfg: Config):
    for fine, coarse in (("full", "arrows_full"), ("zoom", "arrows_zoom")):
        gf, gc = cfg.grid(fine), cfg.grid(coarse)
        ratio = gc.dx / gf.dx
        if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
            raise ValueError(
                f"arrow spacing {gc.dx} must be an integer multiple of the "
                f"{fine} grid spacing {gf.dx}")


def grid_signature(cfg: Config) -> str:
    """Identifies the plotting grids; the cache is only valid for the grids it was built for."""
    parts = []
    for n in ("full", "zoom"):
        g = cfg.grid(n)
        parts.append(f"{n}:{g.shape}:{g.extent[0]:.4f},{g.extent[1]:.4f},{g.extent[2]:.4f},{g.extent[3]:.4f}")
    return "|".join(parts)


def run_prepare(cfg: Config):
    check_arrow_ratios(cfg)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    src = cfg.outputs_dir / f"out2d_{cfg.stack_for_day(cfg.plot_start)}.nc"
    if not src.exists():
        raise FileNotFoundError(src)
    t0 = time.time()
    print(f"Loading mesh from {src}", flush=True)
    x, y, depth, tris = mesh.load_mesh(src)
    print(f"  {x.size} nodes, {tris.shape[0]} triangles "
          f"(after splitting quads)  [{time.time() - t0:.0f}s]", flush=True)
    print(f"  lon {x.min():.3f}..{x.max():.3f}  lat {y.min():.3f}..{y.max():.3f}", flush=True)

    finder = mesh.build_trifinder(x, y, tris)
    print(f"  trifinder built [{time.time() - t0:.0f}s]", flush=True)

    dgrid = {}
    for name in ("full", "zoom"):
        g = cfg.grid(name)
        w = mesh.build_weights(finder, x, y, tris, g)
        w.save(weights_path(cfg, name))
        dgrid[name] = w.apply(depth)
        print(f"  weights '{name}' {g.shape}: {100 * w.valid.mean():.1f}% of cells inside mesh "
              f"[{time.time() - t0:.0f}s]", flush=True)

    mask = stats.coast_band_mask(dgrid["full"], cfg.grid("full"), cfg.coast)
    print(f"  coastal band: {int(mask.any(axis=1).sum())} latitude rows, "
          f"{int(mask.sum())} cells", flush=True)
    np.savez(fields_path(cfg), depth_full=dgrid["full"], depth_zoom=dgrid["zoom"],
             coast_mask=mask, signature=np.array(grid_signature(cfg)))
    print(f"Prepare complete -> {cfg.cache_dir}", flush=True)


def load_cache(cfg: Config):
    """Return (weights dict, depth dict, coast_mask) from the cache."""
    z = np.load(fields_path(cfg))
    if "signature" not in z.files or str(z["signature"]) != grid_signature(cfg):
        raise RuntimeError("the cache in " + str(cfg.cache_dir) + " was built for different "
                           "grids (domain/zoom/grid_dx changed): re-run `prepare`.")
    W = {n: mesh.Weights.load(weights_path(cfg, n)) for n in ("full", "zoom")}
    return W, {"full": z["depth_full"], "zoom": z["depth_zoom"]}, z["coast_mask"]
