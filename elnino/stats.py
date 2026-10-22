"""Daily statistics derived from the regridded model / satellite fields."""

import numpy as np


def common_mask(model, sat):
    return np.isfinite(model) & np.isfinite(sat)


def box_mask(grid, box):
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    return ((lon2d >= box["lon_min"]) & (lon2d <= box["lon_max"])
            & (lat2d >= box["lat_min"]) & (lat2d <= box["lat_max"]))


def area_weights(grid):
    return np.repeat(np.cos(np.deg2rad(grid.lat))[:, None], grid.lon.size, axis=1)


def wmean(a, w, m):
    if not m.any():
        return np.nan
    return float((a[m] * w[m]).sum() / w[m].sum())


def region_means(model, sat, grid, box):
    """Area-weighted means of model and satellite over common-valid cells in box."""
    m = common_mask(model, sat) & box_mask(grid, box)
    w = area_weights(grid)
    return wmean(model, w, m), wmean(sat, w, m)


def skill(model, sat, grid, mask=None):
    """(bias, rmse, corr, n) of model - sat over common-valid cells (area weighted)."""
    m = common_mask(model, sat)
    if mask is not None:
        m &= mask
    n = int(m.sum())
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    w = area_weights(grid)[m]
    a, b = model[m].astype("float64"), sat[m].astype("float64")
    ws = w.sum()
    diff = a - b
    bias = float((diff * w).sum() / ws)
    rmse = float(np.sqrt((diff ** 2 * w).sum() / ws))
    ma, mb = (a * w).sum() / ws, (b * w).sum() / ws
    cov = ((a - ma) * (b - mb) * w).sum() / ws
    va = ((a - ma) ** 2 * w).sum() / ws
    vb = ((b - mb) ** 2 * w).sum() / ws
    corr = float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan
    return (bias, rmse, corr, n)


def lat_band_profile(model, sat, grid, lat_min, lat_max):
    """Mean over a latitude band, per longitude (common-valid cells only)."""
    rows = (grid.lat >= lat_min) & (grid.lat <= lat_max)
    mo, so = model[rows], sat[rows]
    m = common_mask(mo, so)
    cnt = m.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = np.where(cnt > 0, np.where(m, mo, 0).sum(axis=0) / cnt, np.nan)
        ps = np.where(cnt > 0, np.where(m, so, 0).sum(axis=0) / cnt, np.nan)
    return pm.astype("float32"), ps.astype("float32")


def coast_band_mask(depth, grid, cfg_coast):
    """Boolean (ny, nx) mask of the ocean band just offshore of the coast.

    The coast at each latitude is the first land cell met scanning eastward
    from search_lon_min whose next 1.5 deg are mostly land, which skips small
    offshore islands. Land = outside the mesh or non-positive depth.
    """
    land = ~np.isfinite(depth) | (depth <= 0)
    ny, nx = grid.shape
    out = np.zeros((ny, nx), dtype=bool)
    k = max(1, int(round(1.5 / grid.dx)))
    nb = max(1, int(round(cfg_coast["band_width_deg"] / grid.dx)))
    lon_ok = (grid.lon >= cfg_coast["search_lon_min"]) & (grid.lon <= cfg_coast["search_lon_max"])
    cols = np.flatnonzero(lon_ok)
    for j in np.flatnonzero((grid.lat >= cfg_coast["lat_min"]) & (grid.lat <= cfg_coast["lat_max"])):
        row = land[j]
        for i in cols:
            if row[i] and row[i:i + k].mean() >= 0.8:
                lo = max(cols[0], i - nb)
                band = ~row[lo:i]
                out[j, lo:i] = band
                break
    return out


def coast_profile(model, sat, mask):
    """Mean SST in the coastal band for every row of the grid (NaN if empty)."""
    m = mask & common_mask(model, sat)
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = np.where(cnt > 0, np.where(m, model, 0).sum(axis=1) / cnt, np.nan)
        ps = np.where(cnt > 0, np.where(m, sat, 0).sum(axis=1) / cnt, np.nan)
    return pm.astype("float32"), ps.astype("float32")


def block_mean(a, k, min_frac=0.5):
    """NaN-aware k x k block mean; the array is padded so any shape works.

    Output shape is (ceil(ny/k), ceil(nx/k)). A block is valid when at least
    min_frac of its real (non-padding) cells are finite.
    """
    ny, nx = a.shape
    py, px = (-ny) % k, (-nx) % k
    real = np.ones((ny, nx), dtype=bool)
    if py or px:
        a = np.pad(a, ((0, py), (0, px)), constant_values=np.nan)
        real = np.pad(real, ((0, py), (0, px)), constant_values=False)
    NY, NX = a.shape
    b = a.reshape(NY // k, k, NX // k, k)
    good = np.isfinite(b)
    avail = real.reshape(NY // k, k, NX // k, k).sum(axis=(1, 3))
    cnt = good.sum(axis=(1, 3))
    s = np.where(good, b, 0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt >= min_frac * avail, s / np.maximum(cnt, 1), np.nan)
    return out.astype("float32")
