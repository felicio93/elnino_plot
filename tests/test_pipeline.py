from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

from elnino.config import PLOT_TYPES, Config
from elnino import daily, prepare, figures, animate, mesh, stats, satellite_grid
from elnino.config import make_grid

from synthetic import make_run, make_satellite

REPO = Path(__file__).resolve().parent.parent
DAYS = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]


def test_split_quads():
    fn = np.array([[1, 2, 3, np.nan], [1, 2, 3, 4], [4, 5, 6, -1]], dtype=float)
    t = mesh.split_quads(fn)
    assert t.shape == (4, 3)
    assert (t[3] == [0, 2, 3]).all()


def test_bin_and_blockmean():
    g = make_grid("t", {"lon_min": 0, "lon_max": 1, "lat_min": 0, "lat_max": 1}, 0.5)
    lon = np.arange(0.05, 1.0, 0.1); lat = lon.copy()
    f = np.add.outer(lat, lon).astype("float32")
    out = satellite_grid.bin_to_grid(lon, lat, f, g)
    assert out.shape == (2, 2)
    assert abs(out[0, 0] - (0.25 + 0.25)) < 1e-4
    a = np.arange(16, dtype="float32").reshape(4, 4)
    assert stats.block_mean(a, 2)[0, 0] == pytest.approx(2.5)


def test_weights_exact_for_linear_field():
    x = np.array([0., 1., 0., 1.]); y = np.array([0., 0., 1., 1.])
    tris = mesh.split_quads(np.array([[1, 2, 4, 3]], dtype=float))
    g = make_grid("t", {"lon_min": 0, "lon_max": 1, "lat_min": 0, "lat_max": 1}, 0.25)
    w = mesh.build_weights(mesh.build_trifinder(x, y, tris), x, y, tris, g)
    out = w.apply(2 * x + 3 * y)
    lon2d, lat2d = np.meshgrid(g.lon, g.lat)
    assert np.allclose(out, 2 * lon2d + 3 * lat2d, atol=1e-5)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    raw = yaml.safe_load((REPO / "config" / "config.yaml").read_text())
    raw["outputs_dir"] = str(tmp / "outputs")
    raw["out_dir"] = str(tmp / "out")
    raw["plot_start"], raw["plot_end"] = "2023-01-01", "2023-01-03"
    raw["sst"]["dpi"] = 60
    raw["animation"]["fps"] = 2
    raw["animation"]["gif_max_width"] = 600
    c = Config(raw)
    make_run(raw, tmp, DAYS)
    make_satellite(c, DAYS)
    prepare.run_prepare(c)
    return c


def test_day_frames_and_stats(cfg):
    for d in DAYS:
        assert daily.run_day(cfg, d)
    for d in DAYS:
        assert daily.stats_path(cfg, d).exists()
        for t in PLOT_TYPES:
            f = cfg.frames_dir(t) / f"{t}_{d:%Y%m%d}.jpg"
            assert f.exists() and f.stat().st_size > 5000
    assert not daily.run_day(cfg, DAYS[0])          # resume-safe
    z = np.load(daily.stats_path(cfg, DAYS[0]))
    bias, rmse, corr, n = z["skill_full"]
    assert 0.3 < bias < 0.7                         # model was built +0.5 warm
    assert corr > 0.95 and n > 1000
    assert np.isfinite(z["nino34_model"]) and np.isfinite(z["coast_model"]).any()


def test_frame_sizes_constant_and_even(cfg):
    from PIL import Image
    for t in PLOT_TYPES:
        sizes = {Image.open(cfg.frames_dir(t) / f"{t}_{d:%Y%m%d}.jpg").size for d in DAYS}
        assert len(sizes) == 1
        w, h = sizes.pop()
        assert w % 2 == 0 and h % 2 == 0


def test_figures_and_animation(cfg):
    figures.run_figures(cfg)
    for n in ("nino_indices", "hovmoller_equator", "hovmoller_coast", "skill_timeseries"):
        assert (cfg.fig_dir / f"{n}.png").stat().st_size > 5000
    assert (cfg.fig_dir / "daily_stats.csv").exists()
    animate.run_animation(cfg, "zoom_cur")
    assert (cfg.anim_dir / "sst_zoom_cur.gif").stat().st_size > 1000
    assert (cfg.anim_dir / "sst_zoom_cur.mp4").stat().st_size > 1000


def test_submit_dry_run(cfg, capsys):
    from elnino.slurm import submit_all
    cfg.config_path = REPO / "config" / "config.yaml"
    submit_all(cfg, dry_run=True)
    out = capsys.readouterr().out
    assert "prepare" in out and "finalize" in out
    day = (cfg.log_dir / "day.sbatch").read_text()
    assert "--array=1-3%" in day and "{{" not in day
