# elnino_plot

SCHISM (STOFS-3D Pacific, run `RUN300b`) vs satellite SST for the 2023 El Niño.
One frame per day, model and satellite side by side with a shared colour bar.

## Outputs (`out_dir`, default `/work2/noaa/nos-surge/felicioc/STOFS_Pacific/elnino2023`)

| Path | Content |
|---|---|
| `frames/{full,zoom,full_cur,zoom_cur}/*.jpg` | daily frames |
| `animations/sst_<type>.gif`, `.mp4` | animations of each frame set |
| `figures/nino_indices.png` | Niño 3.4 / 3 / 1+2 area-mean SST, model vs satellite (+ change since January) |
| `figures/hovmoller_equator.png` | 2°S–2°N SST, longitude vs date, model and satellite |
| `figures/hovmoller_coast.png` | US West Coast coastal-band SST, latitude vs date |
| `figures/skill_timeseries.png`, `daily_stats.csv` | daily bias / RMSE / correlation (full domain and zoom) |
| `obs/sst_leo/` | satellite SST subset to the plotting grids |
| `stats/` | per-day statistics the summary figures are built from |

Plot types: `full` (whole domain), `zoom` (Santa Maria → south of Tijuana),
`full_cur` and `zoom_cur` (same, with surface-current arrows on the model panel).

## How it works

* **Satellite**: NOAA/STAR ACSPO L3S-LEO-DY daily 0.02° SST (same product as `stofs_ak_workflow`),
  downloaded on the DTN, converted to °C and block-averaged onto the plotting grids
  (0.1° full domain, 0.02° zoom). ~430 MB per day is downloaded and deleted after subsetting.
* **Model**: surface layer (last vertical index) of `temperature_N.nc`, mean of the 24 hourly outputs of the day
  (`sst.match: noon` uses the output nearest 12Z instead). Day D is in stack `D - model_start + 1`.
  Currents come from the surface layer of `horizontalVel{X,Y}_N.nc`, daily mean, block-averaged to the arrow grid.
* **Regridding**: the unstructured mesh (quads split into triangles) is interpolated to the regular grids with
  barycentric weights computed once by the `prepare` job and cached.
  Model and satellite sit on identical grids, so both panels and all statistics are cell-for-cell comparable.
* Arrow length is proportional to speed with a fixed scale for every frame (`currents.scale_*`).
* Skill/indices use only cells where both model and satellite are valid and are area-weighted (cos latitude).
  The "indices" are absolute area-mean SST: no climatology exists for the model, so no anomalies are computed.

## Run on Hercules

Everything is configured in `config/config.yaml` (paths, domain, zoom box, colour limits, Slurm settings).

```bash
git clone <repo> && cd elnino_plot          # or git pull

# 1. satellite data (DTN: needs internet). Resume-safe; rerun to retry failed days.
ssh hercules-dtn.hpc.msstate.edu
cd <repo>
/work2/noaa/nos-surge/felicioc/envs/miniconda3/envs/swf_plot/bin/python -m elnino --config config/config.yaml download
# (use nohup/tmux: ~365 files)

# 2. from a login node: submit prepare -> per-day array -> finalize (figures + animations)
PY=/work2/noaa/nos-surge/felicioc/envs/miniconda3/envs/swf_plot/bin/python
$PY -m elnino --config config/config.yaml submit --dry-run     # inspect first
$PY -m elnino --config config/config.yaml submit
$PY -m elnino --config config/config.yaml submit --skip-prepare --only-missing   # retry afterwards
```

Test one day before launching the whole year (after `prepare`):

```bash
$PY -m elnino --config config/config.yaml day --date 20230115
```

### MP4 needs ffmpeg
Either `module load ffmpeg` (if available), or `pip install imageio-ffmpeg` into `swf_plot`
(on the DTN, which has internet). Without it the MP4 is skipped with a warning; GIFs are unaffected.

## Tests

`pytest tests` builds a small synthetic SCHISM run (quads, land, 4 layers) and fake satellite files and
exercises every stage. Needs numpy, scipy, xarray, netCDF4, matplotlib, pillow, pyyaml, pytest.
