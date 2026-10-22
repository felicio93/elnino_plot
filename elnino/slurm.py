"""Render and submit the SLURM jobs: prepare -> per-day array -> finalize array."""

import shutil
import subprocess
import sys
from pathlib import Path

from . import plotting
from .config import PLOT_TYPES, Config
from .daily import stats_path

TEMPLATES = Path(__file__).resolve().parent / "templates"
FINAL_TASKS = ["figures"] + list(PLOT_TYPES)


def render(template: str, subs: dict) -> str:
    text = (TEMPLATES / template).read_text()
    for k, v in subs.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        left = sorted({s.split("}}")[0] for s in text.split("{{")[1:]})
        raise ValueError(f"{template}: unreplaced placeholders {left}")
    return text


def sbatch(script: Path, dependency: str = None, dry_run: bool = False) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script))
    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return "DRYRUN"
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sbatch failed: {r.stderr.strip()}")
    return r.stdout.strip().split(";")[0]


def missing_days(cfg: Config):
    out = []
    for d in cfg.days():
        if not stats_path(cfg, d).exists() or any(
                not plotting.frame_path(cfg, t, d).exists() for t in PLOT_TYPES):
            out.append(d)
    return out


def submit_all(cfg: Config, skip_prepare=False, only_missing=False, dry_run=False):
    if not dry_run and shutil.which("sbatch") is None:
        sys.exit("sbatch not found; run this from a Hercules login node.")
    logdir = cfg.log_dir
    logdir.mkdir(parents=True, exist_ok=True)
    s = cfg.slurm
    common = {
        "ACCOUNT": s["account"], "PARTITION": s["partition"],
        "WORKDIR": cfg.out_dir, "LOGDIR": logdir, "PY": cfg.python,
        "REPO": cfg.repo_dir,
        "CONFIG": cfg.config_path,
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, subs):
        p = logdir / name
        p.write_text(render(name.replace(".sbatch", "") + ".sbatch", {**common, **subs}))
        return p

    jid = None
    if not skip_prepare:
        script = write("prepare.sbatch", {"MEM": s["prepare_mem"],
                                          "WALLTIME": s["prepare_walltime"]})
        jid = sbatch(script, dry_run=dry_run)
        print(f"  prepare        -> job {jid}")

    days = missing_days(cfg) if only_missing else cfg.days()
    jid_days = None
    if days:
        manifest = logdir / "days.manifest"
        manifest.write_text("\n".join(f"{d:%Y%m%d}" for d in days) + "\n")
        script = write("day.sbatch", {
            "MEM": s["day_mem"], "WALLTIME": s["day_walltime"],
            "NTASKS": len(days), "THROTTLE": s["array_throttle"],
            "MANIFEST": manifest})
        jid_days = sbatch(script, dependency=jid, dry_run=dry_run)
        print(f"  day array ({len(days)} days) -> job {jid_days}")
    else:
        print("  no days left to process")
        jid_days = jid

    script = write("finalize.sbatch", {
        "MEM": s["finalize_mem"], "WALLTIME": s["finalize_walltime"],
        "NTASKS": len(FINAL_TASKS), "TASKS": " ".join(FINAL_TASKS)})
    jid_fin = sbatch(script, dependency=jid_days, dry_run=dry_run)
    print(f"  finalize array ({', '.join(FINAL_TASKS)}) -> job {jid_fin}")
    print(f"  logs: {logdir}   outputs: {cfg.out_dir}")
