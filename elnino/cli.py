"""Command line interface: python -m elnino <command> --config config.yaml"""

import argparse
from datetime import date

from .config import PLOT_TYPES, load_config


def _parse_day(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="elnino", description=__doc__)
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="download + subset satellite SST (run on the DTN)")
    sub.add_parser("prepare", help="build mesh interpolation weights (once)")
    p = sub.add_parser("day", help="stats + frames for one day")
    p.add_argument("--date", required=True, help="YYYYMMDD")
    p.add_argument("--force", action="store_true")
    p.add_argument("--types", nargs="+", choices=list(PLOT_TYPES))
    p = sub.add_parser("figures", help="summary figures from the daily stats")
    sub.add_parser("map-domain", help="map of the model domain with the Niño and zoom boxes")
    sub.add_parser("map-zoom", help="map of the zoom area with the mesh, bathymetry and cities")
    sub.add_parser("maps", help="both maps")
    p = sub.add_parser("animate", help="GIF + MP4 from the frames")
    p.add_argument("--type", required=True, choices=list(PLOT_TYPES))
    p = sub.add_parser("finalize", help="run one finalize task (figures or a plot type)")
    p.add_argument("--task", required=True, choices=["figures"] + list(PLOT_TYPES))
    p = sub.add_parser("submit", help="submit prepare -> day array -> finalize to SLURM")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "download":
        from .satellite import run_download
        run_download(cfg)
    elif args.cmd == "prepare":
        from .prepare import run_prepare
        run_prepare(cfg)
    elif args.cmd == "day":
        from .daily import run_day
        run_day(cfg, _parse_day(args.date), force=args.force, types=args.types)
    elif args.cmd == "figures" or (args.cmd == "finalize" and args.task == "figures"):
        from .figures import run_figures
        run_figures(cfg)
    elif args.cmd in ("map-domain", "maps"):
        from .map_domain import run_domain_map
        run_domain_map(cfg)
        if args.cmd == "maps":
            from .map_zoom import run_zoom_map
            run_zoom_map(cfg)
    elif args.cmd == "map-zoom":
        from .map_zoom import run_zoom_map
        run_zoom_map(cfg)
    elif args.cmd in ("animate", "finalize"):
        from .animate import run_animation
        run_animation(cfg, args.type if args.cmd == "animate" else args.task)
    elif args.cmd == "submit":
        from .slurm import submit_all
        submit_all(cfg, args.skip_prepare, args.only_missing, args.dry_run)
