"""Assemble daily JPEG frames into a GIF and an MP4."""

import shutil
import subprocess
from pathlib import Path

from .config import PLOT_TYPES, Config


def _frames(cfg, ptype):
    return sorted(cfg.frames_dir(ptype).glob(f"{ptype}_*.jpg"))


def make_gif(frames, out: Path, fps: int, max_width: int):
    from PIL import Image

    def gen():
        for f in frames:
            im = Image.open(f).convert("RGB")
            if max_width and im.width > max_width:
                h = int(round(im.height * max_width / im.width))
                im = im.resize((max_width, h), Image.LANCZOS)
            yield im.convert("P", palette=Image.ADAPTIVE, colors=256)

    it = gen()
    first = next(it)
    first.save(out, save_all=True, append_images=it, duration=int(round(1000 / fps)),
               loop=0, optimize=False)


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_mp4(frames, out: Path, fps: int, ffmpeg: str):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps),
           "-vcodec", "mjpeg", "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(Path(f).read_bytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def run_animation(cfg: Config, ptype: str):
    frames = _frames(cfg, ptype)
    if not frames:
        print(f"{ptype}: no frames found, nothing to assemble.")
        return
    cfg.anim_dir.mkdir(parents=True, exist_ok=True)
    a = cfg.anim
    fps = int(a.get("fps", 6))
    print(f"{ptype}: {len(frames)} frames", flush=True)
    if a.get("make_gif", True):
        out = cfg.anim_dir / f"sst_{ptype}.gif"
        make_gif(frames, out, fps, int(a.get("gif_max_width", 1400)))
        print(f"  wrote {out}", flush=True)
    if a.get("make_mp4", True):
        ff = find_ffmpeg()
        if ff is None:
            print("  WARNING: no ffmpeg found (install imageio-ffmpeg in the env or "
                  "load an ffmpeg module); skipping MP4.", flush=True)
        else:
            out = cfg.anim_dir / f"sst_{ptype}.mp4"
            make_mp4(frames, out, fps, ff)
            print(f"  wrote {out}", flush=True)
