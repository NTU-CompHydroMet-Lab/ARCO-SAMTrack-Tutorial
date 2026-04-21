"""
================================================================================
  GIF builder for talks  —  NOT a publication-grade artifact.
  Reads existing per-day PNGs from outputs/serra_do_mar_2011_daily/
  and stitches them into outputs/serra_do_mar_2011_daily.gif.

  Run event_serra_do_mar_region_frames.py first to produce the frames.
================================================================================
"""

from pathlib import Path

from PIL import Image

FRAMES_DIR = Path("outputs/serra_do_mar_2011_daily")
GIF_PATH = Path("outputs/serra_do_mar_2011_daily.gif")
FRAME_DURATION_MS = 600  # ~1.7 fps
DOWNSCALE = 2  # PNGs are 300 dpi × 16 inch = 4800 px wide; halve for GIF size

frame_paths = sorted(FRAMES_DIR.glob("*.png"))
if not frame_paths:
    raise SystemExit(f"no frames in {FRAMES_DIR}; run the frame renderer first.")

print(f"found {len(frame_paths)} frames in {FRAMES_DIR}")
images = [Image.open(p).convert("RGB") for p in frame_paths]

# Uniform size (cartopy can produce slightly different outputs per frame)
w = min(im.width for im in images)
h = min(im.height for im in images)
images = [im.resize((w, h), Image.LANCZOS) if im.size != (w, h) else im
          for im in images]
if DOWNSCALE > 1:
    images = [im.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS)
              for im in images]

images[0].save(
    GIF_PATH,
    save_all=True,
    append_images=images[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
    optimize=False,
)
print(f"saved GIF -> {GIF_PATH}  ({len(images)} frames, "
      f"{GIF_PATH.stat().st_size / 1e6:.1f} MB)")
