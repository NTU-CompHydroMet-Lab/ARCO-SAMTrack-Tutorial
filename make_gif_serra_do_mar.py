"""
================================================================================
  GIF builder for talks — NOT a publication-grade artifact.

  Reads per-day PNGs from each of the three view directories and stitches
  each into a GIF for easy sharing in presentations.

  Prerequisite: run event_box_region_frames.py first to render the frames.

  Outputs:
    outputs/box_abs.gif
    outputs/box_pdf.gif
    outputs/box_cdf.gif
================================================================================
"""

from pathlib import Path

from PIL import Image

VIEW_DIRS = {
    "abs": Path("outputs/box_abs"),
    "pdf": Path("outputs/box_pdf"),
    "cdf": Path("outputs/box_cdf"),
}
FRAME_DURATION_MS = 600  # ~1.7 fps — slow enough to read, fast enough to flow
DOWNSCALE = 2  # PNGs are 300 dpi × 8–9 inch ≈ 2400 px; halve for lighter GIFs


def build_gif(src_dir: Path, out_gif: Path) -> None:
    frames = sorted(src_dir.glob("*.png"))
    if not frames:
        print(f"  [skip] {src_dir} empty")
        return
    images = [Image.open(p).convert("RGB") for p in frames]
    w = min(im.width for im in images)
    h = min(im.height for im in images)
    images = [
        (im.resize((w, h), Image.LANCZOS) if im.size != (w, h) else im)
        for im in images
    ]
    if DOWNSCALE > 1:
        images = [
            im.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS)
            for im in images
        ]
    images[0].save(
        out_gif,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )
    print(f"  saved {out_gif}  ({len(images)} frames, "
          f"{out_gif.stat().st_size / 1e6:.1f} MB)")


for tag, src in VIEW_DIRS.items():
    build_gif(src, Path(f"outputs/box_{tag}.gif"))
