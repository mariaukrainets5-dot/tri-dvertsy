#!/usr/bin/env python3
"""Higgsfield originals in art-src/ -> game files in art/: heroes and stickers cut out of the green background, map as webp."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

root = Path(__file__).parent
src, out = root / "art-src", root / "art"
out.mkdir(exist_ok=True)


def key(img):
    rgb = np.asarray(img.convert("RGB")).astype(float)
    corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3), rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, 0)
    d = np.sqrt(((rgb - bg) ** 2).sum(-1))
    alpha = np.clip((d - 38) / 45, 0, 1)
    g, m = rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2])
    rgb[..., 1] = np.where((g > m) & (alpha < 1), m + (g - m) * .25, g)  # despill the fringe
    a = Image.fromarray((alpha * 255).astype("uint8")).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(.8))
    res = Image.fromarray(rgb.astype("uint8")).convert("RGBA")
    res.putalpha(a)
    return res.crop(a.point(lambda v: 255 if v > 40 else 0).getbbox())


for f in sorted(src.glob("*.png")):
    name = f.stem
    if name == "map":
        im = Image.open(f).convert("RGB"); im.thumbnail((1600, 1600), Image.LANCZOS)
        im.save(out / "map.webp", quality=78, method=6)
    else:
        im = key(Image.open(f))
        im.thumbnail((520 if not name.startswith("st-") else 360,) * 2, Image.LANCZOS)
        im.save(out / f"{name}.webp", quality=84, alpha_quality=90, method=6)
    print(name, (out / (name + ".webp")).stat().st_size // 1024, "KB")
