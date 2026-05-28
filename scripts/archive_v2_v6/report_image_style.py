#!/usr/bin/env python3
"""Shared report image typography helpers.

The existing report scripts draw most labels with OpenCV Hershey fonts. This
module monkey-patches ``cv2.putText`` so the same drawing calls use DejaVuSans
through PIL instead, keeping text content and coordinates unchanged while making
the exported report images easier to read.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = {
    False: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    True: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ],
}


@lru_cache(maxsize=64)
def _load_font(size, bold):
    for candidate in FONT_CANDIDATES[bool(bold)]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), int(size))
    return ImageFont.load_default()


def _font_size(font_scale):
    return max(10, int(round(float(font_scale) * 30.0)))


def _normalize_color(color, channels):
    if isinstance(color, (int, float, np.integer, np.floating)):
        vals = [int(color)] * channels
    else:
        vals = [int(v) for v in color]
        if len(vals) < channels:
            vals = vals + [vals[-1] if vals else 255] * (channels - len(vals))
    return tuple(max(0, min(255, v)) for v in vals[:channels])


def _pil_put_text(img, text, org, font_scale, color, thickness=1, bottom_left_origin=False):
    if bottom_left_origin:
        return False
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim not in (2, 3):
        return False

    channels = 1 if img.ndim == 2 else img.shape[2]
    if channels not in (1, 3, 4):
        return False

    size = _font_size(font_scale)
    font = _load_font(size, bool(thickness and thickness >= 2))
    x, y = int(org[0]), int(org[1])
    try:
        ascent, _ = font.getmetrics()
    except Exception:
        ascent = size
    top = int(round(y - ascent))

    contiguous = np.ascontiguousarray(img)
    mode = "L" if channels == 1 else "RGB" if channels == 3 else "RGBA"
    pil_img = Image.fromarray(contiguous, mode=mode)
    draw = ImageDraw.Draw(pil_img)
    fill = _normalize_color(color, channels)
    draw.text((x, top), str(text), font=font, fill=fill)
    img[...] = np.asarray(pil_img)
    return True


def patch_cv2_text(cv2_module):
    """Patch ``cv2.putText`` once, preserving the original as fallback."""
    if getattr(cv2_module, "_movie3r_report_font_patched", False):
        return

    original_put_text = cv2_module.putText

    def put_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, bottomLeftOrigin=False):
        ok = _pil_put_text(img, text, org, fontScale, color, thickness=thickness, bottom_left_origin=bottomLeftOrigin)
        if ok:
            return img
        return original_put_text(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

    cv2_module._movie3r_original_putText = original_put_text
    cv2_module.putText = put_text
    cv2_module._movie3r_report_font_patched = True
