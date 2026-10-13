"""ไอคอน/แคปซูลที่วาดด้วย PIL แบบ supersample แล้วย่อ (ขอบเรียบ ไม่หยักเหมือนวาดบน tk.Canvas)"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageTk

import theme

_SS = 4                       # supersample
_cache: dict = {}


def _hex(c: str, alpha: int = 255) -> tuple:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha


def _finish(img: Image.Image, size: tuple[int, int]) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))


def badge(kind: str, size: int) -> ImageTk.PhotoImage:
    """วงกลมสถานะ: kind = 'x' (คัดออก, แดง) หรือ 'check' (ผ่าน, เขียว)"""
    key = ("badge", kind, size)
    if key not in _cache:
        s = size * _SS
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        col = theme.BAD if kind == "x" else theme.GOOD
        d.ellipse((0, 0, s - 1, s - 1), fill=_hex("#000000", 150))               # ขอบดำโปร่ง ให้เห็นบนรูปสว่าง
        m = int(s * 0.07)
        d.ellipse((m, m, s - 1 - m, s - 1 - m), fill=_hex(col))
        w = max(2, int(s * 0.11))
        if kind == "x":
            a, b = int(s * 0.32), int(s * 0.68)
            d.line((a, a, b, b), fill=(255, 255, 255, 255), width=w)
            d.line((a, b, b, a), fill=(255, 255, 255, 255), width=w)
        else:
            d.line((int(s * 0.28), int(s * 0.52), int(s * 0.44), int(s * 0.68)), fill=(255, 255, 255, 255), width=w)
            d.line((int(s * 0.44), int(s * 0.68), int(s * 0.72), int(s * 0.34)), fill=(255, 255, 255, 255), width=w)
        _cache[key] = _finish(img, (size, size))
    return _cache[key]


def pill(w: int, h: int, fill: str = "#000000", alpha: int = 165) -> ImageTk.PhotoImage:
    """แคปซูลโปร่งแสง (พื้นของชิปบนรูป)"""
    key = ("pill", w, h, fill, alpha)
    if key not in _cache:
        img = Image.new("RGBA", (w * _SS, h * _SS), (0, 0, 0, 0))
        ImageDraw.Draw(img).rounded_rectangle((0, 0, w * _SS - 1, h * _SS - 1), radius=h * _SS // 2, fill=_hex(fill, alpha))
        _cache[key] = _finish(img, (w, h))
    return _cache[key]


def logo(size: int) -> ImageTk.PhotoImage:
    key = ("logo", size)
    if key not in _cache:
        s = size * _SS
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.26), fill=_hex(theme.ACCENT))
        c, r = s / 2, s * 0.27
        d.ellipse((c - r, c - r, c + r, c + r), outline=(255, 255, 255, 255), width=int(s * 0.07))
        r2 = s * 0.11
        d.ellipse((c - r2, c - r2, c + r2, c + r2), fill=(255, 255, 255, 255))
        _cache[key] = _finish(img, (size, size))
    return _cache[key]
