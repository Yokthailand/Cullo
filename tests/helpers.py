from pathlib import Path

import cv2
import numpy as np


def make_photo(seed: int, w: int = 2400, h: int = 1600) -> np.ndarray:
    """ภาพสังเคราะห์ที่มีพื้นผิวคมชัด"""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 1.2)
    for _ in range(40):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.rectangle(img, p1, p2, tuple(int(c) for c in rng.integers(0, 256, 3)), 3)
    return img


def blur(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def save_jpg(img: np.ndarray, path: Path) -> None:
    """เขียนไฟล์ผ่าน imencode เพื่อรองรับ path ภาษาไทยบน Windows"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    buf.tofile(str(path))
