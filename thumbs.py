"""โหลดรูปย่อ/พรีวิวแบบไม่บล็อกหน้าจอ (ถอดรหัส JPG ในเธรด สร้าง PhotoImage ในเธรดหลักเท่านั้น)"""
from __future__ import annotations

import queue
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from PIL import Image, ImageEnhance, ImageOps, ImageTk

THUMB_ASPECT = 2 / 3     # พื้นที่รูปย่อ สูง/กว้าง (ช่องรูป 3:2 เหมือนรูปจากกล้องส่วนใหญ่)
DIM_BRIGHTNESS = 0.42    # รูปที่คัดออกแสดงมืดลง ให้แยกจากรูปที่ผ่านได้ทันที


def load_image(path: str, box: tuple[int, int]) -> Image.Image:
    """เปิด JPG ให้พอดีกรอบ box=(กว้าง, สูง) แบบย่อระหว่างถอดรหัส (draft) เร็วกว่าเปิดเต็ม 5-10 เท่า + หมุนตาม EXIF"""
    im = Image.open(path)
    im.draft("RGB", (box[0] * 2, box[1] * 2))   # ถอดรหัสที่ ~2 เท่าของกรอบแล้วค่อยย่อให้คม
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail(box, Image.LANCZOS)
    return im


def load_full(path: str) -> Image.Image:
    """เปิดภาพความละเอียดเต็ม (ใช้ตอนซูม 100%)"""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_half(path: str) -> Image.Image:
    """ภาพย่อครึ่ง (ขนาดเดียวกับที่ตัววิเคราะห์เห็น) ใช้ครอปหน้าคนเด่นมาดูความคม/ตา"""
    im = Image.open(path)
    im.draft("RGB", (im.width // 2, im.height // 2))
    return ImageOps.exif_transpose(im).convert("RGB")


class ThumbCache:
    def __init__(self, master, on_ready: Callable[[], None], capacity: int = 700):
        self.master = master
        self.on_ready = on_ready
        self.capacity = capacity
        self.size = 200
        self._cache: OrderedDict[tuple[str, int, bool], ImageTk.PhotoImage] = OrderedDict()
        self._pending: set[tuple[str, int, bool]] = set()
        self._done: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._poll()

    def set_size(self, size: int) -> None:
        self.size = size

    def get(self, path: str, dim: bool = False) -> ImageTk.PhotoImage | None:
        key = (path, self.size, dim)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if key not in self._pending:
            self._pending.add(key)
            self._pool.submit(self._work, key)
        return None

    def _work(self, key: tuple[str, int, bool]) -> None:
        try:
            im = load_image(key[0], (key[1], int(key[1] * THUMB_ASPECT)))
            if key[2]:
                im = ImageEnhance.Brightness(im).enhance(DIM_BRIGHTNESS)
            self._done.put((key, im))
        except Exception:  # noqa: BLE001 - ไฟล์เสียหนึ่งไฟล์ต้องไม่ทำให้กริดล้ม
            self._done.put((key, None))

    def _poll(self) -> None:
        got = False
        try:
            while True:
                key, im = self._done.get_nowait()
                self._pending.discard(key)
                if im is not None:
                    self._cache[key] = ImageTk.PhotoImage(im)
                    while len(self._cache) > self.capacity:
                        self._cache.popitem(last=False)
                    got = True
        except queue.Empty:
            pass
        if got:
            self.on_ready()
        self.master.after(40, self._poll)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
