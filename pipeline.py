"""สแกนโฟลเดอร์ -> วิเคราะห์ขนาน -> ตัดสิน -> คัดลอกเป็น ผ่าน/คัดออก + รายงาน CSV"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import BIAS, JPEG_EXTS, PRESETS, W_BLINK, W_FACE_SHARP, Preset
from judge import Verdict, judge_detailed, usable_main
from scorer import Metrics, analyze, init_worker

PASS_DIR = "ผ่าน"
REJECT_DIR = "คัดออก"
REPORT_NAME = "รายงาน.csv"
FEEDBACK_COLUMNS = [
    "schema_version", "run_id", "source_relative_path", "sha256", "file_bytes",
    "auto_keep", "exported_keep", "preset", "main_only", "reject_threshold",
    "model_w_face_sharp", "model_w_blink", "model_bias", "main_usable",
    "main_has_expr", "main_size", "main_prominence", "main_sharp", "main_blink",
    "main_funny", "face_sharp_ratio", "reject_probability",
]


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

ProgressCb = Callable[[str, int, int], None]  # (ขั้นตอน, ทำแล้ว, ทั้งหมด)


class Cancelled(Exception):
    pass


@dataclass
class Summary:
    out_dir: Path
    total: int
    passed: int
    rejected: int
    unreadable: int


def scan(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in JPEG_EXTS)


def camera_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else ""


def flat_name(path: Path, root: Path) -> str:
    """ชื่อไฟล์ปลายทาง: ใส่ชื่อโฟลเดอร์กล้องนำหน้า กันชื่อซ้ำข้ามกล้อง (กล้อง1_DSC_0001.jpg)"""
    return "_".join(path.relative_to(root).parts)


def make_out_dir(root: Path, parent: Path | None = None) -> Path:
    parent = parent or root.parent
    base = parent / f"{root.name}_คัดแล้ว"
    out, n = base, 1
    while out.exists():
        n += 1
        out = parent / f"{base.name} ({n})"
    return out


def _unique(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    stem, suffix, n = dest.stem, dest.suffix, 1
    while dest.exists():
        n += 1
        dest = dest_dir / f"{stem}~{n}{suffix}"
    return dest


def analyze_all(paths: list[Path], root: Path, progress: ProgressCb | None,
                cancel: threading.Event | None) -> list[Metrics]:
    workers = max(1, min((os.cpu_count() or 2) - 1, 8))
    results: dict[int, Metrics] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as ex:
        futures = {
            ex.submit(analyze, str(p), camera_of(p, root)): i for i, p in enumerate(paths)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            if cancel is not None and cancel.is_set():
                ex.shutdown(wait=False, cancel_futures=True)
                raise Cancelled
            results[futures[fut]] = fut.result()
            if progress:
                progress("วิเคราะห์รูป", done, len(paths))
    return [results[i] for i in range(len(paths))]


def analyze_folder(root: Path, progress: ProgressCb | None = None,
                   cancel: threading.Event | None = None) -> tuple[list[Path], list[Metrics]]:
    """สแกน + วิเคราะห์ทุกรูป (ขั้นตอนช้า) แยกจากการตัดสินเพื่อให้ปรับเกณฑ์แล้วเห็นผลทันทีโดยไม่ต้องวิเคราะห์ใหม่"""
    paths = scan(root)
    if not paths:
        raise FileNotFoundError("ไม่พบไฟล์ .jpg ในโฟลเดอร์ที่เลือก")
    return paths, analyze_all(paths, root, progress, cancel)


def export(root: Path, paths: list[Path], metrics: list[Metrics], verdicts: list[Verdict],
           keep: list[bool], out_dir: Path | None = None, manual: set[int] | None = None,
           progress: ProgressCb | None = None, cancel: threading.Event | None = None,
           *, preset: Preset | None = None, main_only: bool = True, viewed: set[int] | None = None) -> Summary:
    """คัดลอกรูปเป็น ผ่าน/คัดออก ตาม keep[i] (True = ผ่าน) + เขียนรายงาน.csv
    manual = ดัชนีรูปที่ผู้ใช้กลับคำตัดสินของโปรแกรมด้วยมือ (บันทึกไว้ในรายงาน ใช้เป็นป้ายจากมนุษย์ตอนปรับเกณฑ์)"""
    manual = manual or set()
    if not (len(paths) == len(metrics) == len(verdicts) == len(keep)):
        raise ValueError("จำนวนรูป ผลวิเคราะห์ และคำตัดสินไม่ตรงกัน")
    out_dir = out_dir or make_out_dir(root)
    # ไม่ให้สำเนาผลลัพธ์ปะปนในต้นฉบับจนการสแกนครั้งถัดไปนับรูปซ้ำ
    if out_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("เลือกที่เก็บผลลัพธ์นอกโฟลเดอร์ต้นฉบับ")
    (out_dir / PASS_DIR).mkdir(parents=True)
    (out_dir / REJECT_DIR).mkdir()

    run_id = uuid.uuid4().hex
    passed = rejected = unreadable = 0
    rows = []
    for i, (p, m, v, ok) in enumerate(zip(paths, metrics, verdicts, keep)):
        if cancel is not None and cancel.is_set():
            raise Cancelled
        dest = _unique(out_dir / (PASS_DIR if ok else REJECT_DIR), flat_name(p, root))
        shutil.copy2(p, dest)
        passed += ok
        rejected += not ok
        unreadable += bool(m.error)
        main = m.faces[0] if m.faces else None
        expr = [f for f in m.faces if f.has_expr]
        rows.append([
            dest.parent.name, dest.name, m.camera, " + ".join(v.reasons), f"{m.sharp:.1f}",
            len(m.faces), f"{main.sharp:.1f}" if main else "",
            f"{main.blink:.2f}" if main and main.has_expr else "",
            f"{main.funny:.2f}" if main and main.has_expr else "",
            f"{max(f.blink for f in expr):.2f}" if expr else "",
            f"{max(f.funny for f in expr):.2f}" if expr else "", m.error,
            f"{v.prob:.2f}" if v.prob is not None else "", f"{v.ratio:.2f}" if v.ratio is not None else "",
            "แก้ด้วยมือ" if i in manual else "",
            "2", run_id, p.relative_to(root).as_posix(), file_sha256(dest), dest.stat().st_size,
            int(not v.reasons), int(ok), preset.label if preset else "",
            int(main_only), preset.reject_p if preset else "",
            W_FACE_SHARP, W_BLINK, BIAS, int(usable_main(m) is not None and not m.error),
            int(main.has_expr) if main else "", main.size if main else "",
            main.prominence if main else "", main.sharp if main else "",
            main.blink if main and main.has_expr else "",
            main.funny if main and main.has_expr else "",
            v.ratio if v.ratio is not None else "", v.prob if v.prob is not None else "",
            "ตรวจแล้ว" if viewed and i in viewed else "",     # ผู้ใช้เปิดดู/ตัดสินรูปนี้แล้ว = คำตัดสินสุดท้ายเป็นป้ายจากคนจริง
        ])
        if progress:
            progress("คัดลอกไฟล์", i + 1, len(paths))

    with open(out_dir / REPORT_NAME, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ผลลัพธ์", "ชื่อไฟล์", "กล้อง", "เหตุผลที่คัดออก", "ความคมทั้งภาพ",
                    "จำนวนหน้า", "ความคมหน้าเด่น", "หลับตา(คนเด่น)", "หน้าตลก(คนเด่น)",
                    "หลับตาสูงสุด(ทุกคน)", "หน้าตลกสูงสุด(ทุกคน)", "ข้อผิดพลาด",
                    "ความน่าจะเป็นรูปหลุด", "ความคมหน้า/ค่ากลางกล้อง", "แก้ด้วยมือ"] + FEEDBACK_COLUMNS + ["ตรวจแล้ว"])
        w.writerows(rows)

    return Summary(out_dir, len(paths), passed, rejected, unreadable)


def run(root: Path, preset: Preset | str = "ปกติ", progress: ProgressCb | None = None,
        cancel: threading.Event | None = None, main_only: bool = True,
        out_dir: Path | None = None) -> Summary:
    """วิเคราะห์ + ตัดสิน + ส่งออกในคำสั่งเดียว (โหมดบรรทัดคำสั่ง/ทดสอบ)"""
    if isinstance(preset, str):
        preset = PRESETS[preset]
    paths, metrics = analyze_folder(root, progress, cancel)
    verdicts = judge_detailed(metrics, preset, main_only)
    return export(root, paths, metrics, verdicts, [not v.reasons for v in verdicts], out_dir,
                  progress=progress, cancel=cancel, preset=preset, main_only=main_only)
