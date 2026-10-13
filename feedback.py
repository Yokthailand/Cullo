"""เก็บป้ายจากการตรวจสำเนาที่ส่งออก โดยไม่เปลี่ยนรูปหรือฝึกโมเดลอัตโนมัติ"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from config import JPEG_EXTS
from pipeline import PASS_DIR, REJECT_DIR, REPORT_NAME, file_sha256


@dataclass
class FeedbackSummary:
    out_dir: Path
    total: int
    labeled: int
    moved: int
    manual: int
    unchanged: int
    issues: int
    trainable: int


def _next_dir(folder: Path) -> Path:
    n = 1
    while (folder / f"ผลตรวจ_v{n}").exists():
        n += 1
    return folder / f"ผลตรวจ_v{n}"


def _trainable(row: dict[str, str]) -> bool:
    if row.get("main_only") != "1" or row.get("main_usable") != "1" or row.get("ข้อผิดพลาด"):
        return False
    try:
        ratio, blink = float(row["face_sharp_ratio"]), float(row["main_blink"])
        return math.isfinite(ratio) and ratio >= 0 and math.isfinite(blink) and 0 <= blink <= 1
    except (KeyError, ValueError, TypeError):
        return False


def collect_feedback(folder: Path, reviewed_all: bool = False) -> FeedbackSummary:
    """เทียบรายงานเดิมกับสำเนาปัจจุบัน; รูปที่ไม่แก้จะเป็น unlabeled เว้นแต่ยืนยันว่าตรวจครบ"""
    folder = Path(folder)
    with (folder / REPORT_NAME).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        required = {"ชื่อไฟล์", "ผลลัพธ์", "schema_version", "run_id", "sha256", "file_bytes",
                    "auto_keep", "exported_keep", "source_relative_path"}
        if not required.issubset(columns):
            raise ValueError("รายงานนี้เป็นเวอร์ชันเก่า กรุณาส่งออกรูปด้วยโปรแกรมเวอร์ชันใหม่ก่อนเก็บผลตรวจ")
        rows = list(reader)
    if not rows:
        raise ValueError("รายงานไม่มีรายการรูป")
    for group in (PASS_DIR, REJECT_DIR):
        if not (folder / group).is_dir():
            raise ValueError(f"ไม่พบโฟลเดอร์ {group} ในโฟลเดอร์ผลลัพธ์")

    # ใช้ชื่อแบบ case-insensitive เช่นเดียวกับ Windows และไม่เดาชื่อหลังเปลี่ยนชื่อไฟล์
    found: dict[str, list[Path]] = defaultdict(list)
    for group in (PASS_DIR, REJECT_DIR):
        for p in (folder / group).iterdir():
            if p.is_file() and p.suffix.lower() in JPEG_EXTS:
                found[p.name.casefold()].append(p)
    name_counts = Counter(r["ชื่อไฟล์"].casefold() for r in rows)
    records, labels, issues = [], [], []
    moved = manual = unchanged = 0
    for row in rows:
        name = row["ชื่อไฟล์"]
        rec = dict(row, current_result="", human_label="", feedback_source="",
                   training_eligible="0", issue="")
        candidates = found.get(name.casefold(), [])
        if not name or PureWindowsPath(name).name != name or name in {".", ".."}:
            rec["issue"] = "ชื่อไฟล์ในรายงานไม่ถูกต้อง"
        elif row["schema_version"] != "2" or row["auto_keep"] not in {"0", "1"} or row["exported_keep"] not in {"0", "1"}:
            rec["issue"] = "รูปแบบข้อมูลในรายงานไม่ถูกต้อง"
        elif row["ผลลัพธ์"] != (PASS_DIR if row["exported_keep"] == "1" else REJECT_DIR):
            rec["issue"] = "ผลส่งออกในรายงานไม่ตรงกัน"
        elif name_counts[name.casefold()] > 1:
            rec["issue"] = "ชื่อไฟล์ซ้ำในรายงาน"
        elif not candidates:
            rec["issue"] = "ไม่พบไฟล์ (หายหรือเปลี่ยนชื่อ)"
        elif len(candidates) > 1:
            rec["issue"] = "พบชื่อเดียวกันมากกว่าหนึ่งไฟล์"
        else:
            p = candidates[0]
            rec["current_result"] = p.parent.name
            try:
                if p.stat().st_size != int(row["file_bytes"]) or file_sha256(p) != row["sha256"]:
                    rec["issue"] = "เนื้อหาไฟล์เปลี่ยนจากตอนส่งออก"
            except (OSError, ValueError) as exc:
                rec["issue"] = f"ตรวจสอบไฟล์ไม่ได้: {exc}"
        if rec["issue"]:
            issues.append({"filename": name, "issue": rec["issue"]})
        else:
            if rec["current_result"] != row["ผลลัพธ์"]:
                rec["feedback_source"] = "folder_correction"
                moved += 1
            elif row.get("แก้ด้วยมือ"):
                rec["feedback_source"] = "app_override"
                manual += 1
            elif reviewed_all:
                rec["feedback_source"] = "reviewed_all"
            else:
                unchanged += 1
            if rec["feedback_source"]:
                rec["human_label"] = "keep" if rec["current_result"] == PASS_DIR else "discard"
                rec["training_eligible"] = str(int(_trainable(row)))
                labels.append(rec)
        records.append(rec)

    for key in sorted(found.keys() - name_counts.keys()):
        for p in found[key]:
            issues.append({"filename": p.name, "issue": "มีรูปเพิ่มที่ไม่มีในรายงาน"})

    out = _next_dir(folder)
    out.mkdir()  # ไม่เขียนทับผลเก่า หากมีอีก process สร้างชื่อเดียวกันจะหยุด
    extra_columns = ["current_result", "human_label", "feedback_source", "training_eligible", "issue"]
    with (out / "ผลตรวจ.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns + extra_columns)
        writer.writeheader()
        writer.writerows(records)

    summary = FeedbackSummary(out, len(rows), len(labels), moved, manual, unchanged,
                              len(issues), sum(r["training_eligible"] == "1" for r in labels))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str((folder / REPORT_NAME).resolve()),
        "reviewed_all": reviewed_all,
        "summary": dict(asdict(summary), out_dir=str(out)),
        "notes": [
            "unchanged ไม่ใช่ป้ายจากมนุษย์ เว้นแต่ผู้ใช้ยืนยันว่าตรวจครบทุกภาพ",
            "ป้ายจากรูปที่แก้อย่างเดียวมีอคติจากการเลือกตัวอย่าง ใช้วัด precision/recall ของทั้งชุดไม่ได้",
            "training_eligible ใช้ได้กับ LR คนเด่นเท่านั้น; ยังไม่มีการฝึกหรือเปลี่ยนเกณฑ์",
            "run_id + source_relative_path ระบุรูปในรอบส่งออก; sha256 ใช้ตรวจเนื้อหาและกันนับซ้ำข้ามรอบ",
        ],
        "labels": labels,
        "issues": issues,
    }
    with (out / "ป้ายจากผู้ใช้.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="โฟลเดอร์ที่มี ผ่าน/คัดออก/รายงาน.csv")
    parser.add_argument("--reviewed-all", action="store_true", help="ยืนยันว่าได้ตรวจทุกภาพแล้ว รวมภาพที่ไม่แก้")
    args = parser.parse_args()
    try:
        result = collect_feedback(args.folder, args.reviewed_all)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"เก็บผลตรวจไม่ได้: {exc}\n")
    print(f"เก็บป้าย {result.labeled} รูป · ใช้ฝึกคนเด่นได้ {result.trainable} · ต้องตรวจสอบ {result.issues} รายการ")
    print(result.out_dir)


if __name__ == "__main__":
    main()
