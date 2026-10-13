"""ตัดสินว่ารูปไหนผ่าน/คัดออก จากผลวิเคราะห์ของทุกรูป"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from config import (AWAY_RATIO, BIAS, GROUP_MIN_FACES, MIN_EXPR_FACE_PX, MIN_MAIN_PROMINENCE, PERSON_MIN_SCORE,
                    PERSON_TOP_EXTEND, REASON_BLUR, REASON_EYES, REASON_FUNNY, W_BLINK, W_FACE_SHARP, Preset)
from scorer import Face, Metrics, prominence


def _person_prominence(p: tuple) -> float:
    return prominence(p[0], p[1], p[2], p[3], 1, 1)


def _has_face(p: tuple, f: Face) -> bool:
    fx, fy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
    return p[0] <= fx <= p[0] + p[2] and p[1] - PERSON_TOP_EXTEND * p[3] <= fy <= p[1] + p[3]


def subject_faces_away(m: Metrics, main: Face) -> bool:
    """ตัวเอกหันหลัง: คนตัวใหญ่ที่สุดในภาพไม่มีหน้าให้เห็นเลย ขณะที่หน้าคนเด่นที่เลือกเป็นของคนอีกคนที่เล็กกว่ามาก
    (เช่น ผู้ชมข้างหลังในที่คนเยอะ) — ต้องเห็นชัดว่าหน้านั้นอยู่ในกรอบคนจริงและคนตัวใหญ่กว่าอย่างน้อย AWAY_RATIO เท่า
    ไม่แน่ใจ (เช่น ชุดปีกใหญ่ที่ตัวตรวจจับไม่รู้จักว่าเป็นคน) = ไม่ถือว่าหันหลัง เพื่อไม่ข้ามรูปที่ควรตัดสิน"""
    people = [p for p in m.persons if p[4] >= PERSON_MIN_SCORE]
    if not people:
        return False
    top = max(people, key=_person_prominence)
    if any(_has_face(top, f) for f in m.faces if f.has_expr):
        return False
    owners = [p for p in people if _has_face(p, main)]
    if not owners:
        return False
    owner = min(owners, key=lambda p: p[2] * p[3])
    return _person_prominence(top) >= AWAY_RATIO * _person_prominence(owner)


def usable_main(m: Metrics) -> Face | None:
    """คนเด่นที่เชื่อถือได้พอจะประเมินความคม/หลับตา (ยืนยันแล้วว่าเป็นหน้าจริง ไม่เล็กเกินไป ไม่ใช่แค่คนไกลๆ ในฉากหลัง
    และไม่ใช่กรณีตัวเอกหันหลังแล้วเหลือแต่คนข้างหลัง)"""
    main = m.faces[0] if m.faces else None
    if (main and main.has_expr and main.size >= MIN_EXPR_FACE_PX and main.prominence >= MIN_MAIN_PROMINENCE
            and not subject_faces_away(m, main)):
        return main
    return None


def _camera_medians(all_metrics: list[Metrics]) -> dict[str, tuple[float, float | None]]:
    """ค่ามัธยฐานความคม (ทั้งภาพ, หน้าเด่น) ของกล้องแต่ละตัว; หน้าเด่นถ้ากล้องนั้นมีน้อยกว่า 5 รูปใช้ค่าของทุกกล้องรวมกัน"""
    sharps: dict[str, list[float]] = defaultdict(list)
    face_sharps: dict[str, list[float]] = defaultdict(list)
    for m in all_metrics:
        if m.error:
            continue
        sharps[m.camera].append(m.sharp)
        if (main := usable_main(m)) is not None:
            face_sharps[m.camera].append(main.sharp)
    overall = [s for v in face_sharps.values() for s in v]
    return {
        cam: (median(v), median(face_sharps[cam]) if len(face_sharps[cam]) >= 5
              else (median(overall) if len(overall) >= 5 else None))
        for cam, v in sharps.items()
    }


def reject_probability(face_sharp_ratio: float, blink: float) -> float:
    """ความน่าจะเป็นที่รูปนี้ควรถูกทิ้ง เพราะหน้าคนเด่นเบลอ/หลับตา (ดู config.py)"""
    z = W_FACE_SHARP * math.log(face_sharp_ratio + 1e-3) + W_BLINK * blink + BIAS
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Verdict:
    reasons: list[str]
    prob: float | None = None    # ความน่าจะเป็นรูปหลุดของคนเด่น (None = ไม่มีคนเด่นให้ประเมิน)
    ratio: float | None = None   # ความคมของหน้าคนเด่น / ค่ามัธยฐานกล้อง (1.0 = คมปกติ)
    away: bool = False           # ตัวเอกน่าจะหันหลัง (เหลือแต่หน้าคนข้างหลัง) จึงไม่ตัดสินจากหน้า


ALL_KINDS = frozenset({REASON_BLUR, REASON_EYES, REASON_FUNNY})   # สิ่งที่ตรวจได้ (ผู้ใช้ปิดบางอย่างได้)
# "หน้าตลก" ปิดเป็นค่าเริ่มต้น: ตรวจได้แค่อ้าปากกว้างไม่ยิ้ม ซึ่งคือคนกำลังพูด/ร้องเพลงถือไมค์ (ดูงานปฐมนิเทศ 400 รูป
# คัดออกผิดเกือบทั้งหมด) ผู้ใช้เปิดเองได้ในแอป
DEFAULT_KINDS = frozenset({REASON_BLUR, REASON_EYES})


def _main_reasons(m: Metrics, main: Face, med_face: float | None, p: Preset, on: frozenset) -> list[str]:
    ratio = main.sharp / med_face if med_face else 1.0   # ไม่มีค่าเทียบ = ถือว่าคมปกติ
    # ปิดตรวจอะไร ก็ตัดสัญญาณนั้นออกจากสูตร (ถือว่าคมปกติ/ลืมตา) ไม่ใช่แค่ซ่อนป้าย
    r_eff = ratio if REASON_BLUR in on else 1.0
    b_eff = main.blink if REASON_EYES in on else 0.0
    reasons: list[str] = []
    if reject_probability(r_eff, b_eff) >= p.reject_p:
        # บอกเหตุผลจากสัญญาณที่ชัดที่สุด (เบลอและ/หรือหลับตา)
        if r_eff <= 0.35:
            reasons.append(REASON_BLUR)
        if b_eff >= 0.40:
            reasons.append(REASON_EYES)
        if not reasons:
            reasons.append(REASON_BLUR if -W_FACE_SHARP * math.log(r_eff + 1e-3) > W_BLINK * b_eff
                           else REASON_EYES)
    if REASON_FUNNY in on and main.funny >= p.funny:
        reasons.append(REASON_FUNNY)
    return reasons


def _all_faces_reasons(m: Metrics, med: tuple[float, float | None], p: Preset, on: frozenset) -> list[str]:
    """โหมดตรวจทุกคน: ความคมดูที่หน้าเด่น, หลับตา/หน้าตลกดูทุกหน้า (รูปหมู่ดูสัดส่วน)"""
    reasons: list[str] = []
    main, med_face = usable_main(m), med[1]
    if REASON_BLUR in on:
        if main is not None and med_face:
            if main.sharp <= p.blur_face_ratio * med_face:
                reasons.append(REASON_BLUR)
        elif m.sharp < p.blur_ratio * med[0]:
            reasons.append(REASON_BLUR)

    faces = [f for f in m.faces if f.has_expr and f.size >= MIN_EXPR_FACE_PX]
    if faces:
        eyes = [f.blink >= p.eye_closed for f in faces]
        funny = [f.funny >= p.funny for f in faces]
        if len(faces) >= GROUP_MIN_FACES:
            n = len(faces)
            if REASON_EYES in on and sum(eyes) / n >= p.group_bad_ratio:
                reasons.append(REASON_EYES)
            if REASON_FUNNY in on and sum(funny) / n >= p.group_bad_ratio:
                reasons.append(REASON_FUNNY)
        else:
            if REASON_EYES in on and any(eyes):
                reasons.append(REASON_EYES)
            if REASON_FUNNY in on and any(funny):
                reasons.append(REASON_FUNNY)
    return reasons


def _reasons(m: Metrics, med: tuple[float, float | None], p: Preset, main_only: bool, on: frozenset) -> list[str]:
    if not main_only:
        return _all_faces_reasons(m, med, p, on)
    main = usable_main(m)
    if main is not None:
        return _main_reasons(m, main, med[1], p, on)
    # ไม่มีคนเด่นที่ประเมินได้ (ภาพเวทีไกลๆ ตัวเอกหันหลัง ฯลฯ): เหลือแต่ความคมทั้งภาพ ตัดเฉพาะที่เบลอมาก
    return [REASON_BLUR] if REASON_BLUR in on and m.sharp < p.blur_ratio * med[0] else []


def judge_detailed(all_metrics: list[Metrics], preset: Preset, main_only: bool = True,
                   enabled: frozenset | set | None = None) -> list[Verdict]:
    """เหมือน judge แต่คืนรายละเอียด (ความน่าจะเป็น/อัตราความคม) ไว้แสดงในหน้าจอ
    enabled = ชุดเหตุผลที่จะตรวจ (ค่าเริ่มต้นทั้งหมด: เบลอ/หลับตา/หน้าตลก)"""
    on = DEFAULT_KINDS if enabled is None else frozenset(enabled)
    medians = _camera_medians(all_metrics)
    out: list[Verdict] = []
    for m in all_metrics:
        if m.error:
            out.append(Verdict(["อ่านไฟล์ไม่ได้"]))
            continue
        med = medians[m.camera]
        v = Verdict(_reasons(m, med, preset, main_only, on))
        head = m.faces[0] if m.faces else None
        v.away = bool(head and head.has_expr and head.size >= MIN_EXPR_FACE_PX and head.prominence >= MIN_MAIN_PROMINENCE
                      and subject_faces_away(m, head))
        if (main := usable_main(m)) is not None:
            v.ratio = main.sharp / med[1] if med[1] else 1.0
            v.prob = reject_probability(v.ratio if REASON_BLUR in on else 1.0, main.blink if REASON_EYES in on else 0.0)
        out.append(v)
    return out


def judge(all_metrics: list[Metrics], preset: Preset, main_only: bool = True,
          enabled: frozenset | set | None = None) -> list[list[str]]:
    """คืนรายการเหตุผลที่คัดออกของแต่ละรูป (ลำดับเดียวกับอินพุต); ว่าง = ผ่าน
    main_only=True: ประเมินเฉพาะคนเด่นที่สุด (ความคมของหน้า + หลับตา รวมเป็นความน่าจะเป็นเดียว);
    False: ตรวจทุกคน (รูปหมู่ดูสัดส่วน) รูปที่อ่านไม่ได้จะถูกส่งไปคัดออกด้วยเหตุผลอ่านไม่ได้"""
    return [v.reasons for v in judge_detailed(all_metrics, preset, main_only, enabled)]
