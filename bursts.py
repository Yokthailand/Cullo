"""จัดกลุ่มรูปช็อตรัว (ถ่ายต่อเนื่องฉากเดียวกัน) แล้วแนะนำใบที่ดีที่สุดของแต่ละกอง

รูปสองใบอยู่กองเดียวกันเมื่อ ถ่ายห่างกัน <= MAX_GAP วินาที (EXIF) และภาพเหมือนกัน (dHash ห่าง <= MAX_HAMMING บิต)
รูปที่ไม่มีเวลาถ่าย: ใช้เฉพาะลำดับชื่อไฟล์ติดกัน + ภาพเหมือนมาก (เข้มกว่า)
"""
from __future__ import annotations

from dataclasses import dataclass

from scorer import Metrics

MAX_GAP = 3.0          # วินาที: ห่างกว่านี้ไม่นับเป็นช็อตรัว
# ยิ่งถ่ายติดกันมาก ยอมให้ภาพต่างได้มาก (คนขยับระหว่างรัวชัตเตอร์ dHash ต่างได้ 15-18 บิต จากข้อมูลจริง 0.3-0.6 วินาที)
HAMMING_BY_GAP = ((1.0, 20), (MAX_GAP, 14))      # (ห่างไม่เกิน วินาที, ต่างได้ไม่เกิน บิต จาก 64)
MAX_HAMMING_NO_TIME = 7                          # ไม่มีเวลาถ่าย: ติดกันตามชื่อไฟล์ + ภาพเหมือนมาก
LOOKAHEAD = 5          # เทียบกับรูปถัดไปในลำดับเวลากี่ใบ (กันกรณีหลายกล้องถ่ายสลับกันในโฟลเดอร์เดียว)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class Burst:
    members: list[int]      # ดัชนีรูป เรียงตามเวลา/ลำดับ
    best: int               # ดัชนีรูปที่แนะนำ


def find_bursts(metrics: list[Metrics], risk: list[float]) -> list[Burst]:
    """risk[i] = ความเสี่ยงที่รูป i จะเสีย (ต่ำ = ดี) ใช้เลือกใบที่ดีที่สุดของกอง"""
    order = sorted((i for i, m in enumerate(metrics) if not m.error and m.dhash),
                   key=lambda i: (metrics[i].camera, metrics[i].taken is None, metrics[i].taken or 0.0, i))
    parent = {i: i for i in order}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pos, i in enumerate(order):
        for j in order[pos + 1: pos + 1 + LOOKAHEAD]:
            a, b = metrics[i], metrics[j]
            if a.camera != b.camera:
                continue
            if a.taken is not None and b.taken is not None:
                gap, h = abs(b.taken - a.taken), hamming(a.dhash, b.dhash)
                close = any(gap <= g and h <= lim for g, lim in HAMMING_BY_GAP)
            else:
                close = j == i + 1 and hamming(a.dhash, b.dhash) <= MAX_HAMMING_NO_TIME
            if close:
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(find(i), []).append(i)
    bursts = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # ใบที่ดีที่สุด: ความเสี่ยงต่ำสุด; เท่ากันเลือกใบที่คมทั้งภาพกว่า
        best = min(members, key=lambda k: (risk[k], -metrics[k].sharp))
        bursts.append(Burst(members, best))
    bursts.sort(key=lambda b: b.members[0])
    return bursts
