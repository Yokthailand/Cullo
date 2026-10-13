import numpy as np

from bursts import find_bursts, hamming
from config import PRESETS
from judge import judge
from scorer import Face, Metrics, dhash64
from helpers import blur, make_photo


def _m(name, taken, dh, sharp=100.0, cam="", faces=()):
    return Metrics(path=name, camera=cam, sharp=sharp, dhash=dh, taken=taken, faces=list(faces))


def test_dhash_similar_images_close_different_far():
    import cv2
    a = cv2.cvtColor(make_photo(1, 800, 600), cv2.COLOR_BGR2GRAY)
    a2 = cv2.cvtColor(blur(make_photo(1, 800, 600), 1.5), cv2.COLOR_BGR2GRAY)   # ฉากเดียวกัน ต่างนิดหน่อย
    b = cv2.cvtColor(make_photo(2, 800, 600), cv2.COLOR_BGR2GRAY)
    assert hamming(dhash64(a), dhash64(a2)) < hamming(dhash64(a), dhash64(b))


def test_bursts_group_by_time_and_similarity_and_pick_lowest_risk():
    ms = [
        _m("a1", 100.0, 0b1010), _m("a2", 100.6, 0b1011), _m("a3", 101.2, 0b1010),   # ช็อตรัว 3 ใบ
        _m("b1", 200.0, 0b1010),                                                     # ภาพเหมือนแต่คนละเวลา
        _m("c1", 300.0, 0xFFFFFFFF), _m("c2", 300.5, 0x0),                           # ใกล้เวลาแต่ภาพต่างกัน
    ]
    risk = [0.7, 0.1, 0.4, 0.2, 0.2, 0.2]
    bs = find_bursts(ms, risk)
    assert len(bs) == 1
    assert bs[0].members == [0, 1, 2] and bs[0].best == 1


def test_burst_similarity_limit_relaxes_for_very_close_shots():
    base = 1 << 40                                              # (dhash = 0 ถือว่าไม่มีข้อมูล)
    far = base ^ ((1 << 17) - 1)                                # ห่าง 17 บิต
    fast = [_m("f1", 50.0, base), _m("f2", 50.4, far)]         # ถ่ายห่าง 0.4 วินาที คนขยับ 17 บิต -> ยังนับเป็นกองเดียวกัน
    slow = [_m("s1", 50.0, base), _m("s2", 52.5, far)]         # ห่าง 2.5 วินาที ต่าง 17 บิต -> คนละกอง
    assert len(find_bursts(fast, [0.5, 0.5])) == 1
    assert find_bursts(slow, [0.5, 0.5]) == []


def test_bursts_do_not_mix_cameras_and_handle_missing_time():
    ms = [_m("x1", 10.0, 5, cam="A"), _m("y1", 10.2, 5, cam="B"),
          _m("n1", None, 9), _m("n2", None, 9)]                     # ไม่มีเวลาถ่าย แต่อยู่ติดกันและภาพเหมือนกันมาก
    bs = find_bursts(ms, [0.5] * 4)
    assert [b.members for b in bs] == [[2, 3]]


def test_subject_facing_away_is_not_judged_from_background_faces():
    from judge import usable_main
    bg = Face(120, 0.9, 0.0, 50.0, box=(0.70, 0.30, 0.04, 0.06), prominence=0.004)       # ผู้ชมข้างหลัง หลับตา
    crowd_person = (0.66, 0.30, 0.10, 0.30, 0.8)                                          # กรอบคนของผู้ชม (มีหน้า)
    back_turned = (0.30, 0.15, 0.40, 0.80, 0.9)                                           # ตัวเอกตัวใหญ่ ไม่มีหน้าให้เห็น
    away = Metrics(path="a", camera="", faces=[bg], persons=[back_turned, crowd_person])
    assert usable_main(away) is None                                                      # ไม่ตัดสินจากหน้าคนข้างหลัง
    normal = Metrics(path="b", camera="", faces=[bg], persons=[crowd_person])             # คนตัวใหญ่ที่สุดคือคนที่มีหน้า
    assert usable_main(normal) is bg
    unknown = Metrics(path="c", camera="", faces=[bg], persons=[back_turned])              # หน้าที่เลือกไม่อยู่ในกรอบคนใดเลย = ไม่แน่ใจ ไม่ข้าม
    assert usable_main(unknown) is bg
    no_model = Metrics(path="d", camera="", faces=[bg])                                   # ไม่มีตัวตรวจจับคน = ทำงานแบบเดิม
    assert usable_main(no_model) is bg


def test_disabling_a_check_removes_that_signal_from_the_decision():
    p = PRESETS["ปกติ"]
    closed = Face(200, 0.9, 0.0, 50.0, prominence=0.01)
    filler = [_m("f", None, 1, faces=[Face(200, 0.05, 0.0, 50.0, prominence=0.01)]) for _ in range(6)]
    ms = [_m("closed", None, 2, faces=[closed])] + filler
    assert judge(ms, p)[0] == ["หลับตา"]
    assert judge(ms, p, enabled={"เบลอ", "หน้าตลก"})[0] == []          # ปิดตรวจหลับตา
    assert judge(ms, p, main_only=False, enabled={"เบลอ"})[0] == []
