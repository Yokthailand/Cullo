import csv

import pytest

from config import PRESETS, REASON_BLUR
from judge import judge
from pipeline import PASS_DIR, REJECT_DIR, REPORT_NAME, run
from scorer import Face, Metrics
from helpers import blur, make_photo, save_jpg


def _init_texture_worker():
    """ภาพทดสอบเป็นพื้นผิว ไม่มีคน: ทดสอบความคมจริงโดยไม่ให้โมเดลหาหน้าปลอมบน noise.

    ต้องเป็นฟังก์ชันระดับโมดูล เพื่อให้ ProcessPool แบบ spawn บน Windows เรียกได้.
    """
    import cv2
    import scorer

    cv2.setNumThreads(1)
    scorer._models = None


def test_multi_camera_same_names_split_into_two_folders(tmp_path, monkeypatch):
    import pipeline

    # ยังใช้ ProcessPool, อ่าน JPG, scorer และ judge จริง; แยกตัวตรวจหน้าออกจาก fixture ที่ไม่มีหน้า.
    monkeypatch.setattr(pipeline, "init_worker", _init_texture_worker)
    root = tmp_path / "งานกีฬาสี"
    blurred = set()
    for cam_index, cam in enumerate(("กล้อง1", "กล้อง2"), start=1):
        for i in range(1, 9):  # ชื่อไฟล์ DSC_0001.. ซ้ำกันทั้งสองกล้อง
            # hash(str) เปลี่ยนทุก process; seed จากลำดับกล้อง/รูปจึงทำซ้ำได้เสมอ.
            img = make_photo(cam_index * 1000 + i)
            if i in (3, 7):
                img = blur(img)
                blurred.add(f"{cam}_DSC_{i:04d}.jpg")
            save_jpg(img, root / cam / f"DSC_{i:04d}.jpg")

    s = run(root, "ปกติ")

    passed = {p.name for p in (s.out_dir / PASS_DIR).iterdir()}
    rejected = {p.name for p in (s.out_dir / REJECT_DIR).iterdir()}
    assert s.total == 16 and s.passed + s.rejected == 16 == len(passed) + len(rejected)
    assert rejected == blurred
    assert not passed & rejected
    assert len(list(root.rglob("*.jpg"))) == 16  # ต้นฉบับไม่ถูกแตะ

    rows = list(csv.reader(open(s.out_dir / REPORT_NAME, encoding="utf-8-sig")))
    assert len(rows) == 17 and all(r[3] == REASON_BLUR for r in rows[1:] if r[0] == REJECT_DIR)


def test_export_follows_manual_overrides_and_records_them(tmp_path):
    from pipeline import analyze_folder, export
    from judge import judge_detailed
    root = tmp_path / "งาน"
    for i in range(6):
        save_jpg(make_photo(100 + i), root / "กล้อง1" / f"DSC_{i:04d}.jpg")
    paths, metrics = analyze_folder(root)
    verdicts = judge_detailed(metrics, PRESETS["ปกติ"])
    keep = [not v.reasons for v in verdicts]
    keep[0], keep[3] = not keep[0], not keep[3]           # ผู้ใช้กลับคำตัดสิน 2 รูป
    s = export(root, paths, metrics, verdicts, keep, tmp_path / "out", manual={0, 3})
    assert s.passed + s.rejected == 6 and s.passed == sum(keep)
    names_pass = {p.name for p in (tmp_path / "out" / PASS_DIR).iterdir()}
    assert names_pass == {f"กล้อง1_{paths[i].name}" for i in range(6) if keep[i]}
    rows = list(csv.DictReader(open(tmp_path / "out" / REPORT_NAME, encoding="utf-8-sig")))
    assert sorted(r["ชื่อไฟล์"] for r in rows if r["แก้ด้วยมือ"]) == sorted(f"กล้อง1_{paths[i].name}" for i in (0, 3))


def test_no_jpg_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(tmp_path, "ปกติ")


def _m(cam, sharp, faces=()):
    return Metrics(path="x", camera=cam, sharp=sharp, faces=list(faces))


def test_judge_relative_per_camera():
    # กล้อง A คมปกติ ~30, กล้อง B คมปกติ ~1000: ค่า 30 ในกล้อง B (0.03 เท่าของค่ากลาง) ถือว่าเบลอ ในกล้อง A ไม่เบลอ
    ms = [_m("A", 30)] * 6 + [_m("B", 1000)] * 6 + [_m("B", 30)]
    out = judge(ms, PRESETS["ปกติ"])
    assert out[0] == [] and out[-1] == [REASON_BLUR]


def test_judge_group_uses_ratio_solo_uses_any():
    p = PRESETS["ปกติ"]
    ok, closed = Face(200, 0.1, 0.0, 50), Face(200, 0.9, 0.0, 50)
    solo = _m("A", 100, [ok, closed])
    group_one_bad = _m("A", 100, [ok] * 9 + [closed])       # 10% หลับตา ไม่ถึง 30%
    group_many_bad = _m("A", 100, [ok] * 6 + [closed] * 4)  # 40% หลับตา
    out = judge([solo, group_one_bad, group_many_bad] + [_m("A", 100)] * 5, p, main_only=False)
    assert out[0] == ["หลับตา"] and out[1] == [] and out[2] == ["หลับตา"]


def test_judge_main_only_looks_at_most_prominent_face_only():
    p = PRESETS["ปกติ"]
    ok, closed = Face(200, 0.1, 0.0, 50, prominence=0.01), Face(200, 0.9, 0.0, 50, prominence=0.01)
    filler = [_m("A", 100)] * 5
    main_ok_other_closed = _m("A", 100, [ok, closed, closed])   # faces[0] = คนเด่น
    main_closed = _m("A", 100, [closed, ok])
    main_unreadable = _m("A", 100, [Face(200, 0.0, 0.0, 50, prominence=0.01, has_expr=False), closed])
    main_tiny = _m("A", 100, [Face(60, 0.9, 0.0, 50, prominence=0.01), ok])      # เล็กเกินไปจะตัดสินสีหน้า
    no_clear_subject = _m("A", 100, [Face(200, 0.9, 0.0, 50, prominence=0.0005)])  # ไม่มีตัวเอกชัดเจน
    out = judge([main_ok_other_closed, main_closed, main_unreadable, main_tiny, no_clear_subject] + filler, p)
    assert out[0] == [] and out[1] == ["หลับตา"] and out[2] == [] and out[3] == [] and out[4] == []


def test_reject_probability_rises_with_blur_and_closed_eyes():
    from judge import reject_probability
    sharp_open = reject_probability(1.0, 0.1)
    assert sharp_open < 0.3
    assert reject_probability(0.1, 0.1) > sharp_open        # หน้าเบลอกว่าค่ากลางกล้อง 10 เท่า
    assert reject_probability(1.0, 0.7) > sharp_open        # หลับตา
    assert reject_probability(1.0, 0.9) > 0.9


def test_judge_main_only_blur_uses_face_sharpness_relative_to_camera():
    p = PRESETS["ปกติ"]
    def face(sharp): return Face(200, 0.05, 0.0, sharp, prominence=0.01)
    # หน้าคมปกติ ~150 ทั้งกล้อง; รูปสุดท้ายหน้าเบลอมาก (ratio 0.04) แม้ทั้งภาพจะคมเท่ากัน (ฉากหลังคม) ก็ต้องทิ้ง
    ms = [_m("A", 500, [face(150)]) for _ in range(8)] + [_m("A", 500, [face(6)])]
    out = judge(ms, p)
    assert out[:8] == [[]] * 8 and out[8] == ["เบลอ"]
    # หน้าคม แต่ทั้งภาพเบลอ (โคลสอัพผิวเรียบ ฉากหลังละลาย) ต้องไม่ถูกทิ้ง
    ms = [_m("A", 500, [face(150)]) for _ in range(8)] + [_m("A", 100, [face(150)])]
    assert judge(ms, p)[8] == []


def test_order_faces_confirmed_faces_beat_bigger_false_detections():
    from scorer import order_faces
    fake_big = Face(500, 0, 0, 10, prominence=0.05, has_expr=False)   # ตรวจเจอบนตึก/ป้าย
    real_small = Face(150, 0.1, 0.0, 50, prominence=0.002)
    real_bigger = Face(220, 0.1, 0.0, 50, prominence=0.004)
    assert order_faces([fake_big, real_small, real_bigger]) == [real_bigger, real_small, fake_big]


def test_subject_score_penalizes_faces_cut_by_frame_edge_only():
    from scorer import order_faces
    box = (0.4, 0.3, 0.1, 0.1)
    model = Face(160, 0.1, 0.0, 50, box=box, prominence=0.0100)
    cut = Face(165, 0.1, 0.0, 50, box=(0.0, 0.3, 0.1, 0.1), prominence=0.0104)      # โดนตัดขอบซ้าย ใหญ่กว่านิดเดียว
    assert order_faces([cut, model])[0] is model
    # ท่าทาง (หันข้าง/ก้มมอง) ต้องไม่มีผลต่อลำดับ: ทดลองแล้วทำให้เลือกคนเด่นผิดเพิ่ม
    turned = Face(165, 0.1, 0.0, 50, box=box, prominence=0.0104, frontal=0.0, lookdown=0.9)
    assert order_faces([model, turned])[0] is turned


def test_prominence_prefers_big_and_central():
    from scorer import prominence
    w, h = 3000, 2000
    center_small = prominence(1400, 900, 100, 100, w, h)
    center_big = prominence(1300, 800, 400, 400, w, h)
    edge_big = prominence(0, 0, 400, 400, w, h)
    edge_double = prominence(0, 0, 300, 300, w, h)   # ใหญ่ 2 เท่า (พื้นที่) ของ center_small แต่อยู่มุมภาพ
    assert center_big > center_small and center_big > edge_big
    assert edge_double > center_small
