import pytest

from scorer import analyze, expression_scores
from helpers import blur, make_photo, save_jpg


def test_blurred_image_scores_lower(tmp_path):
    img = make_photo(1)
    save_jpg(img, tmp_path / "sharp.jpg")
    save_jpg(blur(img), tmp_path / "blurry.jpg")
    sharp = analyze(str(tmp_path / "sharp.jpg"), "")
    blurry = analyze(str(tmp_path / "blurry.jpg"), "")
    assert not sharp.error and not blurry.error
    assert blurry.sharp < sharp.sharp * 0.2


def test_thai_path_readable(tmp_path):
    save_jpg(make_photo(2), tmp_path / "กล้อง1" / "รูป.jpg")
    assert not analyze(str(tmp_path / "กล้อง1" / "รูป.jpg"), "กล้อง1").error


def test_corrupt_file_reports_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    assert analyze(str(bad), "").error


def test_expression_scores():
    assert expression_scores({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.7})[0] == 0.8
    # หัวเราะ (ยิ้ม + อ้าปาก) ไม่นับเป็นหน้าตลก แต่อ้าปากไม่ยิ้มนับ
    assert expression_scores({"jawOpen": 0.9, "mouthSmileLeft": 0.8, "mouthSmileRight": 0.8})[1] == 0.0
    assert expression_scores({"jawOpen": 0.3})[1] == pytest.approx(0.72)
    assert expression_scores({"jawOpen": 0.9})[1] == 1.0
    # mouthPucker ติดกับหน้านิ่งๆ ในรูปจริง จึงต้องไม่นับ
    assert expression_scores({"mouthPucker": 0.95})[1] == 0.0
