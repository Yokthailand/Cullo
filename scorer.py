"""วิเคราะห์รูปเดี่ยว: ความคม และหน้าคน (หลับตา / หน้าตลก)"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import MIN_FACE_PX

DETECT_LONG_SIDE = 1600   # ย่อภาพให้ด้านยาวเท่านี้ก่อนหาหน้า
CROP_MARGIN = 0.4         # ขยายกรอบหน้าก่อนส่งเข้า landmarker
CROP_MAX_SIDE = 512
CROP_MIN_SIDE = 256       # หน้าเล็กขยายขึ้นก่อนส่ง landmarker
TILE_SIZE = 768           # ช่องตรวจหน้าขนาดเล็กในภาพเต็ม
TILE_OVERLAP = 256
FACE_SHARP_SIZE = 160     # ย่อหน้าเป็นขนาดนี้ก่อนวัดความคม
GRID = 6

DETECTOR_MODEL = "models/blaze_face_short_range.tflite"
LANDMARKER_MODEL = "models/face_landmarker.task"


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


@dataclass
class Face:
    size: int       # ความกว้างหน้า (พิกเซล) ในภาพเต็ม
    blink: float    # 0-1 ยิ่งสูงยิ่งหลับตา
    funny: float    # 0-1 ยิ่งสูงยิ่งหน้าแปลก
    sharp: float    # ความคมของหน้า
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h เป็นสัดส่วนของภาพ (0-1)
    prominence: float = 0.0   # ความเด่นในภาพ: พื้นที่หน้า x ความใกล้กลางภาพ
    has_expr: bool = True     # False = เจอหน้าแต่โมเดลใบหน้ายืนยันไม่ได้ (หันข้าง/บัง/หน้าปลอมบนของในฉาก)
    ear: float = 0.3          # Eye Aspect Ratio จากจุดใบหน้า (ยิ่งต่ำยิ่งหลับตา)
    lookdown: float = 0.0     # 0-1 มองต่ำ (ก้มหน้า) ทำให้ blink สูงเกินจริงได้
    blur: float = 0.0         # ตัววัดเบลอ Crete 0-1 ของหน้า (ยิ่งสูงยิ่งเบลอ)
    frontal: float = 1.0      # 0-1 หันเข้ากล้องแค่ไหน (1 = หน้าตรง, 0 = หันข้างเต็มที่)


@dataclass
class Metrics:
    path: str
    camera: str
    sharp: float = 0.0
    blur: float = 0.0         # ตัววัดเบลอ Crete 0-1 ของทั้งภาพ
    faces: list[Face] = field(default_factory=list)
    error: str = ""
    persons: list = field(default_factory=list)   # คนในภาพ [(x, y, w, h, คะแนน)] สัดส่วน 0-1
    dhash: int = 0            # ลายนิ้วมือภาพ 64 บิต (ไว้จับรูปช็อตรัวที่หน้าตาเหมือนกัน)
    taken: float | None = None   # เวลาถ่าย (วินาทีตั้งแต่ epoch จาก EXIF) None = ไม่มีข้อมูล


def models_available() -> bool:
    return all(os.path.exists(resource_path(p)) for p in (DETECTOR_MODEL, LANDMARKER_MODEL))


_models = None


def _load_models():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    def buf(rel):  # อ่านเป็น bytes เพื่อเลี่ยงปัญหา path ภาษาไทยของ mediapipe
        with open(resource_path(rel), "rb") as f:
            return f.read()

    detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_buffer=buf(DETECTOR_MODEL)),
        min_detection_confidence=0.5))
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=buf(LANDMARKER_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        output_face_blendshapes=True))
    return mp, detector, landmarker


PERSON_MODEL = "models/efficientdet_lite0.tflite"
_persondet = None


def _load_person_detector():
    """ตัวตรวจจับ "คน" (ไม่บังคับ): ไว้รู้ว่าตัวเอกตัวใหญ่หันหลังอยู่ ไม่ให้ไปตัดสินหน้าคนข้างหลังแทน"""
    from mediapipe.tasks.python import BaseOptions, vision
    with open(resource_path(PERSON_MODEL), "rb") as f:
        buf = f.read()
    return vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.IMAGE,
        score_threshold=0.3, category_allowlist=["person"], max_results=25))


def init_worker() -> None:
    """เรียกครั้งเดียวต่อโปรเซส"""
    global _models, _persondet
    cv2.setNumThreads(1)
    if models_available():
        _models = _load_models()
        if os.path.exists(resource_path(PERSON_MODEL)):
            try:
                _persondet = _load_person_detector()
            except Exception:  # noqa: BLE001 - ไม่มี/โหลดไม่ได้ก็ทำงานต่อได้ (แค่ไม่รู้ว่าตัวเอกหันหลัง)
                _persondet = None


def detect_persons(full: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """คนในภาพ [(x, y, w, h, คะแนน)] เป็นสัดส่วนของภาพ (0-1)"""
    if _persondet is None or _models is None:
        return []
    mp = _models[0]
    h, w = full.shape[:2]
    s = min(1.0, DETECT_LONG_SIDE / max(h, w))
    small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full
    sh, sw = small.shape[:2]
    res = _persondet.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))))
    return [(d.bounding_box.origin_x / sw, d.bounding_box.origin_y / sh, d.bounding_box.width / sw,
             d.bounding_box.height / sh, d.categories[0].score) for d in res.detections]


def read_image(path: str) -> np.ndarray:
    """อ่าน JPG (path ภาษาไทยได้) แบบย่อครึ่งเพื่อความเร็ว; EXIF orientation ถูกจัดการให้"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
    if img is not None and max(img.shape[:2]) < DETECT_LONG_SIDE * 0.9:
        full = cv2.imdecode(data, cv2.IMREAD_COLOR)  # ไฟล์เล็กอยู่แล้ว อย่าย่อซ้ำ
        if full is not None:
            img = full
    if img is None:
        raise ValueError("อ่านไฟล์รูปไม่ได้")
    return img


def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def image_sharpness(gray: np.ndarray) -> float:
    """ความคมของภาพ = เปอร์เซนไทล์ที่ 90 ของความคมรายช่อง
    ฉากหลังเบลอ/ผนังเรียบทั้งช่องไม่ทำให้ทั้งรูปถูกมองว่าเบลอ"""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = gray.shape
    vals = [
        float(lap[i * h // GRID:(i + 1) * h // GRID, j * w // GRID:(j + 1) * w // GRID].var())
        for i in range(GRID) for j in range(GRID)
    ]
    return float(np.percentile(vals, 90))


def expression_scores(b: dict[str, float]) -> tuple[float, float]:
    """คืน (blink, funny) จาก blendshape"""
    g = b.get
    blink = (g("eyeBlinkLeft", 0.0) + g("eyeBlinkRight", 0.0)) / 2
    smile = (g("mouthSmileLeft", 0.0) + g("mouthSmileRight", 0.0)) / 2
    # ทดสอบกับรูปจริงแล้ว: mouthPucker/mouthShrug/mouthRoll ติดกับหน้านิ่งๆ, eyeLookIn ติดตอนหันหน้า,
    # tongueOut/cheekPuff ไม่มีค่า -> ใช้ได้แค่ "อ้าปากค้างโดยไม่ได้ยิ้ม" (ส่วนใหญ่คือกำลังพูด/ร้อง)
    # jawOpen ในโมเดลนี้มีค่าสูงสุดราว 0.35 จึงคูณ 2.4 ให้อยู่ในสเกล 0-1
    funny = min(1.0, g("jawOpen", 0.0) * 2.4) if smile < 0.3 else 0.0
    return float(blink), float(funny)


def _detect_boxes(full: np.ndarray) -> list[tuple[float, float, float, float]]:
    """หาหน้าทั้งภาพ (x, y, w, h) ในพิกัดภาพเต็ม
    BlazeFace ย่อทุกอินพุตเหลือ 128x128 จึงตรวจทั้งภาพ (หน้าใหญ่) + แบ่งช่องซ้อนกัน (หน้าเล็กในรูปเวที/รูปหมู่)"""
    mp, detector, _ = _models
    h, w = full.shape[:2]
    boxes: list[tuple[float, float, float, float, float]] = []  # + score

    def run(img: np.ndarray, ox: float, oy: float, s: float) -> None:
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for d in detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).detections:
            b = d.bounding_box
            boxes.append((ox + b.origin_x * s, oy + b.origin_y * s, b.width * s, b.height * s,
                          d.categories[0].score))

    scale = max(1.0, max(h, w) / DETECT_LONG_SIDE)
    small = cv2.resize(full, (round(w / scale), round(h / scale)), interpolation=cv2.INTER_AREA) if scale > 1 else full
    run(small, 0, 0, scale)

    stride = TILE_SIZE - TILE_OVERLAP
    xs = list(range(0, max(1, w - TILE_OVERLAP), stride))
    ys = list(range(0, max(1, h - TILE_OVERLAP), stride))
    for y0 in ys:
        for x0 in xs:
            x0c, y0c = min(x0, max(0, w - TILE_SIZE)), min(y0, max(0, h - TILE_SIZE))
            run(full[y0c:y0c + TILE_SIZE, x0c:x0c + TILE_SIZE], x0c, y0c, 1.0)
    return _nms(boxes)


def _nms(boxes: list[tuple[float, float, float, float, float]], iou_thr: float = 0.3):
    kept: list[tuple[float, float, float, float, float]] = []
    for b in sorted(boxes, key=lambda b: b[4], reverse=True):
        if all(_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return [k[:4] for k in kept]


def frontal_score(pts: np.ndarray) -> float:
    """หันเข้ากล้องแค่ไหน จากตำแหน่งปลายจมูก (จุด 1) เทียบขอบแก้มซ้าย-ขวา (จุด 234, 454)"""
    left, right = pts[234][0], pts[454][0]
    span = right - left
    if span <= 1e-6:
        return 0.0
    yaw = abs((pts[1][0] - left) / span - 0.5)     # 0 = ตรง, 0.5 = หันข้างสุด
    return float(max(0.0, 1.0 - yaw / 0.25))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def prominence(x: float, y: float, bw: float, bh: float, w: int, h: int) -> float:
    """ความเด่นของหน้า = สัดส่วนพื้นที่หน้าต่อภาพ x น้ำหนักความใกล้กลางภาพ (กลาง = 1.0, มุมภาพ = 0.5)
    หน้าที่ใหญ่กว่าเป็นสองเท่าแต่อยู่ขอบภาพยังชนะหน้าเล็กกลางภาพ"""
    dx = ((x + bw / 2) / w - 0.5) / 0.5
    dy = ((y + bh / 2) / h - 0.5) / 0.5
    dist = min(1.0, (dx * dx + dy * dy) ** 0.5 / 2 ** 0.5)
    return (bw * bh) / (w * h) * (1.0 - 0.5 * dist)


def subject_score(f: Face) -> float:
    """คะแนนความเป็น "ตัวเอกของภาพ" = ความเด่น (พื้นที่ x ใกล้กลางภาพ) ลดถ้าหน้าโดนตัดขอบภาพ
    ลองถ่วงน้ำหนักด้วยหันหน้าตรง (frontal) และก้มมอง (lookdown) แล้ว: ตรวจด้วยตา 48 รูปที่คนเด่นเปลี่ยน
    ผิดถึง 11 (10 รูปเคยถูก) เพราะผู้ประกวดชุดกระดาษมักเอียง/ก้มหน้า แต่สตาฟข้างๆ มองกล้องตรง จึงไม่ใช้"""
    x, y, bw, bh = f.box
    cut = x < 0.01 or y < 0.01 or x + bw > 0.99 or y + bh > 0.99
    return f.prominence * (0.7 if cut else 1.0)


def order_faces(faces: list[Face]) -> list[Face]:
    """เรียงให้ faces[0] = คนเด่นที่สุด โดยหน้าที่โมเดลสีหน้ายืนยันว่าเป็นหน้าจริง (has_expr) มาก่อนเสมอ
    เพราะ face detector เจอ "หน้า" ปลอมบนตึก/ป้าย/ชุด และมักมีขนาดใหญ่จนชนะคนจริง (ทดสอบกับรูปจริงแล้ว)"""
    return sorted(faces, key=lambda f: (f.has_expr, subject_score(f)), reverse=True)


def blur_metric(gray: np.ndarray) -> float:
    """ตัววัดความเบลอแบบ Crete-Roffet (0 = คม, 1 = เบลอ) เทียบความชันของภาพกับภาพที่ถูกทำให้เบลอซ้ำ
    ไม่ขึ้นกับความคมของเนื้อผ้า/ผิว (ค่ากลางของ Laplacian variance ขึ้นกับเนื้อภาพมาก)"""
    g = gray.astype(np.float32)
    out = 0.0
    for axis, ksize in ((1, (9, 1)), (0, (1, 9))):
        d = np.abs(np.diff(g, axis=axis))
        db = np.abs(np.diff(cv2.blur(g, ksize), axis=axis))
        out = max(out, 1.0 - float(np.maximum(0.0, d - db).sum()) / (float(d.sum()) + 1e-9))
    return out


# จุดบนใบหน้า (MediaPipe FaceMesh) ของตาซ้าย/ขวา: [หัวตา, หางตา, เปลือกบน1, ล่าง1, บน2, ล่าง2]
_EYES = ([33, 133, 159, 145, 158, 153], [362, 263, 386, 374, 385, 380])


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR ของตาสองข้างเฉลี่ย (ยิ่งต่ำยิ่งหลับ) คำนวณจากจุดพิกเซลจริง ไม่บิดสัดส่วน"""
    vals = []
    for a, b, u1, l1, u2, l2 in _EYES:
        width = np.linalg.norm(pts[a] - pts[b]) + 1e-9
        vals.append((np.linalg.norm(pts[u1] - pts[l1]) + np.linalg.norm(pts[u2] - pts[l2])) / (2 * width))
    return float(np.mean(vals))


# ลองครอปหน้าตามลำดับ (margin, ด้านยาว) จนกว่าโมเดลใบหน้าจะจับได้ — ทดลองแล้วครอปแบบเดียวจับได้ ~70%, ลองสามแบบได้ ~75%
CROP_CONFIGS = ((CROP_MARGIN, None), (0.8, 384), (0.2, 256))
MIN_LANDMARK_IOU = 0.3    # กรอบจากจุดใบหน้าต้องทับกรอบที่ detector เจอ ไม่งั้นถือว่าเป็นหน้าปลอม/คนอื่นในครอป


def _run_landmarker(full: np.ndarray, x: float, y: float, bw: float, bh: float,
                    margin: float, side: int | None):
    """คืน (blendshapes, จุดใบหน้า 478 จุดเป็นพิกเซลในภาพเต็ม) หรือ None ถ้าโมเดลจับหน้าไม่ได้"""
    mp, _, landmarker = _models
    h, w = full.shape[:2]
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    crop = full[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    longest = max(crop.shape[:2])
    target = side or min(CROP_MAX_SIDE, max(CROP_MIN_SIDE, longest))
    f = target / longest
    if f != 1:
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    if not res.face_blendshapes or not res.face_landmarks:
        return None
    ch, cw = crop.shape[:2]
    pts = np.array([(p.x * cw, p.y * ch) for p in res.face_landmarks[0]]) / f + (x0, y0)
    return {c.category_name: c.score for c in res.face_blendshapes[0]}, pts


def _analyze_faces(full: np.ndarray) -> list[Face]:
    h, w = full.shape[:2]
    gray_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)

    faces: list[Face] = []
    for x, y, bw, bh in _detect_boxes(full):
        if bw < MIN_FACE_PX:
            continue

        found = None
        for margin, side in CROP_CONFIGS:
            found = _run_landmarker(full, x, y, bw, bh, margin, side)
            if found:
                lx0, ly0 = found[1].min(axis=0)
                lx1, ly1 = found[1].max(axis=0)
                if _iou((x, y, bw, bh), (lx0, ly0, lx1 - lx0, ly1 - ly0)) >= MIN_LANDMARK_IOU:
                    break
                found = None  # หน้าที่โมเดลเจอไม่ใช่ที่ detector ชี้ (เช่นกรอบบนอก/ตึกที่ครอปไปทับคนอื่น)

        blink = funny = 0.0
        ear, lookdown, frontal = 0.3, 0.0, 1.0
        if found:
            blend, pts = found
            blink, funny = expression_scores(blend)
            ear = eye_aspect_ratio(pts)
            frontal = frontal_score(pts)
            lookdown = (blend.get("eyeLookDownLeft", 0.0) + blend.get("eyeLookDownRight", 0.0)) / 2
            (x, y), (x2, y2) = pts.min(axis=0), pts.max(axis=0)   # ใช้กรอบจากจุดใบหน้า แม่นกว่ากรอบ detector
            bw, bh = x2 - x, y2 - y

        fx0, fy0 = max(0, int(x)), max(0, int(y))
        face_gray = gray_full[fy0:min(h, int(y + bh)), fx0:min(w, int(x + bw))]
        if face_gray.size == 0:
            continue
        interp = cv2.INTER_AREA if max(face_gray.shape) > FACE_SHARP_SIZE else cv2.INTER_LINEAR
        face_gray = cv2.resize(face_gray, (FACE_SHARP_SIZE, FACE_SHARP_SIZE), interpolation=interp)
        faces.append(Face(
            size=int(bw), blink=blink, funny=funny, sharp=_lap_var(face_gray),
            box=(x / w, y / h, bw / w, bh / h), prominence=prominence(x, y, bw, bh, w, h),
            has_expr=found is not None, ear=ear, lookdown=lookdown, blur=blur_metric(face_gray),
            frontal=frontal))

    # หน้าจริงหน้าเดียวอาจถูก detector เจอหลายกรอบ (ช่องซ้อนกัน) กรอบที่ปรับแล้วจะทับกัน -> เก็บอันเดียว
    faces = order_faces(faces)
    kept: list[Face] = []
    for f in faces:
        if f.has_expr and any(k.has_expr and _iou(f.box, k.box) > 0.5 for k in kept):
            continue
        kept.append(f)
    return kept


def dhash64(gray: np.ndarray) -> int:
    """dHash 64 บิต: เทียบความสว่างพิกเซลข้างเคียงในภาพย่อ 9x8 (ภาพหน้าตาเหมือนกันได้ค่าใกล้กัน)"""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_taken(path: str) -> float | None:
    """เวลาถ่ายจาก EXIF (DateTimeOriginal + SubSec) เป็นวินาที; ไม่มี/อ่านไม่ได้ = None"""
    try:
        from datetime import datetime
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            raw = sub.get(36867) or ex.get(306)
            if not raw:
                return None
            t = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
            frac = str(sub.get(37521, "")).strip()
            return t + (float("0." + frac) if frac.isdigit() else 0.0)
    except Exception:  # noqa: BLE001
        return None


def analyze(path: str, camera: str) -> Metrics:
    m = Metrics(path=path, camera=camera)
    try:
        m.taken = read_taken(path)
        full = read_image(path)
        h, w = full.shape[:2]
        scale = max(1.0, max(h, w) / DETECT_LONG_SIDE)
        gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        if scale > 1:
            gray = cv2.resize(gray, (round(w / scale), round(h / scale)), interpolation=cv2.INTER_AREA)
        m.sharp = image_sharpness(gray)
        m.blur = blur_metric(gray)
        m.dhash = dhash64(gray)
        if _models is not None:
            m.faces = _analyze_faces(full)
            m.persons = detect_persons(full)
    except Exception as e:  # noqa: BLE001 - รูปเสียหนึ่งรูปต้องไม่ทำให้ทั้งงานล้ม
        m.error = f"{type(e).__name__}: {e}"
    return m
