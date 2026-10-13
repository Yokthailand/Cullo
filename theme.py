"""ธีมมืดแบบโปรแกรมคัดรูป/แต่งรูป (สี ฟอนต์ สเกลจอ ตัวช่วยวาด)

หลักการสี: พื้นเป็นเทากลางไล่ 4 ชั้น (กริดมืดสุดให้รูปเด่น) น้ำเงินใช้กับ "เลือก/โฟกัส/ปุ่มหลัก" เท่านั้น
เขียว/แดงสงวนไว้ให้ ผ่าน/คัดออก เหตุผลเบลอ/หลับตา/หน้าตลกใช้แค่จุดสีเล็กๆ
"""
from __future__ import annotations

BG = "#0f0f11"          # กริด/พื้นหน้าต่าง (มืดสุด ให้รูปเด่น)
PANEL = "#1a1a1d"       # แผงข้าง
PANEL_HI = "#26262b"    # คอนโทรล/การ์ด
LINE = "#2f2f35"        # เส้นแบ่ง/ขอบ
LINE_HI = "#45454d"     # ขอบตอนชี้/ปุ่มรอง
CELL = "#161619"        # พื้นช่องรูปย่อ
CELL_HOVER = "#202024"
TEXT = "#ececf0"
TEXT2 = "#c9cad1"       # หัวข้อส่วน
MUTED = "#a1a2ab"       # ข้อความรอง (คอนทราสต์ >= 4.5:1 บน PANEL)
ACCENT = "#3d8bff"
ACCENT_HI = "#5a9dff"
SELECTED = "#3a3a42"    # แท็บ/ตัวเลือกที่เลือกอยู่
GOOD = "#3dd68c"
GOOD_DK = "#1f7a55"
BAD = "#ff6161"
BAD_DK = "#8a2f2f"
WARN = "#f5a524"

REASON_COLORS = {
    "เบลอ": "#ff9f43",
    "หลับตา": "#b18cff",
    "หน้าตลก": "#ff7eb6",
    "อ่านไฟล์ไม่ได้": "#8b8c95",
}

FONT = "Leelawadee UI"

SCALE = 1.0   # สเกลจอของ Windows (1.25 = 125%) ตั้งตอนเปิดแอป ใช้กับสิ่งที่วาดเองบน canvas


def set_scale(s: float) -> None:
    global SCALE
    SCALE = max(1.0, float(s))


def px(n: float) -> int:
    return int(round(n * SCALE))


def font(size: int = 13, weight: str = "normal") -> tuple:
    """ตัวอักษรไทยเล็กแล้วสระ/วรรณยุกต์เบลอ: เนื้อหา 13, รอง 12 ห้ามต่ำกว่า 12"""
    return (FONT, size, weight)


def rounded_rect(canvas, x0, y0, x1, y1, r=6, **kw):
    """วาดสี่เหลี่ยมมุมมนบน tk.Canvas (ใช้ polygon แบบ smooth)"""
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
           x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


def shorten_path(path: str, keep_tail: int = 2, max_len: int = 52) -> str:
    """ตัดกลางพาธยาวๆ: C:\\Users\\…\\ชื่อโฟลเดอร์\\งาน"""
    parts = str(path).replace("/", "\\").split("\\")
    if len(str(path)) <= max_len or len(parts) <= keep_tail + 1:
        return str(path)
    return parts[0] + "\\…\\" + "\\".join(parts[-keep_tail:])
