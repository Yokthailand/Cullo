"""ปรับสูตรตัดสิน (config.W_FACE_SHARP / W_BLINK / BIAS) จากรายงาน.csv ที่คนคัดและส่งออกจากแอปจริง

ใช้: python dev-tools/refit_from_reports.py "<โฟลเดอร์ผลลัพธ์1>\\รายงาน.csv" ["<โฟลเดอร์ผลลัพธ์2>\\รายงาน.csv" ...]

ป้าย = ผลสุดท้ายหลังคนตรวจ (โฟลเดอร์ "คัดออก" = เสีย) เฉพาะแถวที่คอลัมน์ "ตรวจแล้ว" เพราะแถวที่ผู้ใช้ไม่เคยดูเป็นแค่คำตัดสินของโปรแกรมเอง
(ฝึกจากมันจะวนซ้ำตัวเอง) รูปที่ผู้ใช้กลับคำตัดสิน ("แก้ด้วยมือ") ให้น้ำหนักมากกว่า เพราะเป็นข้อมูลที่โปรแกรมเคยพลาดจริง
ต้องใช้ scikit-learn (pip install scikit-learn) ไม่ได้แพ็กใน .exe
"""
import csv
import math
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

C_FOLDER, C_MANUAL, C_VIEWED = "exported_keep", "แก้ด้วยมือ", "ตรวจแล้ว"   # exported_keep: 1 = ผ่าน, 0 = คัดออก (ผลสุดท้ายหลังคนแก้)
C_RATIO, C_BLINK = "face_sharp_ratio", "main_blink"

X, y, w = [], [], []
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if not r.get(C_VIEWED) or not r.get(C_RATIO) or not r.get(C_BLINK):
            continue                                       # ไม่เคยดู / ไม่มีคนเด่นให้ประเมิน
        X.append([math.log(float(r[C_RATIO]) + 1e-3), float(r[C_BLINK])])
        y.append(r[C_FOLDER] == "0")
        w.append(4.0 if r.get(C_MANUAL) else 1.0)
X, y, w = np.array(X), np.array(y), np.array(w)
print(f"แถวที่ใช้ได้: {len(y)} | คัดออก: {int(y.sum())} | แก้ด้วยมือ: {int((w > 1).sum())}")
if len(y) < 60 or y.sum() < 8 or (~y).sum() < 8:
    sys.exit("ข้อมูลน้อยเกินไป (แนะนำ >= 200 แถวที่ตรวจแล้ว และมีทั้งผ่าน/คัดออกอย่างน้อยฝั่งละ 20)")
m = LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000).fit(X, y, sample_weight=w)
print(f"W_FACE_SHARP = {m.coef_[0][0]:.4f}\nW_BLINK = {m.coef_[0][1]:.4f}\nBIAS = {m.intercept_[0]:.4f}")
p = 1 / (1 + np.exp(-(X @ m.coef_[0] + m.intercept_[0])))
for rate in (0.03, 0.05, 0.07, 0.10, 0.12, 0.15):
    th = np.quantile(p, 1 - rate)
    sel = p >= th
    print(f"อัตราคัดออก {rate:.0%}: reject_p >= {th:.3f}  precision={(sel & y).sum() / sel.sum():.2f} recall={(sel & y).sum() / y.sum():.2f}")
print("\nเอาค่า W_* / BIAS ไปแทนใน config.py และตั้ง reject_p ของแต่ละระดับตามตารางด้านบน (ควรวัดซ้ำด้วยผู้ตัดสินอิสระก่อนใช้จริง)")

