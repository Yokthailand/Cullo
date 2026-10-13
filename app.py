"""Cullo — หน้าตาแบบโปรแกรมคัดรูป/แต่งรูป (ธีมมืด กริดรูปย่อ พรีวิว ปรับเกณฑ์แล้วเห็นผลทันที)"""
from __future__ import annotations

import json
import multiprocessing
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import ImageTk

import icons
import theme
from bursts import Burst, find_bursts
from config import DEFAULT_PRESET, PRESETS, REASON_BLUR, REASON_EYES, REASON_FUNNY
from gallery import Gallery, Item
from judge import Verdict, judge_detailed, usable_main
from pipeline import Cancelled, Summary, analyze_folder, export, make_out_dir, run
from scorer import Metrics, models_available
from thumbs import ThumbCache, load_full, load_image

PRESET_ORDER = ["ผ่อนปรน", "ปกติ", "เข้มงวด"]
PRESET_HINT = {
    "ผ่อนปรน": "คัดเฉพาะรูปที่เสียชัดเจน",
    "ปกติ": "สมดุล (แนะนำ)",
    "เข้มงวด": "คัดมากขึ้น อาจมีรูปดีปนไป",
}
ALL_CAMERAS = "ทุกกล้อง"
ALL_REASONS = "ทุกเหตุผล"
RECENT_FILE = Path(os.environ.get("APPDATA", str(Path.home()))) / "Cullo" / "recent.json"
THAI_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


def load_recent() -> list[dict]:
    try:
        rows = json.loads(RECENT_FILE.read_text(encoding="utf-8"))
        return [r for r in rows if Path(r["path"]).is_dir()][:3]      # โฟลเดอร์ที่ยังอยู่จริงเท่านั้น
    except Exception:  # noqa: BLE001
        return []


def save_recent(folder: Path, count: int) -> None:
    try:
        rows = [r for r in load_recent() if r["path"] != str(folder)]
        rows.insert(0, {"path": str(folder), "count": count, "when": time.time()})
        RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_FILE.write_text(json.dumps(rows[:5], ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 - จำโฟลเดอร์ล่าสุดไม่ได้ ไม่กระทบการทำงานหลัก
        pass


def btn(master, text, command, kind="ghost", **kw):
    """ปุ่มสไตล์เดียวกันทั้งแอป: primary(น้ำเงิน) good(เขียว) bad(แดง) ghost(เทา)"""
    styles = {
        "primary": dict(fg_color=theme.ACCENT, hover_color=theme.ACCENT_HI, text_color="#ffffff"),
        "good": dict(fg_color=theme.GOOD_DK, hover_color="#26946a", text_color="#d9fff0", border_width=1, border_color=theme.GOOD),
        "bad": dict(fg_color=theme.BAD_DK, hover_color="#a63a3a", text_color="#ffe0e0", border_width=1, border_color=theme.BAD),
        "ghost": dict(fg_color="transparent", hover_color=theme.PANEL_HI, text_color=theme.TEXT, border_width=1, border_color=theme.LINE_HI),
    }
    opts = dict(corner_radius=8, height=34, font=theme.font(12, "bold" if kind != "ghost" else "normal"))
    opts.update(styles[kind])
    opts.update(kw)
    return ctk.CTkButton(master, text=text, command=command, **opts)


class App(ctk.CTk):
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=theme.BG)
        theme.set_scale(ctk.ScalingTracker.get_window_scaling(self))   # สเกลจอ Windows (125% ฯลฯ)
        self.title("Cullo")
        self.geometry("1280x780")
        self.minsize(1100, 680)

        # ----- สถานะ -----
        self.q: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.root_dir: Path | None = None
        self.paths: list[Path] = []
        self.metrics: list[Metrics] = []
        self.verdicts: list[Verdict] = []
        self.overrides: dict[int, bool] = {}       # idx -> ผู้ใช้สั่ง keep(True)/reject(False) ทับโปรแกรม
        self.history: list[list[tuple[int, bool | None]]] = []   # ไว้เลิกทำ (Ctrl+Z) แต่ละรายการ = การแก้ 1 ครั้ง (อาจหลายรูป)
        self.viewed: set[int] = set()              # รูปที่ผู้ใช้เปิดดู/ตัดสินแล้ว (ไว้เตือนก่อนส่งออก)
        self.bursts: list[Burst] = []
        self.burst_of: dict[int, tuple[int, Burst]] = {}     # idx -> (ลำดับกอง 1..N, กอง)
        self._view_job = None
        self.preset_name = DEFAULT_PRESET
        self.tab = "reject"                        # reject | pass | burst | all
        self.viewer: Viewer | None = None
        self._pv_img = None
        self._export_dir: Path | None = None

        self.thumbs = ThumbCache(self, on_ready=lambda: getattr(self, "gallery", None) and self.gallery.refresh())
        self.thumbs.set_size(theme.px(200))

        self.stage = ctk.CTkFrame(self, fg_color=theme.BG, corner_radius=0)
        self.stage.pack(fill="both", expand=True)
        self.stage.grid_rowconfigure(0, weight=1)
        self.stage.grid_columnconfigure(0, weight=1)
        self.screens = {"welcome": self._build_welcome(), "analyzing": self._build_analyzing(),
                        "review": self._build_review()}
        self._refresh_recent()
        self.show("welcome")

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(80, self._poll)

    def show(self, name: str) -> None:
        for n, f in self.screens.items():
            f.grid(row=0, column=0, sticky="nsew") if n == name else f.grid_remove()

    def _close(self) -> None:
        self.cancel.set()
        self.thumbs.shutdown()
        self.destroy()
        os._exit(0)   # ให้ process วิเคราะห์ที่ค้างอยู่ปิดตามทันที

    # =====================================================================
    # หน้าต้อนรับ
    # =====================================================================
    def _build_welcome(self) -> ctk.CTkFrame:
        f = ctk.CTkFrame(self.stage, fg_color=theme.BG, corner_radius=0)
        card = ctk.CTkFrame(f, fg_color=theme.PANEL, corner_radius=20, border_width=1, border_color=theme.LINE)
        card.place(relx=0.5, rely=0.5, anchor="center")
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=64, pady=48)

        logo = tk.Label(inner, image=icons.logo(theme.px(84)), bg=theme.PANEL, bd=0)
        logo.pack()
        ctk.CTkLabel(inner, text="Cullo", font=theme.font(30, "bold"), text_color=theme.TEXT).pack(pady=(18, 2))
        ctk.CTkLabel(inner, text="ตัดรูปหลุดออกให้อัตโนมัติ แล้วคุณตรวจซ้ำก่อนส่งออก", font=theme.font(13),
                     text_color=theme.MUTED).pack()

        chips = ctk.CTkFrame(inner, fg_color="transparent")
        chips.pack(pady=22)
        for text, col in (("เบลอ", "#ff9f43"), ("หลับตา", "#b18cff"), ("หน้าตลก", "#ff7eb6")):
            ctk.CTkLabel(chips, text=text, font=theme.font(12, "bold"), text_color="#111", fg_color=col,
                         corner_radius=12, width=78, height=26).pack(side="left", padx=5)

        btn(inner, "เลือกโฟลเดอร์รูป", self.pick_folder, "primary", height=52, width=320,
            font=theme.font(16, "bold"), corner_radius=10).pack(pady=(4, 12))
        self.recent_box = ctk.CTkFrame(inner, fg_color="transparent")     # รายการโฟลเดอร์ล่าสุด (เติมใน _refresh_recent)
        self.recent_box.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(inner, text="รองรับไฟล์ JPG · ถ้าแยกโฟลเดอร์ย่อยตามกล้องจะเทียบความคมแยกกล้องให้",
                     font=theme.font(12), text_color=theme.MUTED).pack()
        self.welcome_note = ctk.CTkLabel(inner, text="", font=theme.font(12), text_color=theme.WARN)
        self.welcome_note.pack(pady=(10, 0))
        if not models_available():
            self.welcome_note.configure(text="⚠ ไม่พบไฟล์โมเดลตรวจหน้า จะคัดได้เฉพาะรูปเบลอ")
        return f

    def _refresh_recent(self) -> None:
        for w in self.recent_box.winfo_children():
            w.destroy()
        rows = load_recent()
        if not rows:
            return
        ctk.CTkLabel(self.recent_box, text="เปิดล่าสุด", font=theme.font(12, "bold"), text_color=theme.TEXT2,
                     anchor="w").pack(fill="x", pady=(0, 4))
        for r in rows:
            t = time.localtime(r.get("when", 0))
            when = f"{t.tm_mday} {THAI_MONTHS[t.tm_mon - 1]}"
            p = Path(r["path"])
            b = btn(self.recent_box, f"{p.name}    ·    {r.get('count', 0):,} รูป    ·    {when}",
                    lambda p=p: self.load_folder(p), "ghost", height=34, anchor="w")
            b.pack(fill="x", pady=2)

    # =====================================================================
    # หน้ากำลังวิเคราะห์
    # =====================================================================
    def _build_analyzing(self) -> ctk.CTkFrame:
        f = ctk.CTkFrame(self.stage, fg_color=theme.BG, corner_radius=0)
        card = ctk.CTkFrame(f, fg_color=theme.PANEL, corner_radius=20, border_width=1, border_color=theme.LINE)
        card.place(relx=0.5, rely=0.5, anchor="center")
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=56, pady=44)
        self.an_title = ctk.CTkLabel(inner, text="กำลังวิเคราะห์รูป", font=theme.font(22, "bold"), text_color=theme.TEXT)
        self.an_title.pack()
        self.an_folder = ctk.CTkLabel(inner, text="", font=theme.font(12), text_color=theme.MUTED, wraplength=460)
        self.an_folder.pack(pady=(4, 20))
        self.an_bar = ctk.CTkProgressBar(inner, width=460, height=8, corner_radius=4, progress_color=theme.PANEL_HI,
                                         fg_color=theme.PANEL_HI)
        self.an_bar.set(0)
        self.an_bar.pack()
        self.an_count = ctk.CTkLabel(inner, text="", font=theme.font(13), text_color=theme.TEXT)
        self.an_count.pack(pady=(14, 2))
        ctk.CTkLabel(inner, text="ตรวจความคม หาหน้า และดูว่าตาลืมหรือหลับ — ใช้เวลาราว 1–2 นาทีต่อ 500 รูป",
                     font=theme.font(12), text_color=theme.MUTED).pack(pady=(0, 20))
        btn(inner, "ยกเลิก", self.cancel_analysis, "ghost", width=120).pack()
        return f

    # =====================================================================
    # หน้าคัดรูป
    # =====================================================================
    def _build_review(self) -> ctk.CTkFrame:
        f = ctk.CTkFrame(self.stage, fg_color=theme.BG, corner_radius=0)
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(0, weight=1)
        self._build_sidebar(f).grid(row=0, column=0, sticky="ns")
        self._build_center(f).grid(row=0, column=1, sticky="nsew")
        self._build_preview(f).grid(row=0, column=2, sticky="ns")
        return f

    def _section(self, master, text: str) -> None:
        ctk.CTkLabel(master, text=text, font=theme.font(12, "bold"), text_color=theme.TEXT2, anchor="w"
                     ).pack(fill="x", padx=20, pady=(14, 5))

    def _build_sidebar(self, master) -> ctk.CTkFrame:
        sb = ctk.CTkFrame(master, fg_color=theme.PANEL, corner_radius=0, width=240)
        sb.pack_propagate(False)

        head = ctk.CTkFrame(sb, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=(20, 4))
        ctk.CTkLabel(head, text="Cullo", font=theme.font(20, "bold"), text_color=theme.TEXT, anchor="w").pack(fill="x")
        ctk.CTkLabel(head, text="สำหรับชมรมโสต", font=theme.font(12), text_color=theme.MUTED, anchor="w").pack(fill="x")

        bottom = ctk.CTkFrame(sb, fg_color="transparent")     # จองที่ส่วนล่างก่อน ไม่ให้โดนตัดเมื่อจอเตี้ย
        bottom.pack(side="bottom", fill="x", padx=16, pady=16)

        self._section(sb, "โฟลเดอร์")
        card = ctk.CTkFrame(sb, fg_color=theme.PANEL_HI, corner_radius=10)
        card.pack(fill="x", padx=16)
        self.sb_folder = ctk.CTkLabel(card, text="", font=theme.font(13, "bold"), text_color=theme.TEXT, anchor="w")
        self.sb_folder.pack(fill="x", padx=14, pady=(10, 0))
        self.sb_count = ctk.CTkLabel(card, text="", font=theme.font(12), text_color=theme.MUTED, anchor="w")
        self.sb_count.pack(fill="x", padx=14)
        self.sb_viewed = ctk.CTkLabel(card, text="", font=theme.font(12), text_color=theme.TEXT2, anchor="w")
        self.sb_viewed.pack(fill="x", padx=14, pady=(6, 0))
        self.sb_viewed_bar = ctk.CTkProgressBar(card, height=5, corner_radius=3, fg_color=theme.LINE,
                                                progress_color=theme.GOOD)
        self.sb_viewed_bar.set(0)
        self.sb_viewed_bar.pack(fill="x", padx=14, pady=(3, 0))
        btn(card, "เปลี่ยนโฟลเดอร์", self.pick_folder, "ghost", height=28,
            ).pack(fill="x", padx=12, pady=(10, 12))

        self._section(sb, "ความเข้มงวด")
        self.seg_preset = ctk.CTkSegmentedButton(
            sb, values=PRESET_ORDER, command=self.on_preset, height=34, corner_radius=8, font=theme.font(12),
            fg_color=theme.PANEL_HI, selected_color=theme.SELECTED, selected_hover_color=theme.LINE_HI,
            unselected_color=theme.PANEL_HI, unselected_hover_color=theme.LINE, text_color=theme.TEXT)
        self.seg_preset.set(self.preset_name)
        self.seg_preset.pack(fill="x", padx=16)
        self.preset_hint = ctk.CTkLabel(sb, text=PRESET_HINT[self.preset_name], font=theme.font(12),
                                        text_color=theme.MUTED, anchor="w", justify="left")
        self.preset_hint.pack(fill="x", padx=20, pady=(6, 0))

        self._section(sb, "ตรวจอะไรบ้าง")
        self.chk = {REASON_BLUR: ctk.BooleanVar(value=True), REASON_EYES: ctk.BooleanVar(value=True),
                    REASON_FUNNY: ctk.BooleanVar(value=False)}       # หน้าตลกยังไม่แม่น ปิดไว้ก่อน
        for kind, var in self.chk.items():
            ctk.CTkCheckBox(sb, text=kind + ("  (ทดลอง)" if kind == REASON_FUNNY else ""), variable=var,
                            command=self.on_rejudge, font=theme.font(13),
                            text_color=theme.TEXT, fg_color=theme.REASON_COLORS[kind], hover_color=theme.REASON_COLORS[kind],
                            border_color=theme.LINE_HI, checkmark_color="#111", checkbox_width=20, checkbox_height=20,
                            corner_radius=5).pack(fill="x", padx=20, pady=(0, 6))
        self.main_only = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(sb, text="ดูเฉพาะคนเด่นที่สุดในภาพ", variable=self.main_only, command=self.on_rejudge,
                      font=theme.font(12), text_color=theme.TEXT, progress_color=theme.ACCENT,
                      button_color="#ffffff", fg_color=theme.LINE).pack(fill="x", padx=20, pady=(6, 0))
        self.switch_hint = ctk.CTkLabel(sb, text="", font=theme.font(12), height=0)   # จุดอ้างอิงตำแหน่งกรองกล้อง
        self.switch_hint.pack()

        self.cam_wrap = ctk.CTkFrame(sb, fg_color="transparent")
        ctk.CTkLabel(self.cam_wrap, text="กรองตามกล้อง", font=theme.font(12, "bold"), text_color=theme.MUTED,
                     anchor="w").pack(fill="x", padx=20, pady=(18, 6))
        self.cam_menu = self._menu(self.cam_wrap, [ALL_CAMERAS], self.on_filter)
        self.cam_menu.pack(fill="x", padx=16)

        # ส่วนล่าง: สถิติ + ปุ่มส่งออก
        self.st_line = ctk.CTkLabel(bottom, text="", font=theme.font(13), text_color=theme.TEXT2)
        self.st_line.pack(pady=(0, 8))
        btn(bottom, "ส่งออกรูป", self.open_export, "primary", height=44, font=theme.font(14, "bold"),
            corner_radius=10).pack(fill="x")
        return sb

    def _menu(self, master, values, command) -> ctk.CTkOptionMenu:
        return ctk.CTkOptionMenu(master, values=values, command=command, height=32, corner_radius=8,
                                 font=theme.font(12), dropdown_font=theme.font(12), fg_color=theme.PANEL_HI,
                                 button_color=theme.LINE, button_hover_color="#3c3d44", text_color=theme.TEXT,
                                 dropdown_fg_color=theme.PANEL_HI, dropdown_hover_color=theme.LINE,
                                 dropdown_text_color=theme.TEXT)

    def _build_center(self, master) -> ctk.CTkFrame:
        c = ctk.CTkFrame(master, fg_color=theme.BG, corner_radius=0)
        bar = ctk.CTkFrame(c, fg_color=theme.BG, corner_radius=0)
        bar.pack(fill="x", padx=16, pady=(16, 10))
        self.seg_tab = ctk.CTkSegmentedButton(
            bar, values=["คัดออก", "ผ่าน", "ช็อตรัว", "ทั้งหมด"], command=self.on_tab, height=36, corner_radius=10,
            font=theme.font(13, "bold"), fg_color=theme.PANEL, selected_color=theme.SELECTED,
            selected_hover_color=theme.LINE_HI, unselected_color=theme.PANEL, unselected_hover_color=theme.PANEL_HI,
            text_color=theme.TEXT)
        self.seg_tab.pack(side="left")
        self.reason_menu = self._menu(bar, [ALL_REASONS, "เบลอ", "หลับตา", "หน้าตลก"], self.on_filter)
        self.reason_menu.configure(width=130)
        self.reason_menu.pack(side="left", padx=(12, 0))
        # แท็บช็อตรัว: ปุ่มเลือกใบที่ดีที่สุดของทุกกอง (แทนที่เมนูเหตุผล เพราะไม่ได้ใช้ในแท็บนี้)
        self.burst_btn = btn(bar, "★ เลือกใบดีที่สุดทุกกอง", self.keep_best_all, "ghost", height=36, width=190)

        size = ctk.CTkFrame(bar, fg_color="transparent")
        size.pack(side="right")
        ctk.CTkLabel(size, text="ขนาด", font=theme.font(12), text_color=theme.MUTED).pack(side="left", padx=(0, 8))
        self.size_slider = ctk.CTkSlider(size, from_=120, to=320, number_of_steps=20, width=100, command=self.on_size,
                                         progress_color=theme.ACCENT, button_color="#ffffff",
                                         button_hover_color=theme.ACCENT_HI, fg_color=theme.LINE)
        self.size_slider.set(200)
        self.size_slider.pack(side="left")

        # แถบคีย์ลัด: pack ก่อนกริด (side=bottom) เพื่อจองที่ ไม่ให้กริดกลืนจนตัดแถวล่าง
        hint = ctk.CTkFrame(c, fg_color=theme.PANEL, corner_radius=0, height=40)
        hint.pack(fill="x", side="bottom")
        row = ctk.CTkFrame(hint, fg_color="transparent")
        row.pack(pady=8)
        for key, text in (("P", "ผ่าน"), ("X", "คัดออก"), ("Space", "สลับผ่าน/คัดออก"), ("← ↑ ↓ →", "เลื่อน"),
                          ("Enter", "ดูภาพใหญ่"), ("Ctrl+Z", "เลิกทำ")):
            ctk.CTkLabel(row, text=key, font=theme.font(12, "bold"), text_color=theme.TEXT, fg_color=theme.PANEL_HI,
                         corner_radius=5, height=22).pack(side="left", padx=(12, 4))
            ctk.CTkLabel(row, text=text, font=theme.font(12), text_color=theme.MUTED).pack(side="left")

        self.gallery = Gallery(c, self.thumbs, self.on_select, self.open_viewer, self.on_action)
        self.gallery.pack(fill="both", expand=True, padx=(12, 4), pady=(0, 0))

        # ข้อความแจ้งหลังกดผ่าน/คัดออก + ปุ่มเลิกทำ (ลอยล่างกลางกริด)
        self.toast_box = ctk.CTkFrame(c, fg_color="#2b2b31", corner_radius=10, border_width=1, border_color=theme.LINE_HI)
        self.toast_lbl = ctk.CTkLabel(self.toast_box, text="", font=theme.font(13), text_color=theme.TEXT)
        self.toast_lbl.pack(side="left", padx=(16, 8), pady=8)
        btn(self.toast_box, "เลิกทำ", lambda: self.on_action("undo", Item(-1, "", "", True)), "ghost", width=70,
            height=28).pack(side="left", padx=(0, 10))
        self._toast_job = None
        return c

    def _build_preview(self, master) -> ctk.CTkFrame:
        p = ctk.CTkFrame(master, fg_color=theme.PANEL, corner_radius=0, width=320)
        p.pack_propagate(False)
        self.pv_canvas = tk.Canvas(p, width=theme.px(288), height=theme.px(216), bg="#0b0b0c", highlightthickness=0)
        self.pv_canvas.pack(padx=16, pady=(16, 10))
        self.pv_canvas.bind("<Double-Button-1>", lambda e: self.gallery.current() and self.open_viewer(self.gallery.current()))

        self.pv_name = ctk.CTkLabel(p, text="", font=theme.font(14, "bold"), text_color=theme.TEXT, anchor="w")
        self.pv_name.pack(fill="x", padx=20)
        self.pv_sub = ctk.CTkLabel(p, text="", font=theme.font(12), text_color=theme.MUTED, anchor="w")
        self.pv_sub.pack(fill="x", padx=20)

        self.pv_status = ctk.CTkFrame(p, fg_color="transparent")
        self.pv_status.pack(fill="x", padx=20, pady=(14, 0))

        # กล่องช็อตรัว: แสดงเมื่อรูปนี้อยู่ในกอง (เก็บ 1 ใบที่ดีที่สุด คัดที่เหลือออก)
        self.pv_burst = ctk.CTkFrame(p, fg_color=theme.PANEL_HI, corner_radius=12, border_width=1, border_color="#5c4a1e")
        self.pv_burst_lbl = ctk.CTkLabel(self.pv_burst, text="", font=theme.font(12), text_color="#ffd166", anchor="w",
                                         justify="left")
        self.pv_burst_lbl.pack(fill="x", padx=14, pady=(10, 4))
        btn(self.pv_burst, "★  เก็บใบที่ดีที่สุด คัดที่เหลือออก  (B)", lambda: self._act_current("best"), "ghost",
            height=30).pack(fill="x", padx=12, pady=(0, 10))

        self.pv_card = ctk.CTkFrame(p, fg_color=theme.PANEL_HI, corner_radius=12)
        self.pv_card.pack(fill="x", padx=16, pady=(14, 0))
        ctk.CTkLabel(self.pv_card, text="โอกาสเป็นรูปหลุด", font=theme.font(12), text_color=theme.MUTED, anchor="w"
                     ).pack(fill="x", padx=14, pady=(12, 0))
        row = ctk.CTkFrame(self.pv_card, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(2, 0))
        self.pv_prob = ctk.CTkLabel(row, text="–", font=theme.font(22, "bold"), text_color=theme.TEXT, width=70, anchor="w")
        self.pv_prob.pack(side="left")
        self.pv_bar = ctk.CTkProgressBar(row, height=10, corner_radius=5, fg_color=theme.LINE, progress_color=theme.GOOD)
        self.pv_bar.set(0)
        self.pv_bar.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.pv_sharp = self._kv(self.pv_card, "ความคมของหน้าคนเด่น")
        self.pv_blink = self._kv(self.pv_card, "ตา")
        ctk.CTkFrame(self.pv_card, height=8, fg_color="transparent").pack()

        acts = ctk.CTkFrame(p, fg_color="transparent")
        acts.pack(fill="x", padx=16, pady=(14, 0))
        acts.grid_columnconfigure((0, 1), weight=1)
        btn(acts, "✓  ผ่าน  (P)", lambda: self._act_current("pass"), "good", height=40).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        btn(acts, "✕  คัดออก  (X)", lambda: self._act_current("reject"), "bad", height=40).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.pv_reset = btn(p, "กลับเป็นตามที่โปรแกรมตัดสิน", lambda: self._act_current("auto"), "ghost", height=30)
        return p

    def _kv(self, master, key: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(8, 0))
        ctk.CTkLabel(row, text=key, font=theme.font(12), text_color=theme.MUTED, anchor="w").pack(side="left")
        val = ctk.CTkLabel(row, text="–", font=theme.font(12, "bold"), text_color=theme.TEXT, anchor="e")
        val.pack(side="right")
        return val

    # =====================================================================
    # โฟลเดอร์ / วิเคราะห์
    # =====================================================================
    def pick_folder(self) -> None:
        d = filedialog.askdirectory(title="เลือกโฟลเดอร์รูปกิจกรรม")
        if d:
            self.load_folder(Path(os.path.normpath(d)))

    def load_folder(self, folder: Path) -> None:
        self.root_dir = folder
        self.cancel.clear()
        self.an_folder.configure(text=f"{folder.parent.name}  ›  {folder.name}")
        self.an_bar.set(0)
        self.an_count.configure(text="กำลังสแกนโฟลเดอร์…")
        self.show("analyzing")
        threading.Thread(target=self._analyze_work, args=(folder,), daemon=True).start()

    def _analyze_work(self, folder: Path) -> None:
        try:
            paths, metrics = analyze_folder(folder, lambda s, d, t: self.q.put(("progress", d, t)), self.cancel)
            self.q.put(("analyzed", folder, paths, metrics))
        except Cancelled:
            self.q.put(("cancelled",))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", str(e)))

    def cancel_analysis(self) -> None:
        self.cancel.set()
        self.an_count.configure(text="กำลังยกเลิก…")

    def _poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, d, t = msg
                    self.an_bar.configure(progress_color=theme.ACCENT if d else theme.PANEL_HI)   # ที่ 0% ไม่โผล่จุดฟ้า
                    self.an_bar.set(d / max(t, 1))
                    self.an_count.configure(text=f"{d:,} / {t:,} รูป")
                elif kind == "analyzed":
                    _, folder, paths, metrics = msg
                    self._on_analyzed(folder, paths, metrics)
                elif kind == "cancelled":
                    self.show("review" if self.paths else "welcome")
                elif kind == "error":
                    self.show("review" if self.paths else "welcome")
                    messagebox.showerror("เกิดข้อผิดพลาด", msg[1])
                elif kind == "export_progress":
                    self._export_progress(msg[1], msg[2])
                elif kind == "export_done":
                    self._export_done(msg[1])
                elif kind == "export_error":
                    self._export_error(msg[1])
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _on_analyzed(self, folder: Path, paths: list[Path], metrics: list[Metrics]) -> None:
        self.root_dir, self.paths, self.metrics = folder, paths, metrics
        self.overrides.clear()
        self.history.clear()
        self.viewed.clear()
        save_recent(folder, len(paths))
        self._refresh_recent()
        self.sb_folder.configure(text=folder.name if len(folder.name) <= 22 else folder.name[:20] + "…")
        cams = sorted({m.camera for m in metrics})
        self.sb_count.configure(text=f"{len(paths):,} รูป" + (f" · {len(cams)} กล้อง" if len(cams) > 1 else ""))
        if len(cams) > 1:
            self.cam_menu.configure(values=[ALL_CAMERAS] + [c or "(ไม่ระบุ)" for c in cams])
            self.cam_menu.set(ALL_CAMERAS)
            self.cam_wrap.pack(fill="x", after=self.switch_hint)
        else:
            self.cam_wrap.pack_forget()
        self.cam_menu.set(ALL_CAMERAS)
        self.reason_menu.set(ALL_REASONS)
        self.on_rejudge(reset_view=True)
        self.show("review")
        self.gallery.canvas.focus_set()

    # =====================================================================
    # ตัดสิน / รายการ
    # =====================================================================
    def on_preset(self, name: str) -> None:
        self.preset_name = name
        self.preset_hint.configure(text=PRESET_HINT[name])
        self.on_rejudge()

    def on_rejudge(self, reset_view: bool = False) -> None:
        if not self.metrics:
            return
        enabled = {k for k, var in self.chk.items() if var.get()}
        self.verdicts = judge_detailed(self.metrics, PRESETS[self.preset_name], self.main_only.get(), enabled)
        # ช็อตรัว: จัดกลุ่มแล้วเลือกใบที่ความเสี่ยงต่ำสุด (ไม่มีคนเด่นให้ประเมิน = กลางๆ 0.5)
        risk = [v.prob if v.prob is not None else 0.5 for v in self.verdicts]
        self.bursts = find_bursts(self.metrics, risk)
        self.burst_of = {i: (g + 1, b) for g, b in enumerate(self.bursts) for i in b.members}
        if reset_view:
            self.tab = "reject" if any(v.reasons for v in self.verdicts) else "all"
        self.refresh(select=0 if reset_view else None)

    def keep_of(self, i: int) -> bool:
        return self.overrides.get(i, not self.verdicts[i].reasons)

    def _tab_key(self) -> str:
        return self.tab

    def refresh(self, select: int | None = None) -> None:
        n = len(self.paths)
        keeps = [self.keep_of(i) for i in range(n)]
        n_pass = sum(keeps)
        self.st_line.configure(text=f"ผ่าน {n_pass:,}  ·  คัดออก {n - n_pass:,}"
                                    + (f"  ·  แก้เอง {len(self.overrides):,}" if self.overrides else ""))
        n_burst = sum(len(b.members) for b in self.bursts)
        labels = {"reject": f"คัดออก {n - n_pass:,}", "pass": f"ผ่าน {n_pass:,}",
                  "burst": f"ช็อตรัว {len(self.bursts):,}", "all": f"ทั้งหมด {n:,}"}
        self.seg_tab.configure(values=list(labels.values()))
        self.seg_tab.set(labels[self.tab])
        self._labels = labels
        self.reason_menu.configure(state="normal" if self.tab == "reject" else "disabled")   # ไม่ซ่อน กันแถบเครื่องมือกระโดด
        if self.tab == "burst":                     # แท็บช็อตรัว: ปุ่มเลือกใบดีที่สุดทุกกองแทนเมนูเหตุผล
            self.reason_menu.pack_forget()
            self.burst_btn.pack(side="left", padx=(12, 0))
        else:
            self.burst_btn.pack_forget()
            self.reason_menu.pack(side="left", padx=(12, 0))
        self._update_viewed()

        cam = self.cam_menu.get()
        reason = self.reason_menu.get()
        order = ([i for b in self.bursts for i in b.members] if self.tab == "burst" else range(n))
        items: list[Item] = []
        for i in order:
            p, m, v = self.paths[i], self.metrics[i], self.verdicts[i]
            keep = keeps[i]
            if self.tab == "reject" and keep or self.tab == "pass" and not keep:
                continue
            if cam != ALL_CAMERAS and (m.camera or "(ไม่ระบุ)") != cam:
                continue
            if self.tab == "reject" and reason != ALL_REASONS and reason not in v.reasons and i not in self.overrides:
                continue
            g, b = self.burst_of.get(i, (0, None))
            items.append(Item(i, str(p), p.name, keep, list(v.reasons), i in self.overrides,
                              burst=g, burst_n=len(b.members) if b else 0, star=bool(b and b.best == i)))
        self.gallery.set_items(items, keep_selection=select is None, select_index=select)

    # ----- ตรวจแล้ว / เตือนก่อนส่งออก -----
    def _update_viewed(self) -> None:
        n = len(self.paths)
        self.sb_viewed.configure(text=f"ตรวจแล้ว {len(self.viewed):,} / {n:,} รูป")
        self.sb_viewed_bar.set(len(self.viewed) / max(n, 1))

    def _mark_viewed(self, i: int) -> None:
        if i not in self.viewed:
            self.viewed.add(i)
            self._update_viewed()

    def unviewed_rejects(self) -> list[int]:
        return [i for i in range(len(self.paths)) if not self.keep_of(i) and i not in self.viewed]

    def review_unviewed(self) -> None:
        """ไปตรวจรูปที่คัดออกแต่ยังไม่เคยดู (เรียกจากคำเตือนในหน้าต่างส่งออก)"""
        todo = self.unviewed_rejects()
        self.tab = "reject"
        self.reason_menu.set(ALL_REASONS)
        self.cam_menu.set(ALL_CAMERAS)
        self.refresh(select=0)
        if todo:
            pos = next((k for k, it in enumerate(self.gallery.items) if it.idx == todo[0]), 0)
            self.gallery.select(pos)
        self.gallery.canvas.focus_set()

    def on_tab(self, label: str) -> None:
        self.tab = next(k for k, v in self._labels.items() if v == label)
        self.refresh(select=0)
        self.gallery.canvas.yview_moveto(0)

    def on_filter(self, _=None) -> None:
        self.refresh(select=0)
        self.gallery.canvas.yview_moveto(0)

    def on_size(self, v: float) -> None:
        self.gallery.target_size = theme.px(int(v) // 10 * 10)
        self.gallery._layout()

    # ----- แก้ด้วยมือ -----
    def _set_keep(self, i: int, target: bool) -> tuple[int, bool | None]:
        """ตั้งผลตัดสินรูป i (True = ผ่าน) คืนสถานะ override เดิมไว้เลิกทำ"""
        prev = self.overrides.get(i)
        if target == (not self.verdicts[i].reasons):
            self.overrides.pop(i, None)
        else:
            self.overrides[i] = target
        return i, prev

    def on_action(self, action: str, item: Item) -> None:
        if action == "undo":
            return self.undo()
        if action == "best":
            return self.keep_best(item.idx)
        i = item.idx
        auto = not self.verdicts[i].reasons
        target = {"pass": True, "reject": False, "toggle": not item.keep, "auto": auto}[action]
        self.history.append([self._set_keep(i, target)])
        self._mark_viewed(i)
        pos = self.gallery.sel
        # รายการที่ย้ายแท็บออกไป -> ตำแหน่งเดิมคือรูปถัดไปพอดี; แท็บ "ทั้งหมด"/"ช็อตรัว" ต้องขยับเอง
        self.refresh(select=pos + 1 if self.tab in ("all", "burst") and action != "auto" else pos)
        self.toast(f"{'✓ ผ่าน' if target else '✕ คัดออก'}  {item.name}")
        if self.viewer is not None and self.viewer.winfo_exists():
            self.viewer.show_current()

    def keep_best(self, idx: int) -> None:
        """กองช็อตรัวของรูป idx: เก็บใบที่ดีที่สุดเป็น ผ่าน คัดใบที่เหลือออก"""
        if idx not in self.burst_of:
            return self.toast("รูปนี้ไม่ได้อยู่ในกองช็อตรัว", undo=False)
        _, b = self.burst_of[idx]
        change = [self._set_keep(k, k == b.best) for k in b.members]
        self.history.append(change)
        self.viewed.update(b.members)
        self.refresh(select=self.gallery.sel)
        self.toast(f"เก็บ {self.paths[b.best].name} · คัดออกอีก {len(b.members) - 1} ใบ")

    def keep_best_all(self) -> None:
        if not self.bursts:
            return self.toast("ไม่พบรูปช็อตรัว", undo=False)
        change = [self._set_keep(k, k == b.best) for b in self.bursts for k in b.members]
        self.history.append(change)
        self.viewed.update(k for b in self.bursts for k in b.members)
        self.refresh(select=0)
        self.toast(f"เลือกใบที่ดีที่สุดของ {len(self.bursts)} กองแล้ว (เลิกทำได้)")

    def undo(self) -> None:
        if not self.history:
            return self.toast("ไม่มีอะไรให้เลิกทำ", undo=False)
        for i, prev in reversed(self.history.pop()):
            if prev is None:
                self.overrides.pop(i, None)
            else:
                self.overrides[i] = prev
        self.refresh()
        self.toast("เลิกทำแล้ว", undo=False)
        if self.viewer is not None and self.viewer.winfo_exists():
            self.viewer.show_current()

    def toast(self, text: str, undo: bool = True) -> None:
        self.toast_lbl.configure(text=text)
        self.toast_box.place(relx=0.5, rely=1.0, y=-theme.px(58), anchor="s")
        if self._toast_job is not None:
            self.after_cancel(self._toast_job)
        self._toast_job = self.after(2600, self.toast_box.place_forget)

    def _act_current(self, action: str) -> None:
        if (it := self.gallery.current()) is not None:
            self.on_action(action, it)

    # =====================================================================
    # พรีวิวด้านขวา
    # =====================================================================
    def on_select(self, item: Item | None) -> None:
        c = self.pv_canvas
        c.delete("all")
        for w in self.pv_status.winfo_children():
            w.destroy()
        self.pv_reset.pack_forget()
        self.pv_burst.pack_forget()
        if self._view_job is not None:
            self.after_cancel(self._view_job)
            self._view_job = None
        if item is not None:                                  # ดูค้างครึ่งวินาที = นับว่าตรวจแล้ว (ไล่เร็วๆ ไม่นับ)
            self._view_job = self.after(500, lambda i=item.idx: self._mark_viewed(i))
            if item.burst_n:
                where = "★ ใบที่ดีที่สุดของกอง" if item.star else "โปรแกรมแนะนำใบอื่น (มี ★)"
                self.pv_burst_lbl.configure(text=f"ช็อตรัว กลุ่ม {item.burst}  ·  {item.burst_n} ใบ\n{where}")
                self.pv_burst.pack(fill="x", padx=16, pady=(12, 0), after=self.pv_status)
        if item is None:
            self.pv_name.configure(text="ไม่มีรูปในหมวดนี้")
            self.pv_sub.configure(text="")
            self.pv_prob.configure(text="–")
            self.pv_bar.set(0)
            self.pv_sharp.configure(text="–")
            self.pv_blink.configure(text="–")
            return
        i = item.idx
        m, v = self.metrics[i], self.verdicts[i]
        self.pv_name.configure(text=item.name)
        self.pv_sub.configure(text=(m.camera + " · " if m.camera else "") + f"{len(m.faces)} หน้า")
        self._draw_preview(c, item, m)

        # สถานะ + เหตุผล
        pill = ctk.CTkLabel(self.pv_status, text="  ผ่าน  " if item.keep else "  คัดออก  ", height=28, corner_radius=14,
                            font=theme.font(12, "bold"), text_color="#062b1c" if item.keep else "#2a0707",
                            fg_color=theme.GOOD if item.keep else theme.BAD)
        pill.pack(side="left")
        for r in v.reasons:
            ctk.CTkLabel(self.pv_status, text=r, height=24, corner_radius=12, font=theme.font(11, "bold"),
                         text_color="#111", fg_color=theme.REASON_COLORS.get(r, theme.MUTED)
                         ).pack(side="left", padx=(6, 0))
        if item.manual:
            ctk.CTkLabel(self.pv_status, text="แก้ด้วยมือ", font=theme.font(11, "bold"), text_color=theme.ACCENT_HI
                         ).pack(side="left", padx=(10, 0))
            self.pv_reset.pack(fill="x", padx=16, pady=(8, 0))

        main = usable_main(m)
        if v.prob is not None and main is not None:
            self.pv_prob.configure(text=f"{v.prob * 100:.0f}%")
            self.pv_bar.set(v.prob)
            self.pv_bar.configure(progress_color=theme.BAD if v.prob >= PRESETS[self.preset_name].reject_p else theme.GOOD)
            r = v.ratio or 1.0
            self.pv_sharp.configure(text=("คมปกติ" if r >= 0.6 else "ซอฟต์" if r >= 0.25 else "เบลอ") + f"  ({r:.2f}×)")
            b = main.blink
            self.pv_blink.configure(text=("ลืมตา" if b < 0.3 else "ตากึ่งปิด" if b < 0.55 else "หลับตา") + f"  ({b * 100:.0f}%)")
        else:
            self.pv_prob.configure(text="–")
            self.pv_bar.set(0)
            self.pv_sharp.configure(text="ตัวเอกหันหลัง ไม่ตัดสินจากหน้าคนข้างหลัง" if v.away else "ไม่มีคนเด่นให้ประเมิน")
            self.pv_blink.configure(text="–")

    def _draw_preview(self, c: tk.Canvas, item: Item, m: Metrics) -> None:
        cw, ch = int(c["width"]), int(c["height"])
        try:
            im = load_image(item.path, (cw, ch))
        except Exception:  # noqa: BLE001
            c.create_text(cw // 2, ch // 2, text="เปิดรูปไม่ได้", fill=theme.MUTED, font=theme.font(12))
            return
        self._pv_img = ImageTk.PhotoImage(im)
        ox, oy = (cw - im.width) // 2, (ch - im.height) // 2
        c.create_image(ox, oy, image=self._pv_img, anchor="nw")
        draw_face_boxes(c, m, ox, oy, im.width, im.height)

    # =====================================================================
    # ภาพใหญ่ / ส่งออก
    # =====================================================================
    def open_viewer(self, item: Item) -> None:
        if self.viewer is not None and self.viewer.winfo_exists():
            self.viewer.focus()
            self.viewer.show_current()
            return
        self.viewer = Viewer(self)

    def open_export(self) -> None:
        if not self.paths:
            return
        ExportDialog(self)

    def _export_progress(self, d: int, t: int) -> None:
        if getattr(self, "export_dlg", None) and self.export_dlg.winfo_exists():
            self.export_dlg.progress(d, t)

    def _export_done(self, s: Summary) -> None:
        if getattr(self, "export_dlg", None) and self.export_dlg.winfo_exists():
            self.export_dlg.done(s)

    def _export_error(self, msg: str) -> None:
        if getattr(self, "export_dlg", None) and self.export_dlg.winfo_exists():
            self.export_dlg.fail(msg)


def draw_face_boxes(c: tk.Canvas, m: Metrics, ox: int, oy: int, w: int, h: int) -> None:
    main = usable_main(m)
    for f in m.faces:
        if not f.has_expr:
            continue
        x, y, bw, bh = f.box
        is_main = f is main
        c.create_rectangle(ox + x * w, oy + y * h, ox + (x + bw) * w, oy + (y + bh) * h,
                           outline=theme.GOOD if is_main else "#d0d0d8", width=theme.px(2) if is_main else 1)


class Viewer(ctk.CTkToplevel):
    """หน้าต่างดูภาพใหญ่ (จุดที่ใช้ตัดสินจริง): มีตัวนับ ป้ายสถานะใหญ่ กรอบสีตามสถานะ ปุ่มกดได้
    ← → เลื่อน · P ผ่าน · X คัดออก · Z ซูม 100% (ลากเพื่อเลื่อน เริ่มที่หน้าคนเด่น) · F กรอบหน้า · Esc ปิด"""

    def __init__(self, app: App) -> None:
        super().__init__(app, fg_color="#0b0b0c")
        self.app = app
        self.title("ดูภาพ")
        self.geometry("1180x760+40+30")
        self.minsize(700, 500)
        self.after(60, lambda: self.state("zoomed"))      # เต็มจอ (ยังเห็นแถบงาน) — เหมาะกับการไล่ดูรายละเอียดหน้า
        self.show_boxes = True
        self.zoom = False
        self._full = None            # (path, ภาพเต็ม) แคชไว้ซูม
        self._center = None          # จุดกึ่งกลางที่ซูม (พิกัดภาพเต็ม)
        self._drag = None
        self._img = None
        self.canvas = tk.Canvas(self, bg="#0b0b0c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        bar = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=0, height=56)
        bar.pack(fill="x", side="bottom")
        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(pady=8)
        btn(inner, "◀", lambda: self.move(-1), "ghost", width=44, height=38).pack(side="left", padx=4)
        btn(inner, "✕  คัดออก  (X)", lambda: self.act("reject"), "bad", width=150, height=38).pack(side="left", padx=4)
        btn(inner, "✓  ผ่าน  (P)", lambda: self.act("pass"), "good", width=150, height=38).pack(side="left", padx=4)
        btn(inner, "▶", lambda: self.move(1), "ghost", width=44, height=38).pack(side="left", padx=4)
        btn(inner, "ซูม 100%  (Z)", self.toggle_zoom, "ghost", width=130, height=38).pack(side="left", padx=(24, 4))
        btn(inner, "กรอบหน้า  (F)", self.toggle_boxes, "ghost", width=130, height=38).pack(side="left", padx=4)

        for key, fn in (("<Escape>", lambda e: self.destroy()), ("<Left>", lambda e: self.move(-1)),
                        ("<Right>", lambda e: self.move(1)), ("p", lambda e: self.act("pass")),
                        ("x", lambda e: self.act("reject")), ("f", lambda e: self.toggle_boxes()),
                        ("z", lambda e: self.toggle_zoom()), ("<space>", lambda e: self.act("toggle")),
                        ("<Control-z>", lambda e: self.app.undo())):
            self.bind(key, fn)
        self.canvas.bind("<Configure>", lambda e: self.show_current())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<Double-Button-1>", lambda e: self.toggle_zoom())
        self.after(100, self.focus_force)
        self.show_current()

    def move(self, d: int) -> None:
        g = self.app.gallery
        g.select(g.sel + d)
        self._center = None
        self.show_current()

    def act(self, action: str) -> None:
        self.app._act_current(action)

    def toggle_boxes(self) -> None:
        self.show_boxes = not self.show_boxes
        self.show_current()

    def toggle_zoom(self) -> None:
        self.zoom = not self.zoom
        self._center = None
        self.show_current()

    def _press(self, e) -> None:
        self._drag = (e.x, e.y)

    def _motion(self, e) -> None:
        if self.zoom and self._drag and self._center:
            self._center = (self._center[0] - (e.x - self._drag[0]), self._center[1] - (e.y - self._drag[1]))
            self._drag = (e.x, e.y)
            self.show_current()

    def _chip(self, x: int, y: int, text: str, fill: str, fg: str = "#ffffff", size: int = 15, right: bool = False) -> None:
        c = self.canvas
        t = c.create_text(0, 0, text=text, anchor="w", fill=fg, font=theme.font(size, "bold"))
        x0, y0, x1, y1 = c.bbox(t)
        pad = theme.px(12)
        if right:                                  # x = ขอบขวาของชิป
            x -= (x1 - x0) + pad
        theme.rounded_rect(c, x - pad, y - theme.px(8) - (y1 - y0) // 2, x + (x1 - x0) + pad, y + theme.px(8) + (y1 - y0) // 2,
                           theme.px(10), fill=fill, outline="")
        c.coords(t, x, y)
        c.tag_raise(t)

    def show_current(self) -> None:
        it = self.app.gallery.current()
        c = self.canvas
        c.delete("all")
        if it is None:
            return
        cw, ch = max(c.winfo_width(), 200), max(c.winfo_height(), 200)
        m = self.app.metrics[it.idx]
        v = self.app.verdicts[it.idx]
        main = usable_main(m)
        try:
            if self.zoom:
                if self._full is None or self._full[0] != it.path:
                    self._full = (it.path, load_full(it.path))
                full = self._full[1]
                if self._center is None:
                    x, y, bw, bh = main.box if main else (0.5, 0.5, 0, 0)
                    self._center = ((x + bw / 2) * full.width, (y + bh / 2) * full.height)
                cx = min(max(self._center[0], cw / 2), max(full.width - cw / 2, cw / 2))
                cy = min(max(self._center[1], ch / 2), max(full.height - ch / 2, ch / 2))
                self._center = (cx, cy)
                left, top = int(cx - cw / 2), int(cy - ch / 2)
                im = full.crop((left, top, left + cw, top + ch))
                ox = oy = 0
                self._img = ImageTk.PhotoImage(im)
                c.create_image(0, 0, image=self._img, anchor="nw")
                if self.show_boxes:
                    for f in m.faces:
                        if f.has_expr:
                            bx, by, bw, bh = f.box
                            c.create_rectangle((bx * full.width - left), (by * full.height - top),
                                               ((bx + bw) * full.width - left), ((by + bh) * full.height - top),
                                               outline=theme.GOOD if f is main else "#ffffff", width=2 if f is main else 1)
            else:
                im = load_image(it.path, (cw, ch))
                self._img = ImageTk.PhotoImage(im)
                ox, oy = (cw - im.width) // 2, (ch - im.height) // 2
                c.create_image(ox, oy, image=self._img, anchor="nw")
                if self.show_boxes:
                    draw_face_boxes(c, m, ox, oy, im.width, im.height)
        except Exception:  # noqa: BLE001
            return

        # กรอบสีตามสถานะรอบภาพ + HUD
        col = theme.GOOD if it.keep else theme.BAD
        c.create_rectangle(2, 2, cw - 2, ch - 2, outline=col, width=theme.px(3))
        n = len(self.app.gallery.items)
        self._chip(theme.px(24), theme.px(30), f"{self.app.gallery.sel + 1} / {n}   ·   {it.name}", "#000000", size=14)
        self._chip(cw // 2 - theme.px(50), theme.px(30), "✓  ผ่าน" if it.keep else "✕  คัดออก", col, fg="#04140d" if it.keep else "#2a0707", size=16)
        note = []
        if it.reasons:
            note.append("โปรแกรมให้คัดออก: " + " + ".join(v.reasons))
        if it.manual:
            note.append("คุณแก้เอง")
        if v.ratio is not None and main is not None:
            note.append(f"ความคมหน้า {v.ratio:.2f}×  ·  โอกาสรูปหลุด {v.prob * 100:.0f}%")
        if note:
            self._chip(theme.px(24), ch - theme.px(28), "    ".join(note), "#000000", size=13)
        if self.show_boxes and main is not None:
            self._chip(cw - theme.px(24), theme.px(30), "กรอบเขียว = คนเด่นที่ใช้ตัดสิน", "#000000", fg=theme.GOOD, size=13, right=True)


class ExportDialog(ctk.CTkToplevel):
    def __init__(self, app: App) -> None:
        super().__init__(app, fg_color=theme.PANEL)
        self.app = app
        app.export_dlg = self
        self.title("ส่งออกรูป")
        self.geometry("640x440")
        self.resizable(False, False)
        self.transient(app)
        self.after(120, self.grab_set)

        n = len(app.paths)
        keeps = [app.keep_of(i) for i in range(n)]
        self.n_pass = sum(keeps)
        self.parent_dir = app.root_dir.parent
        self.out_dir = make_out_dir(app.root_dir, self.parent_dir)

        ctk.CTkLabel(self, text="ส่งออกรูป", font=theme.font(22, "bold"), text_color=theme.TEXT).pack(pady=(24, 2))
        ctk.CTkLabel(self, text="คัดลอกรูปไปเป็น 2 โฟลเดอร์ ต้นฉบับไม่ถูกแตะ", font=theme.font(12),
                     text_color=theme.MUTED).pack()

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=18)
        for text, num, col in (("ผ่าน", self.n_pass, theme.GOOD), ("คัดออก", n - self.n_pass, theme.BAD)):
            card = ctk.CTkFrame(row, fg_color=theme.PANEL_HI, corner_radius=12, width=170, height=88)
            card.pack(side="left", padx=8)
            card.pack_propagate(False)
            ctk.CTkLabel(card, text=f"{num:,}", font=theme.font(28, "bold"), text_color=col).pack(pady=(10, 0))
            ctk.CTkLabel(card, text=text, font=theme.font(12), text_color=theme.MUTED).pack()

        ctk.CTkLabel(self, text="บันทึกไว้ที่", font=theme.font(12, "bold"), text_color=theme.MUTED, anchor="w"
                     ).pack(fill="x", padx=36)
        prow = ctk.CTkFrame(self, fg_color=theme.PANEL_HI, corner_radius=8)
        prow.pack(fill="x", padx=32, pady=(4, 0))
        txt = ctk.CTkFrame(prow, fg_color="transparent")
        txt.pack(side="left", padx=12, pady=8, fill="x", expand=True)
        self.name_lbl = ctk.CTkLabel(txt, text=self.out_dir.name, font=theme.font(14, "bold"), text_color=theme.TEXT, anchor="w")
        self.name_lbl.pack(fill="x")
        self.path_lbl = ctk.CTkLabel(txt, text=theme.shorten_path(str(self.out_dir.parent), 2, 46), font=theme.font(12),
                                     text_color=theme.MUTED, anchor="w")
        self.path_lbl.pack(fill="x")
        btn(prow, "เปลี่ยน…", self.change_dir, "ghost", width=80, height=28).pack(side="right", padx=8)
        ctk.CTkLabel(self, text=f"จะสร้างโฟลเดอร์ย่อย  ●  ผ่าน ({self.n_pass:,} รูป)   ●  คัดออก ({n - self.n_pass:,} รูป)   + รายงาน.csv",
                     font=theme.font(12), text_color=theme.MUTED).pack(pady=(10, 0))

        todo = app.unviewed_rejects()
        if todo:      # เตือน: ยังไม่ได้ดูรูปที่โปรแกรมคัดออก อาจมีรูปดีปนไป
            self.geometry("640x520")
            warn = ctk.CTkFrame(self, fg_color="#2d2410", corner_radius=10, border_width=1, border_color="#6b5316")
            warn.pack(fill="x", padx=32, pady=(14, 0))
            ctk.CTkLabel(warn, text=f"⚠  ยังไม่ได้ดูรูปที่คัดออก {len(todo):,} รูป\nอาจมีรูปดีถูกคัดออกโดยไม่รู้ตัว — ควรไล่ดูก่อน",
                         font=theme.font(12), text_color=theme.WARN, justify="left", anchor="w"
                         ).pack(side="left", padx=14, pady=10)
            btn(warn, "ไปตรวจต่อ", lambda: (self.destroy(), app.review_unviewed()), "ghost", width=100, height=32
                ).pack(side="right", padx=10)

        self.bar = ctk.CTkProgressBar(self, height=8, corner_radius=4, progress_color=theme.PANEL_HI, fg_color=theme.PANEL_HI)
        self.bar.set(0)
        self.status = ctk.CTkLabel(self, text="", font=theme.font(13), text_color=theme.TEXT)
        self.buttons = ctk.CTkFrame(self, fg_color="transparent")
        self.buttons.pack(side="bottom", pady=22)
        self.go_btn = btn(self.buttons, "เริ่มส่งออก", self.start, "primary", width=170, height=42, font=theme.font(14, "bold"))
        self.go_btn.pack(side="left", padx=6)
        self.close_btn = btn(self.buttons, "ยกเลิก", self.destroy, "ghost", width=110, height=42)
        self.close_btn.pack(side="left", padx=6)

    def change_dir(self) -> None:
        d = filedialog.askdirectory(title="เลือกที่เก็บผลลัพธ์", parent=self)
        if d:
            self.parent_dir = Path(os.path.normpath(d))
            self.out_dir = make_out_dir(self.app.root_dir, self.parent_dir)
            self.name_lbl.configure(text=self.out_dir.name)
            self.path_lbl.configure(text=theme.shorten_path(str(self.out_dir.parent), 2, 46))

    def start(self) -> None:
        self.go_btn.configure(state="disabled")
        self.close_btn.configure(state="disabled")
        self.bar.pack(fill="x", padx=36, pady=(14, 0))       # แสดงแถบเมื่อเริ่มจริง
        self.status.pack(pady=(8, 0))
        self.status.configure(text="กำลังคัดลอกไฟล์…")
        app = self.app
        keeps = [app.keep_of(i) for i in range(len(app.paths))]
        args = (app.root_dir, app.paths, app.metrics, app.verdicts, keeps, self.out_dir, set(app.overrides))

        viewed = set(app.viewed)
        preset, main_only = PRESETS[app.preset_name], app.main_only.get()

        def work() -> None:
            try:
                s = export(*args, progress=lambda st, d, t: app.q.put(("export_progress", d, t)), viewed=viewed,
                           preset=preset, main_only=main_only)
                app.q.put(("export_done", s))
            except Exception as e:  # noqa: BLE001
                app.q.put(("export_error", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def progress(self, d: int, t: int) -> None:
        self.bar.configure(progress_color=theme.ACCENT if d else theme.PANEL_HI)
        self.bar.set(d / max(t, 1))
        self.status.configure(text=f"กำลังคัดลอก {d:,} / {t:,}")

    def done(self, s: Summary) -> None:
        self.bar.configure(progress_color=theme.GOOD)
        self.bar.set(1)
        self.status.configure(text=f"เสร็จแล้ว!  ผ่าน {s.passed:,} · คัดออก {s.rejected:,}", text_color=theme.GOOD)
        self.go_btn.configure(text="เปิดโฟลเดอร์", state="normal", command=lambda: os.startfile(s.out_dir))
        self.close_btn.configure(text="ปิด", state="normal")

    def fail(self, msg: str) -> None:
        self.status.configure(text=f"ผิดพลาด: {msg}", text_color=theme.BAD)
        self.go_btn.configure(state="normal")
        self.close_btn.configure(state="normal")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # จำเป็นตอนแพ็กเป็น .exe บน Windows
    if len(sys.argv) > 1:  # โหมดไม่มีหน้าต่าง (ไว้ทดสอบ): คัดรูปแล้วออก  ใช้: app <โฟลเดอร์> [ระดับ] [all]
        # ท้ายคำสั่งใส่ all = ตรวจทุกคนในภาพ (ปกติตรวจเฉพาะคนเด่นที่สุด)
        print(run(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PRESET,
                  main_only=sys.argv[3:4] != ["all"]))
    else:
        App().mainloop()
