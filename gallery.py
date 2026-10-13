"""กริดรูปย่อแบบเลื่อนลื่น (วาดเฉพาะช่องที่เห็น รองรับหลักพันรูป)
รูปที่ผ่านแสดงเงียบๆ; รูปที่คัดออกมืดลง + กรอบแดง + ไอคอน ✕ + ชิปเหตุผล ให้แยกได้ในพริบตา"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable

import customtkinter as ctk

import icons
import theme
from thumbs import THUMB_ASPECT, ThumbCache

GAP = 4


@dataclass
class Item:
    idx: int                         # ดัชนีในรายการรูปทั้งหมดของแอป
    path: str
    name: str
    keep: bool                       # สถานะสุดท้าย (รวมที่ผู้ใช้แก้เอง)
    reasons: list[str] = field(default_factory=list)   # เหตุผลที่โปรแกรมให้คัดออก
    manual: bool = False             # ผู้ใช้กลับคำตัดสินของโปรแกรม
    burst: int = 0                   # กองช็อตรัวลำดับที่ (0 = ไม่อยู่ในกอง)
    burst_n: int = 0                 # จำนวนรูปในกอง
    star: bool = False               # เป็นใบที่ดีที่สุดของกองที่โปรแกรมแนะนำ


class Gallery(tk.Frame):
    def __init__(self, master, thumbs: ThumbCache, on_select: Callable[[Item | None], None],
                 on_open: Callable[[Item], None], on_action: Callable[[str, Item], None]):
        super().__init__(master, bg=theme.BG)
        self.thumbs = thumbs
        self.on_select, self.on_open, self.on_action = on_select, on_open, on_action
        self.items: list[Item] = []
        self.sel = -1
        self.hover = -1
        self.cols = 1
        self.cell_w = self.cell_h = 0
        self.target_size = thumbs.size          # ขนาดที่ผู้ใช้เลือก (พิกเซลจริง) — ขนาดที่วาดจริงปรับให้เต็มช่อง
        self._redraw_job = None
        self.gap = theme.px(GAP)

        self.canvas = tk.Canvas(self, bg=theme.BG, highlightthickness=0, takefocus=True,
                                yscrollincrement=theme.px(48))
        self.bar = ctk.CTkScrollbar(self, command=self.canvas.yview, fg_color=theme.BG, width=10,
                                    button_color="#3a3a40", button_hover_color="#5a5a62")
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._bar_shown = False

        c = self.canvas
        c.bind("<Configure>", lambda e: self._layout())
        c.bind("<Button-1>", self._click)
        c.bind("<Double-Button-1>", self._double)
        c.bind("<Motion>", lambda e: self._set_hover(self._index_at(e)))
        c.bind("<Leave>", lambda e: self._set_hover(-1))
        c.bind("<MouseWheel>", lambda e: c.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        for key in ("<Left>", "<Right>", "<Up>", "<Down>", "<Home>", "<End>", "<Prior>", "<Next>"):
            c.bind(key, self._nav)
        for key, act in (("p", "pass"), ("P", "pass"), ("x", "reject"), ("X", "reject"),
                         ("b", "best"), ("B", "best"),
                         ("<space>", "toggle"), ("<Return>", "open"), ("<Control-z>", "undo"), ("<Control-Z>", "undo")):
            c.bind(key, lambda e, a=act: self._act(a))

    # ----- ข้อมูล -----
    def set_items(self, items: list[Item], keep_selection: bool = True, select_index: int | None = None) -> None:
        prev = self.items[self.sel].idx if keep_selection and 0 <= self.sel < len(self.items) else None
        self.items = items
        if select_index is not None:
            self.sel = min(max(select_index, 0), len(items) - 1) if items else -1
        elif prev is not None:
            self.sel = next((i for i, it in enumerate(items) if it.idx == prev), min(0, len(items) - 1))
        else:
            self.sel = 0 if items else -1
        self._layout()
        self.on_select(self.current())

    def current(self) -> Item | None:
        return self.items[self.sel] if 0 <= self.sel < len(self.items) else None

    def refresh(self) -> None:
        self._schedule_redraw()

    # ----- เลย์เอาต์ -----
    def _layout(self) -> None:
        w = max(self.canvas.winfo_width(), 200)
        g = self.gap = theme.px(GAP)
        self.cols = max(1, w // (self.target_size + 2 * g))
        self.cell_w = w // self.cols
        size = max(theme.px(80), (self.cell_w - 2 * g) // 10 * 10)      # ปัดเป็นทีละ 10px ให้ใช้แคชซ้ำได้
        self.thumbs.set_size(size)
        self.cell_h = int((self.cell_w - 2 * g) * THUMB_ASPECT) + 2 * g
        rows = -(-len(self.items) // self.cols)
        self.canvas.configure(scrollregion=(0, 0, w, max(rows * self.cell_h + theme.px(8), 1)))
        self._schedule_redraw()

    def _on_scroll(self, a, b) -> None:
        self.bar.set(a, b)
        need = not (float(a) <= 0.0 and float(b) >= 1.0)      # เนื้อหาไม่เกินจอ = ซ่อนแถบเลื่อน
        if need != self._bar_shown:
            self._bar_shown = need
            self.bar.pack(side="right", fill="y", padx=(0, 2), pady=2) if need else self.bar.pack_forget()
        self._schedule_redraw()

    def _schedule_redraw(self) -> None:
        if self._redraw_job is None:
            self._redraw_job = self.after(12, self._redraw)

    # ----- วาด -----
    def _cell_xy(self, i: int) -> tuple[int, int]:
        return (i % self.cols) * self.cell_w, (i // self.cols) * self.cell_h

    def _redraw(self) -> None:
        self._redraw_job = None
        c = self.canvas
        c.delete("all")
        if not self.items:
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2 - theme.px(10), text="ไม่มีรูปในหมวดนี้",
                          fill=theme.MUTED, font=theme.font(14))
            return
        top = int(c.canvasy(0))
        bottom = top + c.winfo_height()
        first = max(0, top // self.cell_h) * self.cols
        last = min(len(self.items), (bottom // self.cell_h + 1) * self.cols)
        for i in range(first, last):
            self._draw_cell(i)

    def _chip(self, x: int, y: int, text: str, dot: str | None, anchor: str = "nw") -> tuple[int, int]:
        """ชิปโปร่งแสงเล็กบนรูป (จุดสี + ข้อความ) คืนพิกัดขวาล่าง"""
        c, px = self.canvas, theme.px
        h, pad = px(22), px(9)
        t = c.create_text(0, 0, text=text, anchor="w", fill="#f2f2f5", font=theme.font(12, "bold"))
        tw = c.bbox(t)[2] - c.bbox(t)[0]
        dot_w = px(12) if dot else 0
        w = tw + 2 * pad + dot_w
        x0 = x if anchor == "nw" else x - w
        bg = c.create_image(x0, y, image=icons.pill(w, h), anchor="nw")
        if dot:
            r = px(4)
            c.create_oval(x0 + pad, y + h // 2 - r, x0 + pad + 2 * r, y + h // 2 + r, fill=dot, outline="")
        c.coords(t, x0 + pad + dot_w, y + h // 2)
        c.tag_raise(t)
        return x0 + w, y + h

    def _draw_cell(self, i: int) -> None:
        c, it, px, g = self.canvas, self.items[i], theme.px, self.gap
        x, y = self._cell_xy(i)
        x0, y0, x1, y1 = x + g, y + g, x + self.cell_w - g, y + self.cell_h - g
        selected, hover = i == self.sel, i == self.hover
        rej = not it.keep

        theme.rounded_rect(c, x0, y0, x1, y1, px(6), fill=theme.CELL_HOVER if hover else theme.CELL, outline="")
        img = self.thumbs.get(it.path, dim=rej)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if img is not None:
            c.create_image(cx, cy, image=img)

        # กรอบ: เลือก = น้ำเงิน, คัดออก = แดง, ชี้ = เทาอ่อน
        if selected:
            theme.rounded_rect(c, x0, y0, x1, y1, px(6), fill="", outline=theme.ACCENT, width=px(2))
        elif rej:
            theme.rounded_rect(c, x0 + 1, y0 + 1, x1 - 1, y1 - 1, px(6), fill="", outline=theme.BAD, width=px(2))
        elif hover:
            theme.rounded_rect(c, x0, y0, x1, y1, px(6), fill="", outline=theme.LINE_HI, width=1)

        inset = px(8)
        # ชิปมุมซ้ายบนในรูป: ★ ใบที่ดีที่สุดของกอง แล้วตามด้วยเหตุผลที่โปรแกรมให้คัดออก
        bx = x0 + inset
        if it.star:
            bx, _ = self._chip(bx, y0 + inset, "★ ดีที่สุด", "#ffd166")
            bx += px(5)
        if it.reasons and rej:
            for r in it.reasons:
                bx, _ = self._chip(bx, y0 + inset, r, theme.REASON_COLORS.get(r, theme.MUTED))
                bx += px(5)
        # ไอคอนสถานะ: ✕ สำหรับรูปที่คัดออก; รูปผ่านเงียบ ยกเว้นที่ผู้ใช้กลับคำตัดสินให้ผ่าน
        if rej or it.manual:
            s = px(24)
            c.create_image(x1 - inset, y0 + inset, image=icons.badge("x" if rej else "check", s), anchor="ne")
        # ล่างซ้าย: ชื่อไฟล์ (เมื่อชี้/เลือก) · ล่างขวา: ป้ายแก้เอง
        bx = x0 + inset
        if hover or selected:
            bx, _ = self._chip(bx, y1 - inset - px(22), it.name if len(it.name) <= 24 else it.name[:11] + "…" + it.name[-11:], None)
            bx += px(5)
        if it.burst_n:
            self._chip(bx, y1 - inset - px(22), f"ช็อตรัว ×{it.burst_n}" if not (hover or selected) else f"กลุ่ม {it.burst} · ×{it.burst_n}",
                       "#ffd166")
        if it.manual:
            self._chip(x1 - inset, y1 - inset - px(22), "แก้เอง", theme.ACCENT_HI, anchor="ne")

    # ----- อีเวนต์ -----
    def _index_at(self, e) -> int:
        x, y = e.x, self.canvas.canvasy(e.y)
        col, row = int(x // self.cell_w), int(y // self.cell_h)
        i = row * self.cols + col
        return i if 0 <= col < self.cols and 0 <= i < len(self.items) else -1

    def _click(self, e) -> None:
        self.canvas.focus_set()
        i = self._index_at(e)
        if i >= 0:
            self.select(i)

    def _double(self, e) -> None:
        i = self._index_at(e)
        if i >= 0:
            self.on_open(self.items[i])

    def _set_hover(self, i: int) -> None:
        if i != self.hover:
            self.hover = i
            self._schedule_redraw()

    def select(self, i: int, scroll: bool = True) -> None:
        if not self.items:
            return
        self.sel = min(max(i, 0), len(self.items) - 1)
        if scroll:
            self._ensure_visible(self.sel)
        self._schedule_redraw()
        self.on_select(self.current())

    def _ensure_visible(self, i: int) -> None:
        c = self.canvas
        _, y = self._cell_xy(i)
        top, h = c.canvasy(0), c.winfo_height()
        total = max(1, -(-len(self.items) // self.cols) * self.cell_h + theme.px(8))
        if y < top:
            c.yview_moveto(y / total)
        elif y + self.cell_h > top + h:
            c.yview_moveto((y + self.cell_h - h + theme.px(8)) / total)

    def _nav(self, e) -> str:
        step = {"Left": -1, "Right": 1, "Up": -self.cols, "Down": self.cols,
                "Prior": -self.cols * 3, "Next": self.cols * 3}.get(e.keysym)
        if e.keysym == "Home":
            self.select(0)
        elif e.keysym == "End":
            self.select(len(self.items) - 1)
        elif step is not None and self.items:
            self.select(self.sel + step)
        return "break"

    def _act(self, action: str) -> None:
        if action == "undo":
            self.on_action("undo", self.current() or Item(-1, "", "", True))
        elif (it := self.current()) is not None:
            self.on_open(it) if action == "open" else self.on_action(action, it)
