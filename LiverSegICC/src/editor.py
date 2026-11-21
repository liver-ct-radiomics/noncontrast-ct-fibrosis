"""Reader correction window (AUTH1 / AUTH2).

The reader starts from a copy of the TS mask and sees only TS and their OWN mask.
Tools: Brush (adds) and Eraser (removes), with a diameter in mm, placed in the strip beside the
image. They act on the visible slice only unless "3D sphere" is on, which also applies a sphere of
that diameter to neighbouring slices.

Mouse:  left = selected tool | right = the opposite (erases while on Brush, adds while on Eraser)
        middle / Space+left = pan | wheel = slice | Ctrl+wheel = zoom | Shift+wheel = diameter
Keys:   B brush, E eraser, H pan, + - or [ ] diameter, Up/Down slice, PgUp/PgDn 10 slices,
        M show mask, C contour, 1/2/3 window, 0 reset view, Ctrl+Z undo, Ctrl+Y redo, Ctrl+S save
"""

import math
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageTk
from scipy import ndimage

import config as C
import sitk_io
from cases import case_dir, read_json, write_json
from uiutil import enable_dpi_awareness, px

FILL_RGB = np.array([255, 140, 20], np.float32)
CONTOUR_RGB = np.array([255, 235, 0], np.uint8)
WL_PRESETS = [("Liver", 150, 50), ("Abdomen", 400, 40), ("Narrow", 100, 55)]
IDLE_LIMIT_S = 120          # longer inactivity does not count as active time
AUTOSAVE_S = 120
UNDO_LIMIT = 40
DIAM_MIN, DIAM_MAX = 1, 80

STRIP_BG = "#2b2d30"
STRIP_FG = "#f0f0f0"
BTN_BG = "#45484d"
BRUSH_ON = "#1e8e3e"
ERASER_ON = "#c62828"
PAN_ON = "#1565c0"


class CorrectionEditor(tk.Toplevel):
    def __init__(self, master, case_id, reader, on_closed=None):
        super().__init__(master)
        if reader not in C.READERS:
            raise ValueError(reader)
        self.case_id, self.reader, self.on_closed = case_id, reader, on_closed
        self.dir = case_dir(case_id)
        self.mask_path = self.dir / C.mask_file(reader)
        self.log_path = self.dir / C.reader_log_file(reader)

        # data
        self.ct_img = sitk_io.read_image(self.dir / C.CT_FILE)
        self.ct = sitk.GetArrayFromImage(self.ct_img)
        self.sx, self.sy, self.sz = (float(v) for v in self.ct_img.GetSpacing())
        self.Z, self.H, self.W = self.ct.shape
        self.ts = sitk.GetArrayFromImage(sitk_io.read_image(self.dir / C.mask_file("TS"))) > 0
        if self.mask_path.exists():
            self.mask = (sitk.GetArrayFromImage(sitk_io.read_image(self.mask_path)) > 0).astype(np.uint8)
        else:
            self.mask = self.ts.astype(np.uint8)
        self.voxel_mm3 = self.sx * self.sy * self.sz

        now = datetime.now().isoformat(timespec="seconds")
        self.log = read_json(self.log_path, None) or {
            "reader": reader, "case_id": case_id, "status": C.IN_PROGRESS, "active_seconds": 0.0,
            "strokes": 0, "sessions": [], "started_at": now, "completed_at": None,
            "reopened_after_complete": 0,
        }
        self.was_complete = self.log.get("status") == C.DONE
        self.session = {"start": now, "end": now, "active_seconds": 0.0, "strokes": 0}
        self.active = float(self.log.get("active_seconds", 0.0))
        self.strokes = int(self.log.get("strokes", 0))
        self._recount()

        # state
        self.z = int(np.argwhere(self.mask.any(axis=(1, 2))).mean()) if self.mask.any() else self.Z // 2
        self.ww, self.wl = WL_PRESETS[0][1], WL_PRESETS[0][2]
        self.zoom, self.cx, self.cy = 1.0, self.W / 2, self.H / 2
        self.scale_x = self.scale_y = self.px_per_mm = 1.0
        self.side_y, self.side_x = self.H // 2, self.W // 2
        self._gray_key = None
        self._gray = None
        self._photo = self._side_photo = None
        self._painting = None
        self._pan_last = None
        self._space = False
        self._stroke_snap = {}
        self.undo_stack = []
        self.redo_stack = []
        self._dirty = False
        self._render_pending = self._side_pending = False
        self._last_input = time.time()
        self._last_tick = time.time()
        self._last_autosave = time.time()
        self._save_lock = threading.Lock()
        self._save_error = None
        self._mouse = None
        self._side_map = None

        self._build_ui()
        self.title(f"{reader}  -  {case_id}  -  Liver mask correction")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._fit_initial)
        self.after(1000, self._tick)
        if self.was_complete:
            self.after(300, lambda: messagebox.showinfo(
                "Case already done",
                "This case was already marked as done.\n"
                "If you change the mask it returns to 'in progress' and has to be marked as done again.",
                parent=self))

    # interface
    def _build_ui(self):
        P = lambda n: px(self, n)
        sw_, sh_ = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{int(sw_ * 0.9)}x{int(sh_ * 0.85)}+{int(sw_ * 0.05)}+{int(sh_ * 0.04)}")
        self.minsize(P(900), P(600))
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.status = ttk.Label(self, anchor="w", padding=(6, 3), relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.tool_var = tk.StringVar(value="brush")
        self.diam_var = tk.DoubleVar(value=10.0)
        self.sphere_var = tk.BooleanVar(value=False)
        self._build_tool_strip(P)

        # Display settings stay in a single row above the image
        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Window").pack(side=tk.LEFT)
        for i, (name, w, l) in enumerate(WL_PRESETS):
            ttk.Button(bar, text=f"{i + 1}: {name}", width=11,
                       command=lambda w=w, l=l: self._set_wl(w, l)).pack(side=tk.LEFT, padx=1)
        self.ww_var = tk.IntVar(value=self.ww)
        self.wl_var = tk.IntVar(value=self.wl)
        ttk.Label(bar, text=" W").pack(side=tk.LEFT)
        sw = ttk.Spinbox(bar, from_=1, to=4000, increment=10, width=5, textvariable=self.ww_var,
                         command=self._wl_from_spin)
        sw.pack(side=tk.LEFT)
        ttk.Label(bar, text="L").pack(side=tk.LEFT)
        sl = ttk.Spinbox(bar, from_=-1000, to=1000, increment=5, width=5, textvariable=self.wl_var,
                         command=self._wl_from_spin)
        sl.pack(side=tk.LEFT)
        for s in (sw, sl):
            s.bind("<Return>", lambda e: self._wl_from_spin())
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(bar, text="Opacity").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=35)
        ttk.Scale(bar, from_=0, to=100, variable=self.alpha_var, length=P(120),
                  command=lambda _=None: self._schedule_all()).pack(side=tk.LEFT, padx=3)
        self.contour_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Contour (C)", variable=self.contour_var,
                        command=self._schedule_all).pack(side=tk.LEFT, padx=2)
        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Mask (M)", variable=self.show_var,
                        command=self._schedule_all).pack(side=tk.LEFT, padx=2)

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        axial = ttk.Frame(body)
        body.add(axial, weight=4)
        self.ax_scroll = ttk.Scrollbar(axial, orient=tk.VERTICAL, command=self._on_axial_scroll, takefocus=0)
        self.ax_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.ax_scroll.bind("<MouseWheel>", self._wheel_on_bar(self._on_axial_scroll))
        self.canvas = tk.Canvas(axial, bg="black", highlightthickness=0, cursor="tcross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side = ttk.Frame(body, padding=4)
        body.add(side, weight=2)

        top_side = ttk.Frame(side)
        top_side.pack(side=tk.TOP, fill=tk.X)
        self.side_mode = tk.StringVar(value="cor")
        ttk.Radiobutton(top_side, text="Coronal", value="cor", variable=self.side_mode,
                        command=self._schedule_side).pack(side=tk.LEFT)
        ttk.Radiobutton(top_side, text="Sagittal", value="sag", variable=self.side_mode,
                        command=self._schedule_side).pack(side=tk.LEFT)
        self.follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top_side, text="Follow cursor", variable=self.follow_var).pack(side=tk.LEFT, padx=8)
        side_view = ttk.Frame(side)
        side_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)
        self.side_scroll = ttk.Scrollbar(side_view, orient=tk.VERTICAL, command=self._on_side_scroll, takefocus=0)
        self.side_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_scroll.bind("<MouseWheel>", self._wheel_on_bar(self._on_side_scroll))
        self.side_canvas = tk.Canvas(side_view, bg="black", highlightthickness=0, height=P(420), cursor="hand2")
        self.side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        help_txt = (
            "Left click: selected tool   •   Right click: the opposite tool\n"
            "Wheel: slice   •   Ctrl+wheel: zoom   •   Shift+wheel: diameter\n"
            "Middle button / Space+left: pan   •   + / − (or [ ]): diameter ±1 mm\n"
            "↑ ↓: slice   •   PgUp/PgDn: ±10   •   0: reset view\n"
            "M: mask   •   C: contour   •   1/2/3: window\n"
            "Ctrl+Z / Ctrl+Y: undo / redo   •   Ctrl+S: save\n"
            "Click the side image to jump to that slice.\n"
            "Its scrollbar moves the coronal or sagittal plane and turns off Follow cursor."
        )
        ttk.Label(side, text=help_txt, justify=tk.LEFT, foreground="#555").pack(side=tk.TOP, fill=tk.X)

        self._img_item = self.canvas.create_image(0, 0, anchor="nw")
        self._cursor_item = self.canvas.create_oval(0, 0, 0, 0, outline="#3f3", width=1, state="hidden")
        self._txt_tl = self.canvas.create_text(8, 8, anchor="nw", fill="white", font=("Segoe UI", 10))
        self._txt_tr = self.canvas.create_text(0, 8, anchor="ne", fill="white", font=("Segoe UI", 10))
        self._txt_r = self.canvas.create_text(0, 0, anchor="w", fill="#ccc", text="R", font=("Segoe UI", 11, "bold"))
        self._txt_l = self.canvas.create_text(0, 0, anchor="e", fill="#ccc", text="L", font=("Segoe UI", 11, "bold"))
        self._side_img_item = self.side_canvas.create_image(0, 0, anchor="nw")
        self._side_line = self.side_canvas.create_line(0, 0, 0, 0, fill="#0ff", dash=(4, 3))
        self._side_txt = self.side_canvas.create_text(8, 8, anchor="nw", fill="white", font=("Segoe UI", 10))

        cv = self.canvas
        cv.bind("<Configure>", lambda e: self._schedule_render())
        cv.bind("<ButtonPress-1>", self._b1_down)
        cv.bind("<B1-Motion>", self._b1_move)
        cv.bind("<ButtonRelease-1>", self._b1_up)
        cv.bind("<ButtonPress-3>", self._b3_down)
        cv.bind("<B3-Motion>", self._paint_move)
        cv.bind("<ButtonRelease-3>", self._paint_end)
        cv.bind("<ButtonPress-2>", self._pan_start)
        cv.bind("<B2-Motion>", self._pan_move)
        cv.bind("<ButtonRelease-2>", lambda e: self._pan_end())
        cv.bind("<MouseWheel>", self._wheel)
        cv.bind("<Motion>", self._motion)
        cv.bind("<Leave>", lambda e: self._hide_cursor())
        self.side_canvas.bind("<Configure>", lambda e: self._schedule_side())
        self.side_canvas.bind("<ButtonPress-1>", self._side_click)
        self.side_canvas.bind("<B1-Motion>", self._side_click)

        cv.bind("<Enter>", lambda e: self.canvas.focus_set())

        def k(seq, fn, keys=None):
            # Shortcuts are off while typing in a text box (W/L)
            def handler(e, fn=fn):
                if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, ttk.Spinbox)):
                    return None
                fn()
                return "break"
            for s in ([seq] if isinstance(seq, str) else seq):
                self.bind(s, handler)

        k(("<KeyPress-b>", "<KeyPress-B>"), lambda: self._set_tool("brush"))
        k(("<KeyPress-e>", "<KeyPress-E>"), lambda: self._set_tool("eraser"))
        k(("<KeyPress-h>", "<KeyPress-H>"), lambda: self._set_tool("pan"))
        # [ ] need AltGr on some keyboards, so + and - (main and keypad) also change the diameter
        k(("<KeyPress-bracketleft>", "<KeyPress-minus>", "<KeyPress-KP_Subtract>"), lambda: self._step_diam(-1))
        k(("<KeyPress-bracketright>", "<KeyPress-plus>", "<KeyPress-KP_Add>"), lambda: self._step_diam(+1))
        k(("<Shift-KeyPress-braceleft>", "<Shift-KeyPress-KP_Subtract>"), lambda: self._step_diam(-5))
        k(("<Shift-KeyPress-braceright>", "<Shift-KeyPress-KP_Add>"), lambda: self._step_diam(+5))
        k("<KeyPress-Up>", lambda: self._step_slice(+1))
        k("<KeyPress-Down>", lambda: self._step_slice(-1))
        k("<KeyPress-Prior>", lambda: self._step_slice(+10))
        k("<KeyPress-Next>", lambda: self._step_slice(-10))
        k(("<KeyPress-m>", "<KeyPress-M>"), lambda: self._toggle(self.show_var))
        k(("<KeyPress-c>", "<KeyPress-C>"), lambda: self._toggle(self.contour_var))
        k("<KeyPress-0>", self._reset_view)
        for i, (_n, w, l) in enumerate(WL_PRESETS):
            k(f"<KeyPress-{i + 1}>", lambda w=w, l=l: self._set_wl(w, l))
        k(("<Control-z>", "<Control-Z>"), self.undo)
        k(("<Control-y>", "<Control-Y>"), self.redo)
        k(("<Control-s>", "<Control-S>"), lambda: self.save(sync=True))
        self.bind("<KeyPress-space>", self._space_down)
        self.bind("<KeyRelease-space>", self._space_up)
        self.bind("<Any-KeyPress>", self._touch, add="+")

    def _build_tool_strip(self, P):
        """Brush and eraser sit beside the image as large buttons, on the side of the screen where
        the liver lies, so they are one short mouse move away from the work."""
        strip = tk.Frame(self, bg=STRIP_BG, padx=P(8), pady=P(8))
        strip.pack(side=tk.LEFT, fill=tk.Y)
        big = ("Segoe UI", 13, "bold")
        normal = ("Segoe UI", 10)

        def tool(text, value, color, font=big, pad=16):
            tk.Radiobutton(strip, text=text, value=value, variable=self.tool_var, indicatoron=False,
                           font=font, fg="white", bg=BTN_BG, selectcolor=color, activebackground=color,
                           activeforeground="white", bd=2, width=10, pady=P(pad), takefocus=0,
                           command=self._redraw_cursor).pack(fill=tk.X, pady=(0, P(6)))

        def button(text, command, bg=BTN_BG, font=normal, pad=5):
            b = tk.Button(strip, text=text, command=command, font=font, fg="white", bg=bg,
                          activebackground=bg, activeforeground="white", bd=1, pady=P(pad), takefocus=0)
            b.pack(fill=tk.X, pady=(0, P(4)))
            return b

        tool("BRUSH  (B)", "brush", BRUSH_ON)
        tool("ERASER  (E)", "eraser", ERASER_ON)
        tool("Pan  (H)", "pan", PAN_ON, font=normal, pad=6)

        tk.Label(strip, text="Diameter", bg=STRIP_BG, fg=STRIP_FG, font=normal).pack(pady=(P(10), 0))
        self.diam_lbl = tk.Label(strip, text="10 mm", bg=STRIP_BG, fg="white", font=big)
        self.diam_lbl.pack()
        row = tk.Frame(strip, bg=STRIP_BG)
        row.pack(fill=tk.X, pady=(P(2), P(2)))
        for text, delta, side in (("−", -1, tk.LEFT), ("+", +1, tk.RIGHT)):
            tk.Button(row, text=text, font=big, fg="white", bg=BTN_BG, activebackground=BTN_BG,
                      activeforeground="white", bd=1, width=3, takefocus=0,
                      command=lambda d=delta: self._step_diam(d)).pack(side=side)
        ttk.Scale(strip, from_=DIAM_MIN, to=DIAM_MAX, variable=self.diam_var,
                  command=lambda _=None: self._on_diam()).pack(fill=tk.X, pady=(P(2), P(4)))
        tk.Checkbutton(strip, text="3D sphere", variable=self.sphere_var, bg=STRIP_BG, fg=STRIP_FG,
                       selectcolor=STRIP_BG, activebackground=STRIP_BG, activeforeground=STRIP_FG, font=normal,
                       takefocus=0, command=self._redraw_cursor).pack(anchor=tk.W, pady=(0, P(10)))

        button("Undo  (Ctrl+Z)", self.undo)
        button("Redo  (Ctrl+Y)", self.redo)
        tk.Frame(strip, bg=STRIP_BG).pack(fill=tk.BOTH, expand=True)
        button("Reset to TS", self._reset_to_ts)
        button("Save  (Ctrl+S)", lambda: self.save(sync=True))
        button("✔  DONE", self._complete, bg=BRUSH_ON, font=big, pad=10)

    # helpers
    def _touch(self, *_):
        self._last_input = time.time()

    def _toggle(self, var):
        var.set(not var.get())
        self._schedule_all()

    def _set_tool(self, t):
        self.tool_var.set(t)
        self._redraw_cursor()

    def _on_diam(self):
        d = int(round(self.diam_var.get()))
        self.diam_lbl.configure(text=f"{d} mm")
        self._redraw_cursor()
        self._update_status()

    @property
    def diam(self):
        return float(int(round(self.diam_var.get())))

    def _step_diam(self, delta):
        self.diam_var.set(min(DIAM_MAX, max(DIAM_MIN, self.diam + delta)))
        self._on_diam()

    def _set_wl(self, w, l):
        self.ww, self.wl = int(w), int(l)
        self.ww_var.set(self.ww)
        self.wl_var.set(self.wl)
        self._schedule_all()

    def _wl_from_spin(self):
        try:
            self.ww = max(1, int(self.ww_var.get()))
            self.wl = int(self.wl_var.get())
        except (tk.TclError, ValueError):
            return
        self._schedule_all()

    @staticmethod
    def _scroll_target(args, current, count):
        """Position a scrollbar asks for, from the top of the bar (0) to the bottom (count - 1)."""
        if args[0] == "moveto":
            target = int(float(args[1]) * count)
        else:
            target = current + int(args[1]) * (10 if args[2] == "pages" else 1)
        return min(count - 1, max(0, target))

    @staticmethod
    def _wheel_on_bar(on_scroll):
        """One slice per wheel notch, as on the image (a scrollbar's own binding would jump several)."""
        def handler(e):
            on_scroll("scroll", -1 if e.delta > 0 else 1, "units")
            return "break"
        return handler

    def _on_axial_scroll(self, *args):
        # The top of the bar is the head end, in line with the slice marker on the side image
        self._touch()
        self.z = self.Z - 1 - self._scroll_target(args, self.Z - 1 - self.z, self.Z)
        self._schedule_all()

    def _on_side_scroll(self, *args):
        self._touch()
        self.follow_var.set(False)  # the chosen plane must stay when the mouse returns to the axial image
        if self.side_mode.get() == "cor":
            self.side_y = self._scroll_target(args, self.side_y, self.H)
        else:
            self.side_x = self._scroll_target(args, self.side_x, self.W)
        self._schedule_side()

    def _step_slice(self, d):
        self.z = min(self.Z - 1, max(0, self.z + d))
        self._schedule_all()

    def _space_down(self, e):
        self._space = True
        return "break"   # Space must not trigger a focused button

    def _space_up(self, e):
        self._space = False
        return "break"

    def _alive(self):
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    # geometry
    def _update_scale(self):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        fit = min(cw / (self.W * self.sx), ch / (self.H * self.sy))
        self.px_per_mm = fit * self.zoom
        self.scale_x = self.px_per_mm * self.sx
        self.scale_y = self.px_per_mm * self.sy
        return cw, ch

    def _c2i(self, px, py):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        return (self.cx + (px - cw / 2) / self.scale_x, self.cy + (py - ch / 2) / self.scale_y)

    def _fit_initial(self):
        self._reset_view()

    def _reset_view(self):
        self.zoom, self.cx, self.cy = 1.0, self.W / 2, self.H / 2
        self._schedule_all()

    def _zoom_at(self, px, py, factor):
        ix, iy = self._c2i(px, py)
        self.zoom = min(16.0, max(0.5, self.zoom * factor))
        cw, ch = self._update_scale()
        self.cx = ix - (px - cw / 2) / self.scale_x
        self.cy = iy - (py - ch / 2) / self.scale_y
        self._schedule_render()

    # mouse
    def _b1_down(self, e):
        self._touch()
        self.canvas.focus_set()
        if self._space or self.tool_var.get() == "pan":
            self._pan_start(e)
        else:
            self._paint_start(e, 1 if self.tool_var.get() == "brush" else 0)

    def _b1_move(self, e):
        if self._pan_last is not None:
            self._pan_move(e)
        else:
            self._paint_move(e)

    def _b1_up(self, e):
        if self._pan_last is not None:
            self._pan_end()
        else:
            self._paint_end(e)

    def _b3_down(self, e):
        self._touch()
        t = self.tool_var.get()
        if t == "pan":
            return
        self._paint_start(e, 0 if t == "brush" else 1)

    def _pan_start(self, e):
        self._pan_last = (e.x, e.y)

    def _pan_move(self, e):
        if self._pan_last is None:
            return
        dx, dy = e.x - self._pan_last[0], e.y - self._pan_last[1]
        self._pan_last = (e.x, e.y)
        self.cx -= dx / self.scale_x
        self.cy -= dy / self.scale_y
        self._schedule_render()

    def _pan_end(self):
        self._pan_last = None

    def _wheel(self, e):
        self._touch()
        step = 1 if e.delta > 0 else -1
        if e.state & 0x0004:          # Ctrl
            self._zoom_at(e.x, e.y, 1.15 if step > 0 else 1 / 1.15)
        elif e.state & 0x0001:        # Shift
            self._step_diam(step)
        else:
            self._step_slice(step)

    def _motion(self, e):
        self._touch()
        self._mouse = (e.x, e.y)
        self._redraw_cursor()
        self._update_status()
        if self.follow_var.get() and self._painting is None:
            ix, iy = self._c2i(e.x, e.y)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                ny, nx = int(iy), int(ix)
                if (self.side_mode.get() == "cor" and ny != self.side_y) or \
                   (self.side_mode.get() == "sag" and nx != self.side_x):
                    self.side_y, self.side_x = ny, nx
                    self._schedule_side(delay=40)

    def _side_click(self, e):
        if not self._side_map:
            return
        oy, step, Z = self._side_map
        row = int((e.y - oy) // step)
        z = Z - 1 - row
        if 0 <= z < self.Z:
            self.z = z
            self._schedule_all()

    # painting
    def _paint_start(self, e, value):
        self._painting = value
        self._stroke_snap = {}
        ix, iy = self._c2i(e.x, e.y)
        self._last_pt = (ix, iy)
        self._stamp(ix, iy, value)
        self._schedule_render()

    def _paint_move(self, e):
        self._mouse = (e.x, e.y)
        if self._painting is None:
            return
        ix, iy = self._c2i(e.x, e.y)
        lx, ly = self._last_pt
        dist_mm = math.hypot((ix - lx) * self.sx, (iy - ly) * self.sy)
        step_mm = max(self.diam * 0.25, min(self.sx, self.sy) * 0.5)
        n = max(1, int(math.ceil(dist_mm / step_mm)))
        for i in range(1, n + 1):
            t = i / n
            self._stamp(lx + (ix - lx) * t, ly + (iy - ly) * t, self._painting)
        self._last_pt = (ix, iy)
        self._redraw_cursor()
        self._schedule_render()

    def _paint_end(self, e=None):
        if self._painting is None:
            return
        self._painting = None
        snap = self._stroke_snap
        self._stroke_snap = {}
        changed = {z: b for z, b in snap.items() if not np.array_equal(self.mask[z], b)}
        if changed:
            self._apply_counts(changed, sign=+1)
            self.undo_stack.append(changed)
            if len(self.undo_stack) > UNDO_LIMIT:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self.strokes += 1
            self.session["strokes"] += 1
            self._mark_dirty()
        self._schedule_all()

    def _snapshot(self, z):
        if z not in self._stroke_snap:
            self._stroke_snap[z] = self.mask[z].copy()

    def _stamp(self, ix, iy, value):
        r_mm = self.diam / 2.0
        if self.sphere_var.get():
            dz = int(math.floor(r_mm / self.sz))
            zs = range(max(0, self.z - dz), min(self.Z, self.z + dz + 1))
        else:
            zs = (self.z,)
        for z in zs:
            dz_mm = (z - self.z) * self.sz
            if abs(dz_mm) > r_mm:
                continue
            rr = math.sqrt(max(0.0, r_mm * r_mm - dz_mm * dz_mm))
            rx, ry = rr / self.sx, rr / self.sy
            if rx < 0.5 or ry < 0.5:
                if z != self.z:
                    continue
                c, r = int(math.floor(ix)), int(math.floor(iy))
                if 0 <= c < self.W and 0 <= r < self.H:
                    self._snapshot(z)
                    self.mask[z, r, c] = value
                continue
            c0, c1 = max(0, int(math.floor(ix - rx))), min(self.W, int(math.ceil(ix + rx)) + 1)
            r0, r1 = max(0, int(math.floor(iy - ry))), min(self.H, int(math.ceil(iy + ry)) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            cols = (np.arange(c0, c1) + 0.5 - ix) / rx
            rows = (np.arange(r0, r1) + 0.5 - iy) / ry
            disk = (cols[None, :] ** 2 + rows[:, None] ** 2) <= 1.0
            if not disk.any():
                continue
            self._snapshot(z)
            self.mask[z, r0:r1, c0:c1][disk] = value

    # counts / undo
    def _recount(self):
        m = self.mask > 0
        self.n_vox = int(m.sum())
        self.added = int((m & ~self.ts).sum())
        self.removed = int((~m & self.ts).sum())

    def _apply_counts(self, before, sign):
        """before: {z: previous_slice}; adds the difference from the current mask[z] to the counters."""
        for z, b in before.items():
            b = b > 0
            a = self.mask[z] > 0
            t = self.ts[z]
            self.n_vox += int(a.sum()) - int(b.sum())
            self.added += int((a & ~t).sum()) - int((b & ~t).sum())
            self.removed += int((~a & t).sum()) - int((~b & t).sum())

    def undo(self):
        self._swap(self.undo_stack, self.redo_stack)

    def redo(self):
        self._swap(self.redo_stack, self.undo_stack)

    def _swap(self, src, dst):
        if not src:
            return
        entry = src.pop()
        other = {}
        for z, b in entry.items():
            other[z] = self.mask[z].copy()
            self.mask[z] = b
        self._apply_counts(other, sign=+1)
        dst.append(other)
        self._mark_dirty()
        if len(entry) == 1:
            self.z = next(iter(entry))
        self._schedule_all()

    def _reset_to_ts(self):
        if not messagebox.askyesno("Reset to TS",
                                   "Delete all your corrections and return the mask to the TotalSegmentator output?\n"
                                   "(This cannot be undone.)", parent=self):
            return
        self.mask[:] = self.ts.astype(np.uint8)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._recount()
        self._mark_dirty()
        self._schedule_all()

    def _mark_dirty(self):
        self._dirty = True
        if self.log.get("status") == C.DONE:
            self.log["status"] = C.IN_PROGRESS
            self.log["reopened_after_complete"] = int(self.log.get("reopened_after_complete", 0)) + 1
            self.log["completed_at"] = None

    # drawing
    def _schedule_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.after_idle(self._render)

    def _schedule_side(self, delay=0):
        if not self._side_pending:
            self._side_pending = True
            if delay:
                self.after(delay, self._render_side)
            else:
                self.after_idle(self._render_side)

    def _schedule_all(self):
        self._schedule_render()
        self._schedule_side()
        self._update_status()

    def _window(self, arr):
        lo = self.wl - self.ww / 2.0
        g = (arr.astype(np.float32) - lo) * (255.0 / self.ww)
        return np.clip(g, 0, 255).astype(np.uint8)

    def _gray_slice(self):
        key = (self.z, self.ww, self.wl)
        if key != self._gray_key:
            self._gray = self._window(self.ct[self.z])
            self._gray_key = key
        return self._gray

    def _compose(self, gray, m):
        rgb = np.repeat(gray[..., None], 3, axis=2)
        if self.show_var.get():
            mb = m > 0
            if mb.any():
                a = float(self.alpha_var.get()) / 100.0
                if a > 0:
                    sel = rgb[mb].astype(np.float32)
                    rgb[mb] = (sel * (1 - a) + FILL_RGB * a).astype(np.uint8)
                if self.contour_var.get():
                    edge = mb & ~ndimage.binary_erosion(mb, border_value=1)
                    rgb[edge] = CONTOUR_RGB
        return rgb

    def _render(self):
        self._render_pending = False
        if not self._alive():
            return
        cw, ch = self._update_scale()
        x0 = self.cx - cw / 2 / self.scale_x
        y0 = self.cy - ch / 2 / self.scale_y
        c0, c1 = max(0, int(math.floor(x0))), min(self.W, int(math.ceil(x0 + cw / self.scale_x)))
        r0, r1 = max(0, int(math.floor(y0))), min(self.H, int(math.ceil(y0 + ch / self.scale_y)))
        if c1 > c0 and r1 > r0:
            rgb = self._compose(self._gray_slice()[r0:r1, c0:c1], self.mask[self.z, r0:r1, c0:c1])
            ow = max(1, int(round((c1 - c0) * self.scale_x)))
            oh = max(1, int(round((r1 - r0) * self.scale_y)))
            resample = Image.NEAREST if self.scale_x >= 2 else Image.BILINEAR
            img = Image.fromarray(rgb).resize((ow, oh), resample)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfigure(self._img_item, image=self._photo, state="normal")
            self.canvas.coords(self._img_item, (c0 - x0) * self.scale_x, (r0 - y0) * self.scale_y)
        else:
            self.canvas.itemconfigure(self._img_item, state="hidden")
        state = "DONE" if self.log.get("status") == C.DONE else "in progress"
        self.canvas.itemconfigure(self._txt_tl, text=f"Slice {self.z + 1}/{self.Z}\nW {self.ww}  L {self.wl}")
        self.canvas.coords(self._txt_tr, cw - 8, 8)
        self.canvas.itemconfigure(self._txt_tr, text=f"{self.reader}  ·  {self.case_id}\n{state}", justify=tk.RIGHT)
        self.canvas.coords(self._txt_r, 8, ch / 2)
        self.canvas.coords(self._txt_l, cw - 8, ch / 2)
        top = self.Z - 1 - self.z
        self.ax_scroll.set(top / self.Z, (top + 1) / self.Z)
        for it in (self._txt_tl, self._txt_tr, self._txt_r, self._txt_l, self._cursor_item):
            self.canvas.tag_raise(it)
        self._redraw_cursor()

    def _render_side(self):
        self._side_pending = False
        if not self._alive():
            return
        cw, ch = max(1, self.side_canvas.winfo_width()), max(1, self.side_canvas.winfo_height())
        if self.side_mode.get() == "cor":
            pos = min(self.H - 1, max(0, self.side_y))
            img, m, horiz_mm = self.ct[:, pos, :], self.mask[:, pos, :], self.W * self.sx
        else:
            pos = min(self.W - 1, max(0, self.side_x))
            img, m, horiz_mm = self.ct[:, :, pos], self.mask[:, :, pos], self.H * self.sy
        rgb = self._compose(self._window(img[::-1]), m[::-1])
        vert_mm = self.Z * self.sz
        s = min(cw / horiz_mm, ch / vert_mm)
        ow, oh = max(1, int(round(horiz_mm * s))), max(1, int(round(vert_mm * s)))
        ox, oy = (cw - ow) / 2, (ch - oh) / 2
        self._side_photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((ow, oh), Image.BILINEAR))
        self.side_canvas.itemconfigure(self._side_img_item, image=self._side_photo)
        self.side_canvas.coords(self._side_img_item, ox, oy)
        step = self.sz * s
        ly = oy + (self.Z - 1 - self.z + 0.5) * step
        self.side_canvas.coords(self._side_line, ox, ly, ox + ow, ly)
        self.side_canvas.tag_raise(self._side_line)
        self._side_map = (oy, step, self.Z)
        coronal = self.side_mode.get() == "cor"
        count = self.H if coronal else self.W
        self.side_scroll.set(pos / count, (pos + 1) / count)
        self.side_canvas.itemconfigure(self._side_txt, text=f"{'Coronal' if coronal else 'Sagittal'} {pos + 1}/{count}")
        self.side_canvas.tag_raise(self._side_txt)

    def _hide_cursor(self):
        self._mouse = None
        self.canvas.itemconfigure(self._cursor_item, state="hidden")

    def _redraw_cursor(self):
        t = self.tool_var.get()
        if self._mouse is None or t == "pan":
            self.canvas.itemconfigure(self._cursor_item, state="hidden")
            return
        x, y = self._mouse
        r = max(1.5, self.diam / 2 * self.px_per_mm)
        if self._painting is not None:
            color = "#3f3" if self._painting == 1 else "#f44"
        else:
            color = "#3f3" if t == "brush" else "#f44"
        dash = (4, 3) if self.sphere_var.get() else ""
        self.canvas.coords(self._cursor_item, x - r, y - r, x + r, y + r)
        self.canvas.itemconfigure(self._cursor_item, state="normal", outline=color, dash=dash)

    def _update_status(self):
        hu = ""
        if self._mouse is not None:
            ix, iy = self._c2i(*self._mouse)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                hu = f"{int(self.ct[self.z, int(iy), int(ix)])} HU"
        vol = self.n_vox * self.voxel_mm3 / 1000.0
        m, s = divmod(int(self.active), 60)
        st = "DONE" if self.log.get("status") == C.DONE else "in progress"
        dirty = "  •  NOT SAVED" if self._dirty else ""
        err = f"  •  SAVE ERROR: {self._save_error}" if self._save_error else ""
        sph = " (3D sphere)" if self.sphere_var.get() else ""
        self.status.configure(text=(
            f"Slice {self.z + 1}/{self.Z}   |   {hu:>8}   |   Diameter {self.diam:.0f} mm{sph}   |   "
            f"Mask {vol:.1f} mL   |   vs TS +{self.added} / −{self.removed} voxels   |   "
            f"Active time {m:02d}:{s:02d}   |   Status: {st}{dirty}{err}"
        ))

    # time / saving
    def _tick(self):
        if not self._alive():
            return
        now = time.time()
        if now - self._last_input < IDLE_LIMIT_S:
            dt = now - self._last_tick
            self.active += dt
            self.session["active_seconds"] += dt
        self._last_tick = now
        if self._dirty and now - self._last_autosave > AUTOSAVE_S:
            self.save(sync=False)
        self._update_status()
        self.after(1000, self._tick)

    def _build_log(self, status=None):
        log = dict(self.log)
        now = datetime.now().isoformat(timespec="seconds")
        self.session["end"] = now
        sessions = list(log.get("sessions", []))
        if sessions and sessions[-1].get("start") == self.session["start"]:
            sessions[-1] = dict(self.session)
        else:
            sessions.append(dict(self.session))
        log.update({
            "reader": self.reader, "case_id": self.case_id,
            "status": status or log.get("status", C.IN_PROGRESS),
            "active_seconds": round(self.active, 1),
            "strokes": self.strokes,
            "voxels": self.n_vox,
            "volume_ml": round(self.n_vox * self.voxel_mm3 / 1000.0, 3),
            "voxels_added_vs_ts": self.added,
            "voxels_removed_vs_ts": self.removed,
            "last_saved": now,
            "sessions": sessions,
        })
        if status == C.DONE:
            log["completed_at"] = now
        return log

    def save(self, sync=False, status=None):
        self._recount()
        arr = self.mask.copy()
        log = self._build_log(status)
        self.log = log
        self._dirty = False
        self._last_autosave = time.time()

        def work():
            try:
                with self._save_lock:
                    img = sitk.GetImageFromArray(arr)
                    img.CopyInformation(self.ct_img)
                    tmp = self.dir / f"{self.reader}.tmp.nii.gz"
                    sitk_io.write_image(img, tmp)
                    tmp.replace(self.mask_path)
                    write_json(self.log_path, log)
                self._save_error = None
            except Exception as exc:
                self._save_error = str(exc)
                self._dirty = True
                print(f"SAVE ERROR {self.case_id}/{self.reader}: {exc}", file=sys.stderr)

        if sync:
            work()
            if self._save_error:
                messagebox.showerror("Save error", self._save_error, parent=self)
        else:
            threading.Thread(target=work, daemon=False).start()
        self._update_status()

    def _complete(self):
        if not messagebox.askyesno("Done",
                                   f"Mark {self.case_id} as done for {self.reader}?",
                                   parent=self):
            return
        self.save(sync=True, status=C.DONE)
        if not self._save_error:
            self._close()

    def _on_close(self):
        if self._dirty:
            ans = messagebox.askyesnocancel("Unsaved changes",
                                            "Save your changes?", parent=self)
            if ans is None:
                return
            if ans:
                self.save(sync=True)
                if self._save_error:
                    return
            else:
                # corrections are discarded; the saved record (including status) stays, only the time is added
                disk = read_json(self.log_path, None)
                if disk is not None:
                    fresh = self._build_log()
                    disk["active_seconds"] = fresh["active_seconds"]
                    disk["sessions"] = fresh["sessions"]
                    write_json(self.log_path, disk)
        else:
            # mask unchanged; still record the session time
            if self.mask_path.exists() or self.session["active_seconds"] > 5:
                self.save(sync=True)
        self._close()

    def _close(self):
        with self._save_lock:
            pass
        cb = self.on_closed
        self.destroy()
        if cb:
            cb(self.case_id, self.reader)


def main():
    if len(sys.argv) < 3:
        print("Usage: python editor.py <CASE> <AUTH1|AUTH2>")
        return 2
    enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    ed = CorrectionEditor(root, sys.argv[1], sys.argv[2], on_closed=lambda *_: root.destroy())
    ed.focus_force()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
