#!/usr/bin/env python3
"""
Color Preprocessor
==================
Interactive desktop tool for cleaning up scanned font images by remapping
groups of similar pixels to clean target colors.

Usage:
    python color_preprocessor.py [image_path]

Workflow:
    1. Open a scanned image.
    2. Add color slots manually ("+ Add Color") or via K-means auto-suggest.
    3. For each slot: click "pick from image" then click a pixel to sample,
       click the target swatch to choose the replacement color, adjust tolerance.
    4. Click Preview (or enable Live Preview) to see the result.
    5. Toggle Original / Preview to compare, then Export when satisfied.
"""

import sys
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

try:
    import cupy as xp          # NVIDIA GPU
except ImportError:
    import numpy as xp
#import numpy as np
from PIL import Image, ImageTk
from pathlib import Path


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def text_color_for_bg(r: int, g: int, b: int) -> str:
    """Return 'black' or 'white' for legible text on a given background."""
    luminance = (r * 299 + g * 587 + b * 114) / 1000
    return "black" if luminance > 128 else "white"


def rgb_to_lab(arr: xp.ndarray) -> xp.ndarray:
    """
    Convert an (N, 3) float32/uint8 RGB array (values 0-255) to CIE L*a*b*.
    Uses the D65 illuminant.  Pure NumPy — no extra dependencies.
    """
    rgb = arr.astype(xp.float64) / 255.0

    # sRGB gamma expansion → linear light
    linear = xp.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    # Linear RGB → XYZ  (IEC 61966-2-1, D65)
    M = xp.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ M.T  # (N, 3)

    # XYZ → L*a*b*
    ref = xp.array([0.95047, 1.00000, 1.08883])  # D65 white point
    xyz_n = xyz / ref

    eps, kap = 0.008856, 903.3
    f = xp.where(xyz_n > eps, xp.cbrt(xyz_n), (kap * xyz_n + 16.0) / 116.0)

    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])

    return xp.stack([L, a, b], axis=1).astype(xp.float32)


# Approximate maximum distances in each color space (used to scale tolerance)
_MAX_RGB_DIST = float(xp.sqrt(3 * 255**2))   # ≈ 441.7
_MAX_LAB_DIST = float(xp.sqrt(100**2 + 255**2 + 255**2))  # ≈ 383 (generous)


def kmeans_colors(image: Image.Image, k: int, max_iter: int = 25) -> list:
    """
    Find k dominant colors in *image* using Lloyd's K-means algorithm.
    Operates on a random subsample of ≤ 10 000 pixels for speed.
    Returns a list of k (r, g, b) tuples sorted by cluster size (largest first).
    """
    arr = xp.array(image.convert("RGB"), dtype=xp.float32)
    pixels = arr.reshape(-1, 3)

    # Subsample
    if len(pixels) > 10_000:
        rng = xp.random.default_rng(42)
        idx = rng.choice(len(pixels), 10_000, replace=False)
        pixels = pixels[idx]

    # K-means++ initialisation
    rng = xp.random.default_rng(0)
    centroids = [pixels[rng.integers(len(pixels))]]
    for _ in range(k - 1):
        dists = xp.min(
            xp.linalg.norm(pixels[:, None] - xp.array(centroids)[None], axis=2),
            axis=1,
        )
        probs = dists**2 / (dists**2).sum()
        centroids.append(pixels[rng.choice(len(pixels), p=probs)])
    centroids = xp.array(centroids, dtype=xp.float32)  # (k, 3)

    labels = xp.zeros(len(pixels), dtype=xp.int32)
    for _ in range(max_iter):
        # Assignment step
        diff = pixels[:, None, :] - centroids[None, :, :]  # (P, k, 3)
        new_labels = xp.argmin(xp.sum(diff**2, axis=2), axis=1)
        if xp.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update step
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = pixels[mask].mean(axis=0)

    # Sort by cluster size (most dominant first)
    counts = [(xp.sum(labels == j), j) for j in range(k)]
    counts.sort(reverse=True)
    result = []
    for _, j in counts:
        r, g, b = centroids[j].round().astype(xp.uint8)
        result.append((int(r), int(g), int(b)))
    return result


# ---------------------------------------------------------------------------
# ColorSlot widget
# ---------------------------------------------------------------------------

class ColorSlot:
    """
    One row in the right-hand control panel.  Represents a single color mapping:
    sample color → target color, with a per-slot tolerance slider.
    """

    def __init__(
        self,
        parent: tk.Widget,
        index: int,
        on_request_sample,   # callable(slot)
        on_change,           # callable()
        on_delete,           # callable(slot)
    ):
        self.index = index
        self.sample_rgb: tuple | None = None
        self.target_rgb: tuple = (0, 0, 0)
        self._on_request_sample = on_request_sample
        self._on_change = on_change
        self._build(parent, on_delete)

    # ------------------------------------------------------------------ build

    def _build(self, parent: tk.Widget, on_delete) -> None:
        self.frame = ttk.LabelFrame(parent, text=f"  Color {self.index + 1}  ")
        self.frame.pack(fill="x", padx=6, pady=4)

        # Row: Sampled color
        row = ttk.Frame(self.frame)
        row.pack(fill="x", padx=6, pady=(5, 1))
        ttk.Label(row, text="Sampled:", width=8, anchor="e").pack(side="left")
        self.sample_btn = tk.Button(
            row,
            text="pick from image",
            width=17,
            bg="#e0e0e0",
            fg="black",
            relief="groove",
            command=self._request_sample,
        )
        self.sample_btn.pack(side="left", padx=4)

        # Row: Target color
        row = ttk.Frame(self.frame)
        row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text="Target:", width=8, anchor="e").pack(side="left")
        self.target_btn = tk.Button(
            row,
            text="#000000",
            width=10,
            bg="#000000",
            fg="white",
            relief="groove",
            command=self._pick_target,
        )
        self.target_btn.pack(side="left", padx=4)

        # Row: Tolerance slider
        row = ttk.Frame(self.frame)
        row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text="Tolerance:", width=8, anchor="e").pack(side="left")
        self._tol_var = tk.IntVar(value=30)
        self._tol_label = ttk.Label(row, text=" 30", width=4)
        ttk.Scale(
            row,
            from_=0,
            to=100,
            variable=self._tol_var,
            orient="horizontal",
            command=self._on_tol_change,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._tol_label.pack(side="left")

        # Row: Enabled toggle + delete
        row = ttk.Frame(self.frame)
        row.pack(fill="x", padx=6, pady=(1, 5))
        self._enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row, text="Enabled", variable=self._enabled_var, command=self._on_change
        ).pack(side="left")
        ttk.Button(row, text="Delete", command=lambda: on_delete(self)).pack(
            side="right"
        )

    # ---------------------------------------------------------------- actions

    def _request_sample(self) -> None:
        self._on_request_sample(self)

    def _pick_target(self) -> None:
        initial = rgb_to_hex(*self.target_rgb)
        result = colorchooser.askcolor(color=initial, title="Choose Target Color")
        if result[1]:
            self.target_rgb = hex_to_rgb(result[1])
            self._refresh_target_btn()
            self._on_change()

    def _on_tol_change(self, _=None) -> None:
        self._tol_label.config(text=f"{self._tol_var.get():3d}")
        self._on_change()

    # ---------------------------------------------------------------- state

    def activate_sample_mode(self) -> None:
        self.sample_btn.config(text="◉  click image…", bg="#ffe080", fg="black")

    def deactivate_sample_mode(self) -> None:
        if self.sample_rgb:
            r, g, b = self.sample_rgb
            hex_c = rgb_to_hex(r, g, b)
            self.sample_btn.config(
                text=hex_c, bg=hex_c, fg=text_color_for_bg(r, g, b)
            )
        else:
            self.sample_btn.config(text="pick from image", bg="#e0e0e0", fg="black")

    def set_sample(self, rgb: tuple) -> None:
        self.sample_rgb = tuple(int(v) for v in rgb)
        self.deactivate_sample_mode()
        self._on_change()

    def set_target(self, rgb: tuple) -> None:
        self.target_rgb = tuple(int(v) for v in rgb)
        self._refresh_target_btn()

    def _refresh_target_btn(self) -> None:
        r, g, b = self.target_rgb
        hex_c = rgb_to_hex(r, g, b)
        self.target_btn.config(bg=hex_c, fg=text_color_for_bg(r, g, b), text=hex_c)

    # ---------------------------------------------------------------- getters

    @property
    def tolerance(self) -> int:
        return self._tol_var.get()

    @property
    def enabled(self) -> bool:
        return self._enabled_var.get()

    def is_ready(self) -> bool:
        return self.sample_rgb is not None and self.enabled


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class App:
    _ZOOM_LEVELS = [0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    _DEFAULT_ZOOM = 6  # index into _ZOOM_LEVELS → 1.0×

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Color Preprocessor")
        self.root.minsize(960, 620)

        self._orig_image: Image.Image | None = None
        self._proc_image: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None  # keep a reference!

        self._zoom_idx: int = self._DEFAULT_ZOOM
        self._showing_proc: bool = False
        self._color_slots: list[ColorSlot] = []
        self._active_slot: ColorSlot | None = None

        self._use_lab = tk.BooleanVar(value=True)
        self._live_prev = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_keys()

    # ================================================================ UI build

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        tb = ttk.Frame(self.root)
        tb.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)

        ttk.Button(tb, text="Open Image",   command=self._open).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(tb, text="Preview",      command=self._do_preview).pack(side="left", padx=2)
        ttk.Button(tb, text="Show Original",command=self._show_original).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(tb, text="Export…",      command=self._export).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Checkbutton(tb, text="LAB color space", variable=self._use_lab).pack(side="left", padx=2)
        ttk.Checkbutton(tb, text="Live preview",    variable=self._live_prev).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(tb, text="−", width=2, command=self._zoom_out).pack(side="left")
        self._zoom_label = ttk.Label(tb, text="100%", width=5, anchor="center")
        self._zoom_label.pack(side="left")
        ttk.Button(tb, text="+", width=2, command=self._zoom_in).pack(side="left")

        # ── Image canvas ──────────────────────────────────────────────────────
        cf = ttk.Frame(self.root)
        cf.grid(row=1, column=0, sticky="nsew", padx=(4, 0))
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(cf, bg="#404040", cursor="crosshair", highlightthickness=0)
        h_sb = ttk.Scrollbar(cf, orient="horizontal", command=self._canvas.xview)
        v_sb = ttk.Scrollbar(cf, orient="vertical",   command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=h_sb.set, yscrollcommand=v_sb.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        v_sb.grid(row=0, column=1, sticky="ns")
        h_sb.grid(row=1, column=0, sticky="ew")

        self._canvas.bind("<Button-1>",   self._canvas_click)
        self._canvas.bind("<Motion>",     self._canvas_hover)
        self._canvas.bind("<MouseWheel>", self._canvas_scroll_zoom)  # macOS / Win
        self._canvas.bind("<Button-4>",   lambda _e: self._zoom_in())   # Linux scroll up
        self._canvas.bind("<Button-5>",   lambda _e: self._zoom_out())  # Linux scroll down

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = tk.StringVar(value="Open an image to begin.")
        ttk.Label(
            self.root, textvariable=self._status, foreground="gray", anchor="w"
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=2)

        # ── Right panel ───────────────────────────────────────────────────────
        rp = ttk.Frame(self.root, width=275)
        rp.grid(row=1, column=1, sticky="nsew", padx=4, pady=0)
        rp.columnconfigure(0, weight=1)
        rp.rowconfigure(2, weight=1)

        ttk.Button(rp, text="＋  Add Color Slot", command=self._add_slot).grid(
            row=0, column=0, sticky="ew", padx=6, pady=(8, 2)
        )
        ttk.Button(rp, text="✦  Auto-suggest colors…", command=self._auto_suggest).grid(
            row=1, column=0, sticky="ew", padx=6, pady=(0, 6)
        )

        # Scrollable slot area
        slot_outer = ttk.Frame(rp)
        slot_outer.grid(row=2, column=0, sticky="nsew")
        slot_outer.rowconfigure(0, weight=1)
        slot_outer.columnconfigure(0, weight=1)

        self._slot_canvas = tk.Canvas(slot_outer, highlightthickness=0)
        slot_sb = ttk.Scrollbar(slot_outer, orient="vertical", command=self._slot_canvas.yview)
        self._slot_canvas.configure(yscrollcommand=slot_sb.set)

        self._slot_canvas.grid(row=0, column=0, sticky="nsew")
        slot_sb.grid(row=0, column=1, sticky="ns")

        self._slots_frame = ttk.Frame(self._slot_canvas)
        self._slots_win = self._slot_canvas.create_window(
            (0, 0), window=self._slots_frame, anchor="nw"
        )

        def _on_slots_resize(e=None):
            self._slot_canvas.configure(scrollregion=self._slot_canvas.bbox("all"))
            self._slot_canvas.itemconfig(
                self._slots_win, width=self._slot_canvas.winfo_width()
            )

        self._slots_frame.bind("<Configure>", _on_slots_resize)
        self._slot_canvas.bind("<Configure>", _on_slots_resize)

        # Forward mouse-wheel on the slot panel to the slot canvas
        self._slots_frame.bind(
            "<MouseWheel>",
            lambda e: self._slot_canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )

    def _bind_keys(self) -> None:
        self.root.bind("<Control-o>", lambda _: self._open())
        self.root.bind("<Control-O>", lambda _: self._open())
        self.root.bind("<Control-p>", lambda _: self._do_preview())
        self.root.bind("<Control-P>", lambda _: self._do_preview())
        self.root.bind("<Control-s>", lambda _: self._export())
        self.root.bind("<Control-S>", lambda _: self._export())
        self.root.bind("<Escape>",    lambda _: self._cancel_sample_mode())

    # ================================================================ image

    def _open(self, path: str | None = None) -> None:
        if path is None:
            path = filedialog.askopenfilename(
                title="Open Image",
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp"),
                    ("All files", "*.*"),
                ],
            )
        if not path:
            return
        try:
            self._orig_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return

        self._proc_image = None
        self._showing_proc = False
        self._zoom_idx = self._DEFAULT_ZOOM
        self._render_canvas(self._orig_image)
        name = Path(path).name
        w, h = self._orig_image.size
        self._status.set(f"Loaded {name}  ({w} × {h} px)  —  add color slots or use Auto-suggest.")

    # ================================================================ canvas

    def _zoom(self) -> float:
        return self._ZOOM_LEVELS[self._zoom_idx]

    def _zoom_in(self) -> None:
        if self._zoom_idx < len(self._ZOOM_LEVELS) - 1:
            self._zoom_idx += 1
            self._refresh_display()

    def _zoom_out(self) -> None:
        if self._zoom_idx > 0:
            self._zoom_idx -= 1
            self._refresh_display()

    def _canvas_scroll_zoom(self, event) -> None:
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _refresh_display(self) -> None:
        img = self._proc_image if self._showing_proc else self._orig_image
        if img:
            self._render_canvas(img)

    def _render_canvas(self, image: Image.Image) -> None:
        z = self._zoom()
        w = max(1, int(image.width * z))
        h = max(1, int(image.height * z))
        resample = Image.NEAREST if z >= 1 else Image.LANCZOS
        display = image.resize((w, h), resample)
        self._tk_image = ImageTk.PhotoImage(display)
        self._canvas.configure(scrollregion=(0, 0, w, h))
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
        self._zoom_label.config(text=f"{int(z * 100)}%")

    def _canvas_xy(self, event) -> tuple[int, int]:
        """Convert a canvas mouse event to original-image pixel coordinates."""
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        z = self._zoom()
        return int(cx / z), int(cy / z)

    def _pixel_at(self, ix: int, iy: int) -> tuple | None:
        if self._orig_image is None:
            return None
        if 0 <= ix < self._orig_image.width and 0 <= iy < self._orig_image.height:
            return self._orig_image.getpixel((ix, iy))[:3]
        return None

    def _canvas_click(self, event) -> None:
        if self._orig_image is None:
            return
        ix, iy = self._canvas_xy(event)
        rgb = self._pixel_at(ix, iy)
        if rgb is None:
            return

        if self._active_slot is not None:
            slot = self._active_slot
            self._active_slot = None
            slot.set_sample(rgb)
            self._canvas.config(cursor="crosshair")
            self._status.set(
                f"Color {slot.index + 1}: sampled {rgb_to_hex(*rgb)}  "
                f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
            )
            if self._live_prev.get():
                self._do_preview()
        else:
            self._status.set(
                f"({ix}, {iy})  →  {rgb_to_hex(*rgb)}  "
                f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
            )

    def _canvas_hover(self, event) -> None:
        if self._orig_image is None:
            return
        ix, iy = self._canvas_xy(event)
        rgb = self._pixel_at(ix, iy)
        if rgb is None:
            return
        if self._active_slot:
            self._status.set(
                f"Click to sample for Color {self._active_slot.index + 1}  —  "
                f"({ix}, {iy}) = {rgb_to_hex(*rgb)}"
            )
        else:
            self._status.set(f"({ix}, {iy})  →  {rgb_to_hex(*rgb)}")

    # ================================================================ slots

    def _add_slot(self, sample_rgb: tuple | None = None) -> ColorSlot:
        slot = ColorSlot(
            parent=self._slots_frame,
            index=len(self._color_slots),
            on_request_sample=self._enter_sample_mode,
            on_change=self._slot_changed,
            on_delete=self._delete_slot,
        )
        if sample_rgb is not None:
            slot.set_sample(sample_rgb)
        self._color_slots.append(slot)
        return slot

    def _auto_suggest(self) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        n = simpledialog.askinteger(
            "Auto-suggest",
            "How many dominant colors should be detected?",
            initialvalue=3,
            minvalue=1,
            maxvalue=16,
            parent=self.root,
        )
        if not n:
            return

        self._status.set("Running K-means color detection…")
        self.root.update_idletasks()
        try:
            colors = kmeans_colors(self._orig_image, n)
        except Exception as exc:
            messagebox.showerror("K-means failed", str(exc))
            self._status.set("Auto-suggest failed.")
            return

        for rgb in colors:
            slot = self._add_slot(sample_rgb=rgb)
            # Default target: same as sample (user will change it to the clean color)
            slot.set_target(rgb)

        self._status.set(
            f"Added {n} color slot(s) from dominant colors.  "
            "Set each target color, adjust tolerances, then Preview."
        )

    def _enter_sample_mode(self, slot: ColorSlot) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        # Deactivate any previously active slot
        if self._active_slot and self._active_slot is not slot:
            self._active_slot.deactivate_sample_mode()
        self._active_slot = slot
        slot.activate_sample_mode()
        self._canvas.config(cursor="plus")
        self._status.set(
            f"Click on the image to sample a color for Color {slot.index + 1}…  "
            "(Esc to cancel)"
        )

    def _cancel_sample_mode(self) -> None:
        if self._active_slot:
            self._active_slot.deactivate_sample_mode()
            self._active_slot = None
            self._canvas.config(cursor="crosshair")
            self._status.set("Sample mode cancelled.")

    def _delete_slot(self, slot: ColorSlot) -> None:
        if self._active_slot is slot:
            self._active_slot = None
            self._canvas.config(cursor="crosshair")
        slot.frame.destroy()
        self._color_slots.remove(slot)
        # Re-number remaining slots
        for i, s in enumerate(self._color_slots):
            s.index = i
            s.frame.config(text=f"  Color {i + 1}  ")
        self._slot_changed()

    def _slot_changed(self) -> None:
        if self._live_prev.get() and self._orig_image is not None:
            self._do_preview()

    # ================================================================ processing

    def _do_preview(self) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        self._status.set("Processing…")
        self.root.update_idletasks()
        self._proc_image = self._process(self._orig_image)
        self._showing_proc = True
        self._render_canvas(self._proc_image)
        self._status.set("Preview  —  click 'Show Original' to compare, or Export when done.")

    def _show_original(self) -> None:
        if self._orig_image is None:
            return
        self._showing_proc = False
        self._render_canvas(self._orig_image)
        self._status.set("Showing original image.")

    def _process(self, image: Image.Image) -> Image.Image:
        """
        Vectorised NumPy color-remapping pipeline.

        For each pixel, compute its distance (in RGB or LAB space) to every
        active slot's sample color.  The closest matching slot within its
        tolerance threshold wins and the pixel is replaced with that slot's
        target color.  Pixels that match no slot are left unchanged.
        """
        active = [s for s in self._color_slots if s.is_ready()]
        if not active:
            return image.copy()

        arr = xp.array(image, dtype=xp.uint8)
        H, W = arr.shape[:2]
        flat = arr.reshape(-1, 3).astype(xp.float32)  # (P, 3)

        samples = xp.array([s.sample_rgb for s in active], dtype=xp.float32)   # (N, 3)
        targets = xp.array([s.target_rgb for s in active], dtype=xp.uint8)     # (N, 3)

        if self._use_lab.get():
            flat_sp    = rgb_to_lab(flat)     # (P, 3)
            samples_sp = rgb_to_lab(samples)  # (N, 3)
            max_dist   = _MAX_LAB_DIST
        else:
            flat_sp    = flat
            samples_sp = samples
            max_dist   = _MAX_RGB_DIST

        # Scaled tolerance thresholds for each slot
        tols = xp.array(
            [(s.tolerance / 100.0) * max_dist for s in active], dtype=xp.float32
        )  # (N,)

        # Per-pixel distance to each sample: (P, N)
        diff = flat_sp[:, None, :] - samples_sp[None, :, :]   # (P, N, 3)
        dist = xp.sqrt(xp.einsum("pnc,pnc->pn", diff, diff))  # (P, N)

        within    = dist <= tols[None, :]           # (P, N)  bool
        dist_m    = xp.where(within, dist, xp.inf)  # (P, N)  masked
        best      = xp.argmin(dist_m, axis=1)       # (P,)
        has_match = xp.any(within, axis=1)           # (P,)

        result = flat.astype(xp.uint8).copy()
        result[has_match] = targets[best[has_match]]

        return Image.fromarray(result.reshape(H, W, 3))

    # ================================================================ export

    def _export(self) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if self._proc_image is None:
            if not messagebox.askyesno(
                "No preview",
                "No preview has been generated yet.\nProcess and export now?",
            ):
                return
            self._proc_image = self._process(self._orig_image)

        path = filedialog.asksaveasfilename(
            title="Export processed image",
            defaultextension=".png",
            filetypes=[
                ("PNG",       "*.png"),
                ("JPEG",      "*.jpg *.jpeg"),
                ("TIFF",      "*.tif *.tiff"),
                ("BMP",       "*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._proc_image.save(path)
            self._status.set(f"Exported  →  {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    app = App(root)

    # Allow passing an image path as a command-line argument
    if len(sys.argv) > 1:
        root.after(100, lambda: app._open(sys.argv[1]))

    root.mainloop()


if __name__ == "__main__":
    main()
