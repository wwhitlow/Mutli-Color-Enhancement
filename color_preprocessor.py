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
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import numpy as np              # always CPU numpy — needed for PIL-based deskew

try:
    import cupy as xp          # NVIDIA GPU
except ImportError:
    import numpy as xp
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

# Overlay colors for per-slot mask visualization (R, G, B)
_SLOT_OVERLAY_COLORS = [
    (255,  70,  70),   # red
    ( 70, 210,  70),   # green
    ( 70, 130, 255),   # blue
    (255, 200,   0),   # yellow
    (255,  70, 255),   # magenta
    (  0, 210, 210),   # cyan
    (255, 140,   0),   # orange
    (160,   0, 255),   # purple
]


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
# Core processing (module-level so it can be called from background threads)
# ---------------------------------------------------------------------------

def _process_image(
    image: Image.Image,
    slot_params: list,            # [(sample_rgb, target_rgb, tolerance), ...]
    use_lab: bool,
    edge_protect: float = 0.0,   # 0.0–1.0; reduces tolerance near edges
    smooth_mask: bool = True,     # morphological closing to fill patch holes
    assignment_override: "np.ndarray | None" = None,  # H×W uint8; 255=no override
) -> Image.Image:
    """
    Remap pixels in *image* according to the color slot settings.

    Each active slot defines a (sample_rgb, target_rgb, tolerance) triple.
    For every pixel, the nearest sample color within its slot's tolerance wins
    and the pixel is replaced with that slot's target color.

    Memory strategy: slots are processed one at a time, keeping peak working
    memory at O(P×3) instead of the O(P×N×3) that a fully-vectorised
    broadcast would require.  For a 72 MP image with N=5 slots the old
    approach would allocate ~4.3 GB; this approach uses ~864 MB regardless
    of the number of slots.
    """
    if not slot_params:
        return image.copy()

    arr = xp.array(image, dtype=xp.uint8)
    H, W = arr.shape[:2]
    P = H * W
    flat = arr.reshape(P, 3).astype(xp.float32)   # (P, 3)

    max_dist = _MAX_LAB_DIST if use_lab else _MAX_RGB_DIST

    # Convert all pixels to the working colour space once
    flat_sp = rgb_to_lab(flat) if use_lab else flat   # (P, 3)

    # Pre-compute per-pixel edge strength [0, 1] when edge_protect > 0
    if edge_protect > 0.0:
        lum_np = np.array(image.convert("L"), dtype=np.float32)
        gx_np = np.zeros_like(lum_np)
        gy_np = np.zeros_like(lum_np)
        gx_np[:, 1:-1] = lum_np[:, 2:] - lum_np[:, :-2]
        gy_np[1:-1, :] = lum_np[2:, :] - lum_np[:-2, :]
        mag_np = np.hypot(gx_np, gy_np)
        peak = float(mag_np.max()) or 1.0
        edge_str = xp.array(mag_np / peak, dtype=xp.float32).reshape(P)
    else:
        edge_str = None

    targets = xp.array([p[1] for p in slot_params], dtype=xp.uint8)  # (N, 3)

    best_dist = xp.full(P, xp.inf, dtype=xp.float32)
    best_idx  = xp.full(P, -1,    dtype=xp.int32)

    for i, (sample_rgb, _target_rgb, tolerance) in enumerate(slot_params):
        sample    = xp.array([sample_rgb], dtype=xp.float32)
        sample_sp = rgb_to_lab(sample) if use_lab else sample
        tol_base  = (tolerance / 100.0) * max_dist

        d = xp.sqrt(xp.sum((flat_sp - sample_sp) ** 2, axis=1))        # (P,)

        if edge_str is not None:
            eff_tol = tol_base * (1.0 - edge_protect * edge_str)        # (P,)
            raw_match = d <= eff_tol
        else:
            raw_match = d <= tol_base

        if smooth_mask:
            # Morphological closing: dilate then erode fills interior holes
            closed = _morph_erode1(_morph_dilate1(raw_match.reshape(H, W))).reshape(P)
            better = closed & (d < best_dist)
        else:
            better = raw_match & (d < best_dist)

        best_dist = xp.where(better, d, best_dist)
        best_idx  = xp.where(better, i, best_idx)

    # Apply hand-painted overrides (wins over distance matching)
    if assignment_override is not None:
        ov = xp.array(assignment_override.reshape(P).astype(np.int32))
        has_ov = ov != 255
        best_idx = xp.where(has_ov, ov, best_idx)

    has_match = best_idx >= 0
    result = flat.astype(xp.uint8).copy()
    result[has_match] = targets[best_idx[has_match]]

    # CuPy arrays live on the GPU; PIL needs a CPU numpy array
    cpu = result.get() if hasattr(result, "get") else result
    return Image.fromarray(cpu.reshape(H, W, 3))


# ---------------------------------------------------------------------------
# Deskew utilities (always CPU — operates on PIL images, not GPU arrays)
# ---------------------------------------------------------------------------

def _hv_edge_image(gray: np.ndarray) -> np.ndarray:
    """
    Return a binary edge map that keeps only near-horizontal and near-vertical
    edges, suppressing diagonals and curves.

    Strategy: compute central-difference gradient (gx, gy).  An edge is
    "near-H/V" when the smaller gradient component is less than 40 % of the
    larger — i.e. the gradient direction is within ~22° of horizontal or
    vertical.  Box borders satisfy this perfectly.  Diagonal glyph strokes
    (the slanted arms of Z, N, W, swashes, etc.) are suppressed.

    Magnitude threshold: keep any edge whose strength is ≥ 10 % of the
    maximum gradient in the image.  This removes near-zero noise without
    risk of the percentile-vs-equal-magnitude edge case that occurs when all
    detected edges share the same magnitude (as happens in clean solid-colour
    images after colour processing).
    """
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]   # central difference, X
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]   # central difference, Y

    magnitude = np.hypot(gx, gy)
    max_mag = float(magnitude.max())
    if max_mag < 1.0:
        return ((gray < gray.mean()) * 255).astype(np.uint8)   # blank fallback

    abs_gx, abs_gy = np.abs(gx), np.abs(gy)
    max_g = np.maximum(abs_gx, abs_gy)
    min_g = np.minimum(abs_gx, abs_gy)

    # Passes edges whose gradient is within ~22° of H or V
    hv_mask = min_g < 0.4 * np.maximum(max_g, 1e-6)

    # Keep edges with meaningful strength (≥ 10 % of the image's peak gradient)
    strong = magnitude >= max_mag * 0.10

    result = (strong & hv_mask).astype(np.uint8) * 255
    if not result.any():
        return ((gray < gray.mean()) * 255).astype(np.uint8)   # plain fallback
    return result


def _find_skew_angle(image: Image.Image, max_angle: float = 15.0) -> float:
    """
    Estimate scan skew using a gradient-filtered projection profile.

    Two-stage approach that is robust to intricate internal glyph content:

    1. Gradient + orientation filter — compute per-pixel gradient and keep
       only near-horizontal and near-vertical edges (see _hv_edge_image).
       Box borders score well because they are long, straight, axis-aligned
       lines.  Diagonal strokes, serifs, curves, and swashes score near zero
       and are suppressed before any angle search begins.

    2. Projection-profile variance search — for each candidate angle the
       filtered edge image is rotated and the variance of row-pixel sums is
       measured.  When horizontal box borders are perfectly level they
       concentrate into thin, dense rows → high variance.  The winning angle
       is the one that produces the sharpest row structure.

    Works on a downsampled copy (≤ 1200 px wide); searches ±max_angle in
    0.2° steps.  Returns the correction angle for Image.rotate() (CCW +).
    """
    MAX_SIDE = 1200
    scale = min(1.0, MAX_SIDE / max(image.width, image.height))
    small = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.LANCZOS,
    )
    gray = np.array(small.convert("L"), dtype=np.float32)
    edge_arr = _hv_edge_image(gray)
    edge_img = Image.fromarray(edge_arr)

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 0.01, 0.2):
        rot = edge_img.rotate(float(angle), expand=False, fillcolor=0)
        row_sums = np.array(rot, dtype=np.float64).sum(axis=1)
        score = float(row_sums.var())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def _apply_deskew(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotate image by angle degrees CCW (PIL convention), expanding the canvas
    to fit the full rotated content.  Fill colour is inferred from the four
    corner pixels of the original image.
    """
    if abs(angle) < 0.05:
        return image.copy()
    w, h = image.size
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((w - 1, 0)),
        image.getpixel((0, h - 1)),
        image.getpixel((w - 1, h - 1)),
    ]
    bg = tuple(int(round(float(np.median([c[i] for c in corners])))) for i in range(3))
    return image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=bg)


# ---------------------------------------------------------------------------
# Morphological helpers (used by _process_image; work with numpy or CuPy)
# ---------------------------------------------------------------------------

def _morph_dilate1(mask: xp.ndarray) -> xp.ndarray:
    """4-connected binary dilation by 1 px."""
    out = mask.copy()
    out[1:,  :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:,  :]
    out[:,  1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1: ]
    return out


def _morph_erode1(mask: xp.ndarray) -> xp.ndarray:
    """4-connected binary erosion by 1 px."""
    out = mask.copy()
    out[1:,  :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:,  :]
    out[:,  1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1: ]
    return out


def _compute_slot_assignment(
    image: Image.Image,
    slot_params: list,
    use_lab: bool,
    edge_protect: float = 0.0,
    smooth_mask: bool = True,
) -> np.ndarray:
    """
    Return an H×W uint8 array where each pixel holds the index of the winning
    color slot (0-based), or 255 if no slot matched.
    Uses the same distance / edge-protect / morph-closing logic as _process_image.
    """
    if not slot_params:
        arr = np.array(image)
        return np.full(arr.shape[:2], 255, dtype=np.uint8)

    arr = xp.array(image, dtype=xp.uint8)
    H, W = arr.shape[:2]
    P = H * W
    flat = arr.reshape(P, 3).astype(xp.float32)
    max_dist = _MAX_LAB_DIST if use_lab else _MAX_RGB_DIST
    flat_sp = rgb_to_lab(flat) if use_lab else flat

    if edge_protect > 0.0:
        lum_np = np.array(image.convert("L"), dtype=np.float32)
        gx_np = np.zeros_like(lum_np)
        gy_np = np.zeros_like(lum_np)
        gx_np[:, 1:-1] = lum_np[:, 2:] - lum_np[:, :-2]
        gy_np[1:-1, :] = lum_np[2:, :] - lum_np[:-2, :]
        mag_np = np.hypot(gx_np, gy_np)
        peak = float(mag_np.max()) or 1.0
        edge_str = xp.array(mag_np / peak, dtype=xp.float32).reshape(P)
    else:
        edge_str = None

    best_dist = xp.full(P, xp.inf, dtype=xp.float32)
    best_idx  = xp.full(P, -1,    dtype=xp.int32)

    for i, (sample_rgb, _target_rgb, tolerance) in enumerate(slot_params):
        sample    = xp.array([sample_rgb], dtype=xp.float32)
        sample_sp = rgb_to_lab(sample) if use_lab else sample
        tol_base  = (tolerance / 100.0) * max_dist
        d = xp.sqrt(xp.sum((flat_sp - sample_sp) ** 2, axis=1))

        if edge_str is not None:
            raw_match = d <= tol_base * (1.0 - edge_protect * edge_str)
        else:
            raw_match = d <= tol_base

        if smooth_mask:
            closed = _morph_erode1(_morph_dilate1(raw_match.reshape(H, W))).reshape(P)
            better = closed & (d < best_dist)
        else:
            better = raw_match & (d < best_dist)

        best_dist = xp.where(better, d, best_dist)
        best_idx  = xp.where(better, i, best_idx)

    # Move to CPU; remap -1 → 255
    cpu = best_idx.get() if hasattr(best_idx, "get") else best_idx
    raw = cpu.reshape(H, W).astype(np.int16)
    return np.where(raw < 0, 255, raw).astype(np.uint8)


def _make_mask_overlay(
    orig: Image.Image,
    slot_assignment: np.ndarray,   # H×W uint8: slot index or 255 = unmatched
    slot_count: int,
    alpha: int = 160,              # 0-255 overlay opacity
) -> Image.Image:
    """
    Alpha-blend a per-slot colored overlay onto *orig*.
    Each slot gets a distinct color from _SLOT_OVERLAY_COLORS.
    Unmatched pixels show through at full opacity.
    """
    base = orig.convert("RGBA")
    overlay_arr = np.zeros((*slot_assignment.shape, 4), dtype=np.uint8)
    for i in range(slot_count):
        color = _SLOT_OVERLAY_COLORS[i % len(_SLOT_OVERLAY_COLORS)]
        mask = slot_assignment == i
        overlay_arr[mask, :3] = color
        overlay_arr[mask,  3] = alpha
    overlay_img = Image.fromarray(overlay_arr, "RGBA")
    return Image.alpha_composite(base, overlay_img).convert("RGB")


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
        self._settings_dirty: bool = True   # True → cache is stale, must reprocess
        self._processing: bool = False       # True → background worker is running
        self._preview_btn: ttk.Button | None = None
        self._deskew_btn: ttk.Button | None = None

        self._use_lab = tk.BooleanVar(value=True)
        self._live_prev = tk.BooleanVar(value=False)
        self._mask_mode = tk.BooleanVar(value=False)
        self._slot_assignment: "np.ndarray | None" = None
        self._paint_slot_idx: int = 0
        self._brush_size = tk.IntVar(value=10)
        self._paint_slot_label: ttk.Label | None = None
        self._mask_undo_stack: list = []        # list of (ys, xs, old_vals) diffs
        self._stroke_pre: "np.ndarray | None" = None   # pre-stroke copy for undo
        self._mask_painted: bool = False        # True → painted overrides need reprocess
        self._brush_cursor_id: "int | None" = None     # canvas item id for circle cursor
        self._last_cursor_canvasxy: "tuple | None" = None
        self._disp_orig: "Image.Image | None" = None   # display-res cache of orig
        self._disp_orig_zoom: float = 0.0              # zoom level of cached disp_orig
        self._last_overlay_t: float = 0.0              # monotonic time of last overlay draw
        self._edge_protect_var: tk.IntVar | None = None
        self._smooth_var: tk.BooleanVar | None = None
        self._ep_label: ttk.Label | None = None

        self._build_ui()
        self._bind_keys()

    # ================================================================ UI build

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        # ── Toolbar (two rows) ────────────────────────────────────────────────
        tb = ttk.Frame(self.root)
        tb.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 0))

        # Row 1 — primary workflow actions
        tb1 = ttk.Frame(tb)
        tb1.pack(fill="x", side="top", pady=(0, 2))

        ttk.Button(tb1, text="Open Image",    command=self._open).pack(side="left", padx=2)
        ttk.Separator(tb1, orient="vertical").pack(side="left", fill="y", padx=6)
        self._preview_btn = ttk.Button(tb1, text="Preview", command=self._do_preview)
        self._preview_btn.pack(side="left", padx=2)
        ttk.Button(tb1, text="Show Original", command=self._show_original).pack(side="left", padx=2)
        ttk.Checkbutton(
            tb1, text="Show Mask", variable=self._mask_mode, command=self._on_mask_toggle,
        ).pack(side="left", padx=2)
        ttk.Separator(tb1, orient="vertical").pack(side="left", fill="y", padx=6)
        self._deskew_btn = ttk.Button(tb1, text="Auto-Deskew", command=self._do_deskew)
        self._deskew_btn.pack(side="left", padx=2)
        ttk.Separator(tb1, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(tb1, text="Export…",       command=self._export).pack(side="left", padx=2)
        ttk.Separator(tb1, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(tb1, text="−", width=2, command=self._zoom_out).pack(side="left")
        self._zoom_label = ttk.Label(tb1, text="100%", width=5, anchor="center")
        self._zoom_label.pack(side="left")
        ttk.Button(tb1, text="+", width=2, command=self._zoom_in).pack(side="left")

        # Row 2 — processing settings and paint tools
        tb2 = ttk.Frame(tb)
        tb2.pack(fill="x", side="top", pady=(0, 4))

        ttk.Checkbutton(tb2, text="LAB color space", variable=self._use_lab).pack(side="left", padx=2)
        ttk.Checkbutton(tb2, text="Live preview",    variable=self._live_prev).pack(side="left", padx=2)
        ttk.Separator(tb2, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(tb2, text="Edge protect:").pack(side="left", padx=(0, 2))
        self._edge_protect_var = tk.IntVar(value=0)
        self._ep_label = ttk.Label(tb2, text=" 0%", width=4)
        ttk.Scale(
            tb2, from_=0, to=100, orient="horizontal", length=80,
            variable=self._edge_protect_var,
            command=lambda _: (
                self._ep_label.config(text=f"{self._edge_protect_var.get():2d}%"),
                self._slot_changed(),
            ),
        ).pack(side="left")
        self._ep_label.pack(side="left", padx=(0, 4))
        self._smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            tb2, text="Smooth", variable=self._smooth_var, command=self._slot_changed,
        ).pack(side="left", padx=2)
        ttk.Separator(tb2, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(tb2, text="Brush:").pack(side="left", padx=(0, 2))
        ttk.Scale(
            tb2, from_=1, to=50, orient="horizontal", length=60,
            variable=self._brush_size,
        ).pack(side="left")
        ttk.Button(tb2, text="◀", width=2, command=self._prev_paint_slot).pack(side="left", padx=(6, 0))
        self._paint_slot_label = ttk.Label(tb2, text="—", width=9, anchor="center")
        self._paint_slot_label.pack(side="left")
        ttk.Button(tb2, text="▶", width=2, command=self._next_paint_slot).pack(side="left", padx=(0, 4))

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

        self._canvas.bind("<Button-1>",        self._canvas_click)
        self._canvas.bind("<B1-Motion>",       self._canvas_paint_drag)
        self._canvas.bind("<ButtonRelease-1>", self._canvas_stroke_end)
        self._canvas.bind("<Button-3>",        self._canvas_erase_start)
        self._canvas.bind("<B3-Motion>",       self._canvas_erase_drag)
        self._canvas.bind("<ButtonRelease-3>", self._canvas_stroke_end)
        self._canvas.bind("<Motion>",          self._canvas_hover)
        self._canvas.bind("<MouseWheel>",      self._canvas_scroll_zoom)  # macOS / Win
        self._canvas.bind("<Button-4>",        lambda _e: self._zoom_in())   # Linux scroll up
        self._canvas.bind("<Button-5>",        lambda _e: self._zoom_out())  # Linux scroll down

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
        self.root.bind("<Control-z>", lambda _: self._undo_mask())
        self.root.bind("<Control-Z>", lambda _: self._undo_mask())

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
        self._settings_dirty = True
        self._disp_orig = None          # invalidate display-res cache
        self._disp_orig_zoom = 0.0

        # Auto-select a sensible default zoom for large images
        w, h = self._orig_image.size
        pixels = w * h
        if pixels >= 16_000_000:        # 16 MP+  →  25 %
            self._zoom_idx = 1
        elif pixels >= 4_000_000:       # 4–16 MP →  50 %
            self._zoom_idx = 3
        else:
            self._zoom_idx = self._DEFAULT_ZOOM

        self._render_canvas(self._orig_image)
        name = Path(path).name
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

        # In mask mode, left-click starts a paint stroke
        if self._mask_mode.get():
            if self._slot_assignment is not None:
                self._stroke_pre = self._slot_assignment.copy()
                self._paint_at(ix, iy, self._get_paint_slot_val())
                self._display_mask_overlay()
            return

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
        if self._mask_mode.get():
            self._last_cursor_canvasxy = (self._canvas.canvasx(event.x),
                                          self._canvas.canvasy(event.y))
            self._redraw_brush_cursor()
            ready = [s for s in self._color_slots if s.is_ready()]
            if ready:
                idx = self._paint_slot_idx % len(ready)
                slot = ready[idx]
                self._status.set(
                    f"({ix}, {iy})  —  left-drag: paint Color {slot.index + 1}  |  right-drag: erase  |  brush: {self._brush_size.get()}px"
                )
            return
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
        self._settings_dirty = True
        self._slot_assignment = None          # invalidate mask cache
        self._mask_painted = False
        self._mask_undo_stack.clear()         # painted edits are now meaningless
        if self._live_prev.get() and self._orig_image is not None:
            if self._mask_mode.get():
                self._do_mask()
            else:
                self._do_preview()

    # ================================================================ processing

    def _do_preview(self) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if self._processing:
            return  # already running

        # Cache hit — nothing changed since last process, just (re-)display
        if not self._settings_dirty and not self._mask_painted and self._proc_image is not None:
            self._showing_proc = True
            self._render_canvas(self._proc_image)
            self._status.set("Preview (cached)  —  click 'Show Original' to compare, or Export when done.")
            return

        # Snapshot settings on the main thread before handing off to worker
        use_lab      = self._use_lab.get()
        edge_protect = self._edge_protect_var.get() / 100.0 if self._edge_protect_var else 0.0
        smooth_mask  = self._smooth_var.get() if self._smooth_var else True
        assignment_override = self._slot_assignment  # honour any painted overrides
        slot_params  = [
            (s.sample_rgb, s.target_rgb, s.tolerance)
            for s in self._color_slots if s.is_ready()
        ]
        image = self._orig_image

        self._set_processing(True)
        self._status.set("Processing…")
        self.root.update_idletasks()

        def _worker():
            try:
                result = _process_image(image, slot_params, use_lab,
                                        edge_protect=edge_protect,
                                        smooth_mask=smooth_mask,
                                        assignment_override=assignment_override)
            except Exception as exc:
                self.root.after(0, lambda: self._on_process_error(exc))
                return
            self.root.after(0, lambda: self._on_process_done(result))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_process_done(self, result: Image.Image) -> None:
        self._proc_image = result
        self._settings_dirty = False
        self._mask_painted = False
        self._showing_proc = True
        self._set_processing(False)
        self._mask_mode.set(False)            # switch out of mask mode to show result
        self._render_canvas(result)
        self._status.set("Preview  —  click 'Show Original' to compare, or Export when done.")

    def _on_process_error(self, exc: Exception) -> None:
        self._set_processing(False)
        self._status.set(f"Processing failed: {exc}")
        messagebox.showerror("Processing failed", str(exc))

    # ================================================================ mask

    def _on_mask_toggle(self) -> None:
        if self._mask_mode.get():
            self._canvas.config(cursor="none")
            if self._slot_assignment is None or self._settings_dirty:
                self._do_mask()
            else:
                self._display_mask_overlay(force=True)
        else:
            self._canvas.config(cursor="crosshair")
            self._last_cursor_canvasxy = None
            self._redraw_brush_cursor()
            self._refresh_display()
            self._status.set("Mask mode off.  Preview to see processed result, or Export.")

    def _do_mask(self) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if self._processing:
            return

        use_lab      = self._use_lab.get()
        edge_protect = self._edge_protect_var.get() / 100.0 if self._edge_protect_var else 0.0
        smooth_mask  = self._smooth_var.get() if self._smooth_var else True
        slot_params  = [
            (s.sample_rgb, s.target_rgb, s.tolerance)
            for s in self._color_slots if s.is_ready()
        ]
        image = self._orig_image

        self._set_processing(True)
        self._status.set("Computing mask…")
        self.root.update_idletasks()

        def _worker():
            try:
                assignment = _compute_slot_assignment(
                    image, slot_params, use_lab, edge_protect, smooth_mask
                )
            except Exception as exc:
                self.root.after(0, lambda: self._on_mask_error(exc))
                return
            self.root.after(0, lambda: self._on_mask_done(assignment))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_mask_done(self, assignment: np.ndarray) -> None:
        self._slot_assignment = assignment
        self._set_processing(False)
        self._update_paint_slot_label()
        n_matched = int((assignment != 255).sum())
        pct = 100.0 * n_matched / assignment.size
        self._status.set(
            f"Mask ready — {n_matched:,} px matched ({pct:.1f}%).  "
            "Left-drag to paint  |  right-drag to erase  |  ◀▶ to change slot  |  Preview to apply."
        )
        if self._mask_mode.get():
            self._display_mask_overlay(force=True)

    def _on_mask_error(self, exc: Exception) -> None:
        self._set_processing(False)
        self._status.set(f"Mask computation failed: {exc}")
        messagebox.showerror("Mask failed", str(exc))

    def _display_mask_overlay(self, force: bool = False) -> None:
        if self._orig_image is None or self._slot_assignment is None:
            return
        # Throttle to ~30 fps during rapid paint/drag events
        import time
        now = time.monotonic()
        if not force and (now - self._last_overlay_t) < 0.033:
            return
        self._last_overlay_t = now

        n_slots = sum(1 for s in self._color_slots if s.is_ready())
        z = self._zoom()
        dw = max(1, int(self._orig_image.width  * z))
        dh = max(1, int(self._orig_image.height * z))

        # Reuse cached display-res original; rebuild only when zoom changes
        if self._disp_orig is None or self._disp_orig_zoom != z:
            resample = Image.NEAREST if z >= 1 else Image.LANCZOS
            self._disp_orig = self._orig_image.resize((dw, dh), resample)
            self._disp_orig_zoom = z

        small_assign = np.array(
            Image.fromarray(self._slot_assignment, mode="L").resize((dw, dh), Image.NEAREST)
        )
        overlay = _make_mask_overlay(self._disp_orig, small_assign, n_slots)
        self._tk_image = ImageTk.PhotoImage(overlay)
        self._canvas.configure(scrollregion=(0, 0, dw, dh))
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
        self._zoom_label.config(text=f"{int(z * 100)}%")
        self._redraw_brush_cursor()

    # ================================================================ mask painting

    def _redraw_brush_cursor(self) -> None:
        """Draw a dashed circle showing brush size. Called after every canvas update."""
        if self._brush_cursor_id is not None:
            self._canvas.delete(self._brush_cursor_id)
            self._brush_cursor_id = None
        if not self._mask_mode.get() or self._last_cursor_canvasxy is None:
            return
        cx, cy = self._last_cursor_canvasxy
        r = max(1, self._brush_size.get()) * self._zoom()
        self._brush_cursor_id = self._canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="white", width=1, dash=(4, 4),
        )

    def _get_paint_slot_val(self) -> int:
        """Return the slot-assignment value to write when painting (0-based index, or 255)."""
        ready = [s for s in self._color_slots if s.is_ready()]
        if not ready:
            return 255
        self._paint_slot_idx = self._paint_slot_idx % len(ready)
        return self._paint_slot_idx

    def _prev_paint_slot(self) -> None:
        ready = [s for s in self._color_slots if s.is_ready()]
        if not ready:
            return
        self._paint_slot_idx = (self._paint_slot_idx - 1) % len(ready)
        self._update_paint_slot_label()

    def _next_paint_slot(self) -> None:
        ready = [s for s in self._color_slots if s.is_ready()]
        if not ready:
            return
        self._paint_slot_idx = (self._paint_slot_idx + 1) % len(ready)
        self._update_paint_slot_label()

    def _update_paint_slot_label(self) -> None:
        if self._paint_slot_label is None:
            return
        ready = [s for s in self._color_slots if s.is_ready()]
        if not ready:
            self._paint_slot_label.config(text="—")
            return
        idx   = self._paint_slot_idx % len(ready)
        slot  = ready[idx]
        color = _SLOT_OVERLAY_COLORS[idx % len(_SLOT_OVERLAY_COLORS)]
        hex_c = rgb_to_hex(*color)
        self._paint_slot_label.config(
            text=f"Color {slot.index + 1}",
            foreground=hex_c,
        )

    def _paint_at(self, ix: int, iy: int, slot_val: int) -> None:
        """Write slot_val into a circular brush region of _slot_assignment."""
        if self._slot_assignment is None:
            return
        self._mask_painted = True
        H, W = self._slot_assignment.shape
        r  = max(1, self._brush_size.get())
        y0 = max(0, iy - r);  y1 = min(H, iy + r + 1)
        x0 = max(0, ix - r);  x1 = min(W, ix + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - iy) ** 2 + (xx - ix) ** 2 <= r ** 2
        self._slot_assignment[y0:y1, x0:x1][circle] = slot_val

    def _canvas_paint_drag(self, event) -> None:
        if not self._mask_mode.get() or self._slot_assignment is None:
            return
        self._last_cursor_canvasxy = (self._canvas.canvasx(event.x),
                                      self._canvas.canvasy(event.y))
        ix, iy = self._canvas_xy(event)
        self._paint_at(ix, iy, self._get_paint_slot_val())
        self._display_mask_overlay()

    def _canvas_erase_start(self, event) -> None:
        if not self._mask_mode.get() or self._slot_assignment is None:
            return
        self._stroke_pre = self._slot_assignment.copy()
        self._last_cursor_canvasxy = (self._canvas.canvasx(event.x),
                                      self._canvas.canvasy(event.y))
        ix, iy = self._canvas_xy(event)
        self._paint_at(ix, iy, 255)
        self._display_mask_overlay()

    def _canvas_erase_drag(self, event) -> None:
        if not self._mask_mode.get() or self._slot_assignment is None:
            return
        self._last_cursor_canvasxy = (self._canvas.canvasx(event.x),
                                      self._canvas.canvasy(event.y))
        ix, iy = self._canvas_xy(event)
        self._paint_at(ix, iy, 255)
        self._display_mask_overlay()

    def _canvas_stroke_end(self, event) -> None:
        """Commit the current paint/erase stroke to the undo stack."""
        if not self._mask_mode.get() or self._stroke_pre is None or self._slot_assignment is None:
            self._stroke_pre = None
            return
        changed = self._stroke_pre != self._slot_assignment
        ys, xs = np.where(changed)
        if len(ys):
            old_vals = self._stroke_pre[ys, xs]
            self._mask_undo_stack.append((ys, xs, old_vals))
            if len(self._mask_undo_stack) > 20:
                self._mask_undo_stack.pop(0)
        self._stroke_pre = None
        # Force a final redraw so the complete stroke is visible
        self._display_mask_overlay(force=True)

    def _undo_mask(self) -> None:
        if not self._mask_mode.get() or not self._mask_undo_stack:
            return
        if self._slot_assignment is None:
            return
        ys, xs, old_vals = self._mask_undo_stack.pop()
        self._slot_assignment[ys, xs] = old_vals
        self._display_mask_overlay(force=True)
        n = len(self._mask_undo_stack)
        self._status.set(f"Undo applied.  {n} step{'s' if n != 1 else ''} remaining.")

    # ================================================================ deskew

    def _do_deskew(self) -> None:
        if self._proc_image is None:
            messagebox.showwarning(
                "No preview",
                "Run Preview first to apply colour mapping, then use Auto-Deskew.\n\n"
                "The cleaner post-colour image gives much more accurate angle detection.",
            )
            return
        if self._processing:
            return

        image = self._proc_image   # deskew the colour-processed result
        self._set_processing(True)
        self._status.set("Detecting skew angle on processed image…")
        self.root.update_idletasks()

        def _worker():
            try:
                angle  = _find_skew_angle(image)
                result = _apply_deskew(image, angle)
            except Exception as exc:
                self.root.after(0, lambda: self._on_deskew_error(exc))
                return
            self.root.after(0, lambda: self._on_deskew_done(result, angle))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_deskew_done(self, result: Image.Image, angle: float) -> None:
        if abs(angle) < 0.05:
            self._set_processing(False)
            self._status.set("No significant skew detected (< 0.05°).  Image unchanged.")
            return
        # Replace proc_image with deskewed version.
        # _orig_image and _settings_dirty are intentionally left untouched:
        # the cache now reflects the full pipeline (colour + deskew).
        self._proc_image = result
        self._showing_proc = True
        self._set_processing(False)
        self._render_canvas(result)
        self._status.set(
            f"Deskewed by {angle:+.1f}°  —  Export when done, "
            "or adjust colours → Preview → Auto-Deskew again."
        )

    def _on_deskew_error(self, exc: Exception) -> None:
        self._set_processing(False)
        self._status.set(f"Deskew failed: {exc}")
        messagebox.showerror("Deskew failed", str(exc))

    def _set_processing(self, busy: bool) -> None:
        self._processing = busy
        state = "disabled" if busy else "normal"
        if self._preview_btn:
            self._preview_btn.config(state=state)
        if self._deskew_btn:
            self._deskew_btn.config(state=state)

    def _show_original(self) -> None:
        if self._orig_image is None:
            return
        # Do NOT clear _proc_image — keep the cache intact for the next Preview click
        self._showing_proc = False
        self._mask_mode.set(False)
        self._canvas.config(cursor="crosshair")
        self._last_cursor_canvasxy = None
        self._render_canvas(self._orig_image)
        self._status.set("Showing original image.")

    def _process(self, image: Image.Image) -> Image.Image:
        """Full-resolution process used by Export."""
        use_lab      = self._use_lab.get()
        edge_protect = self._edge_protect_var.get() / 100.0 if self._edge_protect_var else 0.0
        smooth_mask  = self._smooth_var.get() if self._smooth_var else True
        slot_params  = [
            (s.sample_rgb, s.target_rgb, s.tolerance)
            for s in self._color_slots if s.is_ready()
        ]
        return _process_image(image, slot_params, use_lab,
                              edge_protect=edge_protect, smooth_mask=smooth_mask,
                              assignment_override=self._slot_assignment)

    # ================================================================ export

    def _export(self) -> None:
        if self._orig_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return

        # Use the cached full-res result if settings haven't changed since last Preview
        if self._settings_dirty or self._proc_image is None:
            if not messagebox.askyesno(
                "No current preview",
                "Settings have changed since the last preview (or no preview exists).\n"
                "Process and export the full-resolution image now?",
            ):
                return
            self._status.set("Exporting (processing full resolution)…")
            self.root.update_idletasks()
            self._proc_image = self._process(self._orig_image)
            self._settings_dirty = False

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
