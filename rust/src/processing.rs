use rayon::prelude::*;
use wide::{f32x8, CmpLe};

use crate::color::{
    dist3_sq, rgb_to_lab, MAX_LAB_DIST, MAX_RGB_DIST,
    SLOT_OVERLAY_COLORS,
};

#[derive(Clone, Debug)]
pub struct SlotParams {
    pub sample_rgb: [u8; 3],
    pub target_rgb: [u8; 3],
    pub tolerance: f32, // 0.0–100.0
}

// ---------------------------------------------------------------------------
// Morphological ops — parallel 4-connected dilate / erode
// ---------------------------------------------------------------------------

fn dilate(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    (0..h * w)
        .into_par_iter()
        .map(|i| {
            let x = i % w;
            let y = i / w;
            mask[i]
                || (x > 0 && mask[i - 1])
                || (x < w - 1 && mask[i + 1])
                || (y > 0 && mask[i - w])
                || (y < h - 1 && mask[i + w])
        })
        .collect()
}

fn erode(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    (0..h * w)
        .into_par_iter()
        .map(|i| {
            let x = i % w;
            let y = i / w;
            mask[i]
                && (x == 0 || mask[i - 1])
                && (x == w - 1 || mask[i + 1])
                && (y == 0 || mask[i - w])
                && (y == h - 1 || mask[i + w])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SIMD distance / match  (8-wide f32)
// ---------------------------------------------------------------------------

/// Compute raw match mask for one slot using SIMD (8-wide).
/// Works on structure-of-arrays (ls, as_, bs_) to maximise SIMD lane utilisation.
/// Compares *squared* distances against *squared* tolerance.
fn compute_raw_match_simd(
    ls: &[f32],
    as_: &[f32],
    bs_: &[f32],
    sample: [f32; 3],
    tol_base_sq: f32,
    edge_protect: f32,
    edge_str: &[f32], // empty slice when edge_protect == 0
) -> Vec<bool> {
    let n = ls.len();
    const CHUNK: usize = 8192; // ~32 KB per channel slice → fits L1
    let mut out = vec![false; n];

    let sl = f32x8::splat(sample[0]);
    let sa = f32x8::splat(sample[1]);
    let sb = f32x8::splat(sample[2]);
    let tb = f32x8::splat(tol_base_sq);
    let use_edge = edge_protect > 0.0;
    let ep = f32x8::splat(edge_protect);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);

    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK;
            let len = out_chunk.len();
            let ls_c = &ls[start..start + len];
            let as_c = &as_[start..start + len];
            let bs_c = &bs_[start..start + len];

            let mut i = 0usize;

            if use_edge {
                let es_c = &edge_str[start..start + len];
                while i + 8 <= len {
                    let l = f32x8::from([
                        ls_c[i], ls_c[i+1], ls_c[i+2], ls_c[i+3],
                        ls_c[i+4], ls_c[i+5], ls_c[i+6], ls_c[i+7],
                    ]);
                    let a = f32x8::from([
                        as_c[i], as_c[i+1], as_c[i+2], as_c[i+3],
                        as_c[i+4], as_c[i+5], as_c[i+6], as_c[i+7],
                    ]);
                    let b = f32x8::from([
                        bs_c[i], bs_c[i+1], bs_c[i+2], bs_c[i+3],
                        bs_c[i+4], bs_c[i+5], bs_c[i+6], bs_c[i+7],
                    ]);
                    let e = f32x8::from([
                        es_c[i], es_c[i+1], es_c[i+2], es_c[i+3],
                        es_c[i+4], es_c[i+5], es_c[i+6], es_c[i+7],
                    ]);
                    let dl = l - sl;
                    let da = a - sa;
                    let db = b - sb;
                    let d2 = dl * dl + da * da + db * db;
                    let scale = (one - ep * e).max(zero);
                    let tol_sq = tb * scale * scale;
                    let cmp: [f32; 8] = f32x8::cmp_le(d2, tol_sq).into();
                    for j in 0..8 {
                        out_chunk[i + j] = cmp[j].to_bits() != 0;
                    }
                    i += 8;
                }
                // scalar tail
                while i < len {
                    let dl = ls_c[i] - sample[0];
                    let da = as_c[i] - sample[1];
                    let db = bs_c[i] - sample[2];
                    let d2 = dl * dl + da * da + db * db;
                    let scale = (1.0 - edge_protect * es_c[i]).max(0.0);
                    out_chunk[i] = d2 <= tol_base_sq * scale * scale;
                    i += 1;
                }
            } else {
                while i + 8 <= len {
                    let l = f32x8::from([
                        ls_c[i], ls_c[i+1], ls_c[i+2], ls_c[i+3],
                        ls_c[i+4], ls_c[i+5], ls_c[i+6], ls_c[i+7],
                    ]);
                    let a = f32x8::from([
                        as_c[i], as_c[i+1], as_c[i+2], as_c[i+3],
                        as_c[i+4], as_c[i+5], as_c[i+6], as_c[i+7],
                    ]);
                    let b = f32x8::from([
                        bs_c[i], bs_c[i+1], bs_c[i+2], bs_c[i+3],
                        bs_c[i+4], bs_c[i+5], bs_c[i+6], bs_c[i+7],
                    ]);
                    let dl = l - sl;
                    let da = a - sa;
                    let db = b - sb;
                    let d2 = dl * dl + da * da + db * db;
                    let cmp: [f32; 8] = f32x8::cmp_le(d2, tb).into();
                    for j in 0..8 {
                        out_chunk[i + j] = cmp[j].to_bits() != 0;
                    }
                    i += 8;
                }
                while i < len {
                    let dl = ls_c[i] - sample[0];
                    let da = as_c[i] - sample[1];
                    let db = bs_c[i] - sample[2];
                    let d2 = dl * dl + da * da + db * db;
                    out_chunk[i] = d2 <= tol_base_sq;
                    i += 1;
                }
            }
        });

    out
}

// ---------------------------------------------------------------------------
// Core pipeline
// ---------------------------------------------------------------------------

/// Compute per-pixel slot assignment.
/// Returns Vec<u8> of length w*h; value = slot index (0-based) or 255 = unmatched.
pub fn compute_slot_assignment(
    pixels: &[[u8; 3]],
    w: usize,
    h: usize,
    slots: &[SlotParams],
    use_lab: bool,
    edge_protect: f32,
    smooth_mask: bool,
) -> Vec<u8> {
    let p = w * h;
    if slots.is_empty() {
        return vec![255u8; p];
    }

    // ---- 1. Convert pixels to working color space (parallel) ----
    let flat_sp: Vec<[f32; 3]> = pixels
        .par_iter()
        .map(|px| {
            if use_lab {
                rgb_to_lab(px[0], px[1], px[2])
            } else {
                [px[0] as f32, px[1] as f32, px[2] as f32]
            }
        })
        .collect();

    // ---- 2. Build structure-of-arrays for SIMD (parallel) ----
    let (ls, as_, bs_): (Vec<f32>, Vec<f32>, Vec<f32>) = {
        let (mut ls, mut as_, mut bs_) = (
            Vec::with_capacity(p),
            Vec::with_capacity(p),
            Vec::with_capacity(p),
        );
        for sp in &flat_sp {
            ls.push(sp[0]);
            as_.push(sp[1]);
            bs_.push(sp[2]);
        }
        (ls, as_, bs_)
    };

    // ---- 3. Edge strength (parallel gradient magnitude, 0..1) ----
    let edge_str: Vec<f32> = if edge_protect > 0.0 {
        let lum: Vec<f32> = pixels
            .par_iter()
            .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
            .collect();
        let raw: Vec<f32> = (0..p)
            .into_par_iter()
            .map(|i| {
                let x = i % w;
                let y = i / w;
                let gx = if x > 0 && x < w - 1 { lum[i + 1] - lum[i - 1] } else { 0.0 };
                let gy = if y > 0 && y < h - 1 { lum[i + w] - lum[i - w] } else { 0.0 };
                gx.hypot(gy)
            })
            .collect();
        let peak = raw.par_iter().cloned().reduce(|| 0f32, f32::max).max(1.0);
        raw.into_par_iter().map(|v| v / peak).collect()
    } else {
        Vec::new() // zero-alloc when unused
    };

    // ---- 4. Per-slot: SIMD match + morphological close (parallel across slots) ----
    let per_slot: Vec<(usize, Vec<bool>, [f32; 3])> = slots
        .par_iter()
        .enumerate()
        .map(|(si, slot)| {
            let sample_sp = if use_lab {
                rgb_to_lab(slot.sample_rgb[0], slot.sample_rgb[1], slot.sample_rgb[2])
            } else {
                [
                    slot.sample_rgb[0] as f32,
                    slot.sample_rgb[1] as f32,
                    slot.sample_rgb[2] as f32,
                ]
            };
            // tol_base is in *linear* distance; square it for squared-dist comparison
            let tol_base = (slot.tolerance / 100.0)
                * if use_lab { MAX_LAB_DIST } else { MAX_RGB_DIST };
            let tol_base_sq = tol_base * tol_base;

            let raw_match = compute_raw_match_simd(
                &ls, &as_, &bs_,
                sample_sp,
                tol_base_sq,
                edge_protect,
                &edge_str,
            );

            let matched = if smooth_mask {
                erode(&dilate(&raw_match, w, h), w, h)
            } else {
                raw_match
            };

            (si, matched, sample_sp)
        })
        .collect();

    // ---- 5. Parallel per-pixel merge: find best-matching slot ----
    (0..p)
        .into_par_iter()
        .map(|i| {
            let mut best_d = f32::MAX;
            let mut best_si = 255u8;
            for (si, matched, sample_sp) in &per_slot {
                if matched[i] {
                    let sp = [ls[i], as_[i], bs_[i]];
                    let d = dist3_sq(&sp, sample_sp);
                    if d < best_d {
                        best_d = d;
                        best_si = *si as u8;
                    }
                }
            }
            best_si
        })
        .collect()
}

/// Remap pixel colors according to slot assignments.
/// If `assignment_override` is Some, it takes precedence over recomputing.
pub fn process_image(
    pixels: &[[u8; 3]],
    w: usize,
    h: usize,
    slots: &[SlotParams],
    use_lab: bool,
    edge_protect: f32,
    smooth_mask: bool,
    assignment_override: Option<&[u8]>,
) -> Vec<[u8; 3]> {
    let assignment = match assignment_override {
        Some(a) => a.to_vec(),
        None => compute_slot_assignment(pixels, w, h, slots, use_lab, edge_protect, smooth_mask),
    };

    pixels
        .par_iter()
        .enumerate()
        .map(|(i, &px)| {
            let idx = assignment[i] as usize;
            if idx < slots.len() {
                slots[idx].target_rgb
            } else {
                px
            }
        })
        .collect()
}

/// Blend per-slot color overlays onto original pixels.
pub fn make_mask_overlay(
    orig: &[[u8; 3]],
    assignment: &[u8],
    n_slots: usize,
    alpha: f32,
) -> Vec<[u8; 3]> {
    orig.par_iter()
        .enumerate()
        .map(|(i, &px)| {
            let idx = assignment[i] as usize;
            if idx < n_slots {
                let c = SLOT_OVERLAY_COLORS[idx % SLOT_OVERLAY_COLORS.len()];
                let inv = 1.0 - alpha;
                [
                    (px[0] as f32 * inv + c[0] as f32 * alpha).round() as u8,
                    (px[1] as f32 * inv + c[1] as f32 * alpha).round() as u8,
                    (px[2] as f32 * inv + c[2] as f32 * alpha).round() as u8,
                ]
            } else {
                px
            }
        })
        .collect()
}

