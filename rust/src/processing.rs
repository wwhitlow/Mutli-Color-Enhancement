use rayon::prelude::*;

use crate::color::{dist3, rgb_to_lab, MAX_LAB_DIST, MAX_RGB_DIST, SLOT_OVERLAY_COLORS};

#[derive(Clone, Debug)]
pub struct SlotParams {
    pub sample_rgb: [u8; 3],
    pub target_rgb: [u8; 3],
    pub tolerance: f32, // 0.0–100.0
}

/// 4-connected morphological dilation on a flat bool array (row-major, height×width).
fn dilate(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    (0..h * w)
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

/// 4-connected morphological erosion.
fn erode(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    (0..h * w)
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

/// Compute per-pixel slot assignment.
/// Returns a Vec<u8> of length w*h where each value is the winning slot index (0-based)
/// or 255 if no slot matched.
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

    let max_dist = if use_lab { MAX_LAB_DIST } else { MAX_RGB_DIST };

    // Convert all pixels to working color space
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

    // Edge strength (normalised gradient magnitude, 0..1)
    let edge_str: Vec<f32> = if edge_protect > 0.0 {
        let lum: Vec<f32> = pixels
            .iter()
            .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
            .collect();
        let raw: Vec<f32> = (0..p)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                let gx = if x > 0 && x < w - 1 {
                    lum[i + 1] - lum[i - 1]
                } else {
                    0.0
                };
                let gy = if y > 0 && y < h - 1 {
                    lum[i + w] - lum[i - w]
                } else {
                    0.0
                };
                gx.hypot(gy)
            })
            .collect();
        let peak = raw.iter().cloned().fold(0f32, f32::max).max(1.0);
        raw.iter().map(|v| v / peak).collect()
    } else {
        vec![0.0f32; p]
    };

    let mut best_dist = vec![f32::MAX; p];
    let mut best_idx = vec![-1i32; p];

    for (si, slot) in slots.iter().enumerate() {
        let sample_sp = if use_lab {
            rgb_to_lab(slot.sample_rgb[0], slot.sample_rgb[1], slot.sample_rgb[2])
        } else {
            [
                slot.sample_rgb[0] as f32,
                slot.sample_rgb[1] as f32,
                slot.sample_rgb[2] as f32,
            ]
        };
        let tol_base = (slot.tolerance / 100.0) * max_dist;

        // Compute raw match mask in parallel
        let raw_match: Vec<bool> = flat_sp
            .par_iter()
            .enumerate()
            .map(|(i, sp)| {
                let d = dist3(sp, &sample_sp);
                let tol = if edge_protect > 0.0 {
                    tol_base * (1.0 - edge_protect * edge_str[i]).max(0.0)
                } else {
                    tol_base
                };
                d <= tol
            })
            .collect();

        let matched = if smooth_mask {
            erode(&dilate(&raw_match, w, h), w, h)
        } else {
            raw_match
        };

        for i in 0..p {
            if matched[i] {
                let d = dist3(&flat_sp[i], &sample_sp);
                if d < best_dist[i] {
                    best_dist[i] = d;
                    best_idx[i] = si as i32;
                }
            }
        }
    }

    best_idx
        .iter()
        .map(|&v| if v < 0 { 255 } else { v as u8 })
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
/// `alpha` is 0.0–1.0 overlay opacity. Unmatched pixels (255) pass through unchanged.
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
                [
                    (px[0] as f32 * (1.0 - alpha) + c[0] as f32 * alpha)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                    (px[1] as f32 * (1.0 - alpha) + c[1] as f32 * alpha)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                    (px[2] as f32 * (1.0 - alpha) + c[2] as f32 * alpha)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                ]
            } else {
                px
            }
        })
        .collect()
}
