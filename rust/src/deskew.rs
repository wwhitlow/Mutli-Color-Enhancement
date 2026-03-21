use image::{GrayImage, RgbImage, imageops};

/// Detect skew angle in degrees via projection-variance method.
/// Tests angles in [-15, +15] at 0.1° steps.
pub fn find_skew_angle(img: &GrayImage) -> f32 {
    // Downsample to at most ~800px on the long side for speed.
    // The projection-variance method is resolution-independent.
    let (orig_w, orig_h) = img.dimensions();
    let max_side = 800u32;
    let scale = if orig_w.max(orig_h) > max_side {
        max_side as f32 / orig_w.max(orig_h) as f32
    } else {
        1.0
    };
    let small: GrayImage = if scale < 1.0 {
        let nw = (orig_w as f32 * scale).round() as u32;
        let nh = (orig_h as f32 * scale).round() as u32;
        imageops::resize(img, nw, nh, imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };

    let (w, h) = small.dimensions();
    let w = w as usize;
    let h = h as usize;

    // Edge magnitude via central-difference Sobel.
    let pix = |x: i32, y: i32| -> f32 {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            0.0
        } else {
            small.get_pixel(x as u32, y as u32).0[0] as f32
        }
    };
    let mag: Vec<f32> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let gx = pix(x as i32 + 1, y as i32) - pix(x as i32 - 1, y as i32);
                let gy = pix(x as i32, y as i32 + 1) - pix(x as i32, y as i32 - 1);
                gx.hypot(gy)
            })
        })
        .collect();

    // Use a relative threshold — 10 % of the max edge magnitude.
    // Avoids the absolute value being wrong for low- or high-contrast images.
    let max_mag = mag.iter().cloned().fold(0f32, f32::max);
    if max_mag < 1.0 {
        return 0.0; // blank image
    }
    let threshold = max_mag * 0.10;

    let mut best_angle = 0f32;
    let mut best_var = -1f32;

    for angle_i in -150i32..=150 {
        let angle = angle_i as f32 * 0.1;
        let rad = angle.to_radians();
        let sin_a = rad.sin();
        let cos_a = rad.cos();

        // The projection of pixel (x,y) onto the "row axis" of the image rotated by
        // `angle` is:  proj = x*sin(a) + y*cos(a)
        //
        // For the range of x ∈ [0,w-1] and y ∈ [0,h-1] the projection spans:
        //   min over 4 corners, max over 4 corners.
        // We keep h bins (one per original row) and discard projections that fall
        // outside [0, h-1].  Corner pixels that escape are a small minority and
        // their exclusion does not bias the angle estimate for |angle| ≤ 15°.
        //
        // Crucially: we do NOT clamp out-of-range values into the boundary bins —
        // that was the original bug causing spurious variance spikes at large angles.

        let bins = h;
        let mut row_sums = vec![0f32; bins];

        for y in 0..h {
            for x in 0..w {
                let m = mag[y * w + x];
                if m < threshold {
                    continue;
                }
                let proj = x as f32 * sin_a + y as f32 * cos_a;
                // Only count pixels whose projection lands within the valid row range.
                if proj >= 0.0 && proj < bins as f32 {
                    let bin = proj.round() as usize;
                    // Guard: round() could produce `bins` for proj just below bins.
                    if bin < bins {
                        row_sums[bin] += m;
                    }
                }
            }
        }

        let mean = row_sums.iter().sum::<f32>() / bins as f32;
        let var = row_sums
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / bins as f32;

        if var > best_var {
            best_var = var;
            best_angle = angle;
        }
    }

    best_angle
}

/// Rotate RGB image by `angle` degrees (bilinear interpolation, white background).
pub fn apply_deskew(img: &RgbImage, angle: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    // Backward mapping: for each output pixel, find its source in the original.
    // To rotate the output by +angle, sample the source at -angle.
    let rad = (-angle).to_radians();
    let cos = rad.cos();
    let sin = rad.sin();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let mut out = RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let sx = cos * dx - sin * dy + cx;
            let sy = sin * dx + cos * dy + cy;
            let px = bilinear(img, sx, sy);
            out.put_pixel(x, y, image::Rgb(px));
        }
    }
    out
}

fn bilinear(img: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let (w, h) = img.dimensions();
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let get = |xi: i32, yi: i32| -> [f32; 3] {
        if xi < 0 || yi < 0 || xi >= w as i32 || yi >= h as i32 {
            return [255.0, 255.0, 255.0];
        }
        let p = img.get_pixel(xi as u32, yi as u32).0;
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    let a = get(x0, y0);
    let b = get(x0 + 1, y0);
    let c = get(x0, y0 + 1);
    let d = get(x0 + 1, y0 + 1);

    let lerp = |i: usize| -> u8 {
        (a[i] * (1.0 - fx) * (1.0 - fy)
            + b[i] * fx * (1.0 - fy)
            + c[i] * (1.0 - fx) * fy
            + d[i] * fx * fy)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    [lerp(0), lerp(1), lerp(2)]
}
