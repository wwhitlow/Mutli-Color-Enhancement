pub const MAX_RGB_DIST: f32 = 441.67;
pub const MAX_LAB_DIST: f32 = 383.0;

pub const SLOT_OVERLAY_COLORS: [[u8; 3]; 8] = [
    [255,  70,  70],
    [ 70, 210,  70],
    [ 70, 130, 255],
    [255, 200,   0],
    [255,  70, 255],
    [  0, 210, 210],
    [255, 140,   0],
    [160,   0, 255],
];

/// Convert a single [R,G,B] u8 triple to CIE L*a*b* (D65), pure f32.
#[inline]
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let to_linear = |c: u8| -> f32 {
        let v = c as f32 / 255.0;
        if v > 0.04045 {
            ((v + 0.055) / 1.055).powf(2.4)
        } else {
            v / 12.92
        }
    };
    let (lr, lg, lb) = (to_linear(r), to_linear(g), to_linear(b));
    let x = lr * 0.412_456_4 + lg * 0.357_576_1 + lb * 0.180_437_5;
    let y = lr * 0.212_672_9 + lg * 0.715_152_2 + lb * 0.072_175_0;
    let z = lr * 0.019_333_9 + lg * 0.119_192_0 + lb * 0.950_304_1;
    let f = |t: f32| {
        if t > 0.008_856 {
            t.cbrt()
        } else {
            (903.3 * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x / 0.950_47), f(y), f(z / 1.088_83));
    [
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    ]
}

/// Squared Euclidean distance — avoids sqrt in hot comparison loops.
#[inline(always)]
pub fn dist3_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

/// Regular Euclidean distance (used where the actual value is needed).
#[inline]
pub fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    dist3_sq(a, b).sqrt()
}

pub fn rgb_to_hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}
