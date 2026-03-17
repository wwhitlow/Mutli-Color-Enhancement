pub const MAX_RGB_DIST: f32 = 441.67; // sqrt(3 * 255^2)
pub const MAX_LAB_DIST: f32 = 383.0;  // sqrt(100^2 + 255^2 + 255^2)

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

/// Convert a single [R,G,B] u8 triple to CIE L*a*b* (D65).
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let to_linear = |c: u8| -> f64 {
        let v = c as f64 / 255.0;
        if v > 0.04045 {
            ((v + 0.055) / 1.055).powf(2.4)
        } else {
            v / 12.92
        }
    };
    let (lr, lg, lb) = (to_linear(r), to_linear(g), to_linear(b));
    let x = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375;
    let y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.0721750;
    let z = lr * 0.0193339 + lg * 0.1191920 + lb * 0.9503041;
    let f = |t: f64| {
        if t > 0.008856 {
            t.cbrt()
        } else {
            (903.3 * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x / 0.95047), f(y / 1.0), f(z / 1.08883));
    [
        (116.0 * fy - 16.0) as f32,
        (500.0 * (fx - fy)) as f32,
        (200.0 * (fy - fz)) as f32,
    ]
}

#[inline]
pub fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
}

pub fn rgb_to_hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}
