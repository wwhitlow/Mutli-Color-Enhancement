use crate::color::dist3;

/// K-means color suggestion (Lloyd's algorithm).
/// Subsamples ≤10K pixels for speed.
pub fn kmeans_suggest(pixels: &[[u8; 3]], k: usize, iters: usize) -> Vec<[u8; 3]> {
    if pixels.is_empty() || k == 0 {
        return vec![];
    }
    let step = ((pixels.len() / 10_000) + 1).max(1);
    let sample: Vec<[f32; 3]> = pixels
        .iter()
        .step_by(step)
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    if sample.is_empty() {
        return vec![];
    }

    // Init with farthest-first seeding
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    centroids.push(sample[0]);
    for _ in 1..k {
        let farthest = sample
            .iter()
            .max_by(|a, b| {
                let da = centroids
                    .iter()
                    .map(|c| dist3(a, c))
                    .fold(f32::MAX, f32::min);
                let db = centroids
                    .iter()
                    .map(|c| dist3(b, c))
                    .fold(f32::MAX, f32::min);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        centroids.push(*farthest);
    }

    // Lloyd iterations
    for _ in 0..iters {
        let mut sums = vec![[0f64; 3]; k];
        let mut counts = vec![0usize; k];
        for p in &sample {
            let ci = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    dist3(p, a)
                        .partial_cmp(&dist3(p, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            sums[ci][0] += p[0] as f64;
            sums[ci][1] += p[1] as f64;
            sums[ci][2] += p[2] as f64;
            counts[ci] += 1;
        }
        for (i, c) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                *c = [
                    (sums[i][0] / counts[i] as f64) as f32,
                    (sums[i][1] / counts[i] as f64) as f32,
                    (sums[i][2] / counts[i] as f64) as f32,
                ];
            }
        }
    }

    centroids
        .iter()
        .map(|c| {
            [
                c[0].round().clamp(0.0, 255.0) as u8,
                c[1].round().clamp(0.0, 255.0) as u8,
                c[2].round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}
