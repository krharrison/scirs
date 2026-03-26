//! Volume rendering for Neural Radiance Fields.
//!
//! Implements the discrete approximation of the volume rendering integral
//! (Mildenhall et al. 2020):
//!
//! ```text
//! C(r) = Σ_i  T_i · αᵢ · cᵢ
//!
//! where  αᵢ = 1 − exp(−σᵢ · δᵢ)           (opacity of sample i)
//!        T_i = Π_{j<i} (1 − αⱼ)            (accumulated transmittance)
//!        δᵢ  = t_{i+1} − t_i               (interval length)
//! ```
//!
//! Also provides stratified sampling and hierarchical (importance) sampling.

use super::types::{Ray, SamplePoint, VolumeRenderResult};

// ── LCG for reproducible sampling ─────────────────────────────────────────
const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── Stratified sampling ────────────────────────────────────────────────────

/// Sample `n` points along a ray using stratified (jittered) sampling.
///
/// The range `[near, far]` is divided into `n` equal bins.  A uniform random
/// jitter is applied within each bin so that the samples are spread across the
/// entire interval while avoiding clustering.
///
/// # Returns
///
/// A sorted `Vec<f64>` of `n` `t`-values.
pub fn stratified_sample(ray: &Ray, near: f64, far: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    // Derive a deterministic per-ray seed from origin + direction bytes
    let seed: u64 = ray
        .origin
        .iter()
        .chain(ray.direction.iter())
        .fold(0u64, |acc, &v| acc.wrapping_add(v.to_bits()));
    let mut rng = Lcg::new(seed ^ 0xDEAD_BEEF_CAFE_1234);

    let bin_width = (far - near) / n as f64;
    (0..n)
        .map(|i| {
            let bin_start = near + i as f64 * bin_width;
            bin_start + rng.next_f64() * bin_width
        })
        .collect()
}

// ── Hierarchical (importance) sampling ────────────────────────────────────

/// Draw `n` additional samples using inverse-CDF sampling from a coarse
/// weight distribution.
///
/// Algorithm (Mildenhall et al. 2020 §4):
/// 1. Normalise coarse weights to form a PDF.
/// 2. Build the CDF.
/// 3. Draw `n` uniform samples from \[0,1\] and apply inverse-CDF.
///
/// # Arguments
///
/// * `t_vals`  – coarse t-values (length M, sorted ascending).
/// * `weights` – corresponding α-compositing weights (length M).
/// * `n`       – number of fine samples to draw.
///
/// # Returns
///
/// A sorted `Vec<f64>` of `n` new t-values interleaved in the support of the
/// coarse weight distribution.
pub fn importance_sample(t_vals: &[f64], weights: &[f64], n: usize) -> Vec<f64> {
    if n == 0 || t_vals.is_empty() || weights.is_empty() {
        return Vec::new();
    }
    let m = t_vals.len().min(weights.len());

    // Build normalised PDF (add small ε to avoid all-zero case)
    let eps = 1e-5;
    let weight_sum: f64 = weights[..m].iter().sum::<f64>() + eps * m as f64;
    let pdf: Vec<f64> = weights[..m]
        .iter()
        .map(|&w| (w + eps) / weight_sum)
        .collect();

    // CDF (length m+1, CDF[0] = 0)
    let mut cdf = vec![0.0_f64; m + 1];
    for i in 0..m {
        cdf[i + 1] = cdf[i] + pdf[i];
    }
    // Clamp the last value to exactly 1.0
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    // Stratified uniform samples in [0, 1]
    let step = 1.0 / n as f64;
    let mut samples = Vec::with_capacity(n);
    for j in 0..n {
        let u = (j as f64 + 0.5) * step;

        // Binary search: find largest i s.t. cdf[i] <= u
        let mut lo = 0_usize;
        let mut hi = m;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid + 1] <= u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Clamp to valid index range
        let idx = lo.min(m - 1);

        // Linear interpolation within bin
        let t_lo = t_vals[idx];
        let t_hi = if idx + 1 < t_vals.len() {
            t_vals[idx + 1]
        } else {
            t_lo
        };
        let denom = cdf[idx + 1] - cdf[idx];
        let t = if denom < 1e-12 {
            t_lo
        } else {
            t_lo + (u - cdf[idx]) / denom * (t_hi - t_lo)
        };
        samples.push(t);
    }

    // Returned values are already in ascending order due to stratified uniform
    // samples, but we sort to be safe.
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples
}

// ── Discrete volume rendering integral ────────────────────────────────────

/// Evaluate the discrete volume-rendering integral over a sorted list of samples.
///
/// # Arguments
///
/// * `samples` – sample points along the ray (must be sorted by ascending `t`).
///
/// # Returns
///
/// A [`VolumeRenderResult`] containing rendered colour, expected depth,
/// remaining transmittance, and per-sample weights.
pub fn volume_render(samples: &[SamplePoint]) -> VolumeRenderResult {
    let n = samples.len();
    if n == 0 {
        return VolumeRenderResult {
            color: [0.0; 3],
            depth: 0.0,
            transmittance: 1.0,
            weights: Vec::new(),
        };
    }

    let mut color = [0.0_f64; 3];
    let mut depth = 0.0_f64;
    let mut transmittance = 1.0_f64;
    let mut weights = Vec::with_capacity(n);

    for i in 0..n {
        // Interval length δᵢ = t_{i+1} − t_i  (use a small epsilon for the last sample)
        let delta = if i + 1 < n {
            (samples[i + 1].t - samples[i].t).max(0.0)
        } else {
            // Extrapolate the last interval as the average of the previous intervals
            // or use a small fixed step
            let avg_delta = if n > 1 {
                (samples[n - 1].t - samples[0].t) / (n - 1) as f64
            } else {
                1e-3
            };
            avg_delta.max(1e-6)
        };

        // αᵢ = 1 − exp(−σᵢ · δᵢ)
        let alpha = 1.0 - (-samples[i].density * delta).exp();
        // wᵢ = Tᵢ · αᵢ
        let weight = transmittance * alpha;
        weights.push(weight);

        // Accumulate colour and depth
        for (c, color_val) in color.iter_mut().enumerate() {
            *color_val += weight * samples[i].color[c];
        }
        depth += weight * samples[i].t;

        // Update transmittance: Tᵢ₊₁ = Tᵢ · (1 − αᵢ)
        transmittance *= 1.0 - alpha;
    }

    // Clamp colour to [0,1] to remove floating-point overshoot
    for c in &mut color {
        *c = c.clamp(0.0, 1.0);
    }

    VolumeRenderResult {
        color,
        depth,
        transmittance,
        weights,
    }
}

// ── Ray generation ─────────────────────────────────────────────────────────

/// Generate `H × W` camera rays from a pinhole camera given a
/// camera-to-world transform.
///
/// For pixel `(u, v)` (column, row) the unnormalised ray direction in camera
/// space is `(u - W/2, -(v - H/2), -focal)` (OpenGL/NeRF convention: -Z
/// looks forward, +Y up).  The direction is then rotated by the top-left 3×3
/// of `c2w` and normalised.
///
/// # Arguments
///
/// * `h`     – image height in pixels.
/// * `w`     – image width in pixels.
/// * `focal` – focal length in pixels.
/// * `c2w`   – 4×4 camera-to-world homogeneous transform (row-major).
///
/// # Returns
///
/// A `Vec<Ray>` of length `H × W` in row-major order.  Invalid rays (zero
/// direction after rotation, which should not occur for valid focal lengths)
/// are silently replaced with a forward-pointing ray.
pub fn generate_rays(h: usize, w: usize, focal: f64, c2w: &[[f64; 4]; 4]) -> Vec<Ray> {
    let mut rays = Vec::with_capacity(h * w);

    // Camera origin = translation column of c2w
    let origin = [c2w[0][3], c2w[1][3], c2w[2][3]];

    for row in 0..h {
        for col in 0..w {
            // Direction in camera space (pinhole model)
            let dx = col as f64 - w as f64 / 2.0;
            let dy = -(row as f64 - h as f64 / 2.0); // +Y up
            let dz = -focal; // -Z forward

            // Rotate by c2w rotation (top-left 3×3)
            let dir_world = [
                c2w[0][0] * dx + c2w[0][1] * dy + c2w[0][2] * dz,
                c2w[1][0] * dx + c2w[1][1] * dy + c2w[1][2] * dz,
                c2w[2][0] * dx + c2w[2][1] * dy + c2w[2][2] * dz,
            ];

            let ray = Ray::new(origin, dir_world).unwrap_or(Ray {
                origin,
                direction: [0.0, 0.0, -1.0],
            });
            rays.push(ray);
        }
    }

    rays
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerf::types::SamplePoint;

    fn dummy_ray() -> Ray {
        Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("valid ray")
    }

    // ── Stratified sampling ──────────────────────────────────────────────

    #[test]
    fn test_stratified_sampling_count() {
        let ray = dummy_ray();
        let ts = stratified_sample(&ray, 2.0, 6.0, 64);
        assert_eq!(ts.len(), 64);
    }

    #[test]
    fn test_stratified_sampling_range() {
        let ray = dummy_ray();
        let near = 1.0;
        let far = 5.0;
        let ts = stratified_sample(&ray, near, far, 128);
        for &t in &ts {
            assert!(t >= near && t <= far, "t={t} out of [{near}, {far}]");
        }
    }

    #[test]
    fn test_stratified_sampling_ordered() {
        // Each bin starts further out than the previous, so values should be
        // monotonically non-decreasing when samples are generated in order.
        let ray = dummy_ray();
        let ts = stratified_sample(&ray, 2.0, 6.0, 64);
        // Check that the bin_start values (ignoring jitter) are increasing.
        // We verify this via the rough fact that ts[i] should be in bin i.
        let near = 2.0_f64;
        let far = 6.0_f64;
        let n = 64_usize;
        let bin_width = (far - near) / n as f64;
        for (i, &t) in ts.iter().enumerate() {
            let bin_lo = near + i as f64 * bin_width;
            let bin_hi = bin_lo + bin_width;
            assert!(
                t >= bin_lo && t <= bin_hi,
                "sample {i}: t={t} not in [{bin_lo}, {bin_hi}]"
            );
        }
    }

    // ── Importance sampling ──────────────────────────────────────────────

    #[test]
    fn test_importance_sampling_count() {
        let t_vals: Vec<f64> = (0..64).map(|i| 2.0 + i as f64 * 4.0 / 63.0).collect();
        let weights: Vec<f64> = t_vals
            .iter()
            .map(|&t| (-((t - 4.0).powi(2))).exp())
            .collect();
        let fine = importance_sample(&t_vals, &weights, 128);
        assert_eq!(fine.len(), 128);
    }

    // ── Volume rendering ─────────────────────────────────────────────────

    #[test]
    fn test_volume_render_color_range() {
        let samples: Vec<SamplePoint> = (0..64)
            .map(|i| {
                let t = 2.0 + i as f64 * 4.0 / 63.0;
                SamplePoint::new([0.0, 0.0, t], t, 1.0, [0.8, 0.5, 0.2])
            })
            .collect();
        let result = volume_render(&samples);
        for &c in result.color.iter() {
            assert!((0.0..=1.0).contains(&c), "color channel {c} out of [0,1]");
        }
    }

    #[test]
    fn test_volume_render_transmittance() {
        // Highly opaque scene: transmittance should approach 0.
        let samples: Vec<SamplePoint> = (0..64)
            .map(|i| {
                let t = 2.0 + i as f64 * 0.1;
                SamplePoint::new([0.0, 0.0, t], t, 1000.0, [1.0, 1.0, 1.0])
            })
            .collect();
        let result = volume_render(&samples);
        assert!(
            result.transmittance < 0.01,
            "transmittance should be near 0 for opaque scene, got {}",
            result.transmittance
        );
    }

    #[test]
    fn test_volume_render_empty_scene() {
        // Near-zero density → colour should be near zero.
        let samples: Vec<SamplePoint> = (0..32)
            .map(|i| {
                let t = 2.0 + i as f64 * 0.1;
                SamplePoint::new([0.0, 0.0, t], t, 1e-9, [1.0, 1.0, 1.0])
            })
            .collect();
        let result = volume_render(&samples);
        for &c in result.color.iter() {
            assert!(c < 1e-3, "empty scene colour should be near 0, got {c}");
        }
    }

    #[test]
    fn test_volume_render_depth() {
        let near = 2.0_f64;
        let far = 6.0_f64;
        let n = 64_usize;
        let samples: Vec<SamplePoint> = (0..n)
            .map(|i| {
                let t = near + i as f64 * (far - near) / (n - 1) as f64;
                SamplePoint::new([0.0, 0.0, t], t, 5.0, [0.5, 0.5, 0.5])
            })
            .collect();
        let result = volume_render(&samples);
        // Depth must lie in [near, far] since all samples are in that range.
        assert!(
            result.depth >= near && result.depth <= far,
            "depth {} outside [{near}, {far}]",
            result.depth
        );
    }

    // ── Ray generation ───────────────────────────────────────────────────

    #[test]
    fn test_ray_generation_count() {
        let c2w = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let h = 8;
        let w = 10;
        let rays = generate_rays(h, w, 100.0, &c2w);
        assert_eq!(rays.len(), h * w);
    }

    #[test]
    fn test_ray_direction_normalized() {
        let c2w = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let rays = generate_rays(4, 4, 50.0, &c2w);
        for (i, ray) in rays.iter().enumerate() {
            let d = &ray.direction;
            let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-12,
                "ray {i} direction magnitude = {mag}, expected 1.0"
            );
        }
    }
}
