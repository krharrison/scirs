//! Level set methods for image segmentation.
//!
//! Implements:
//! - Chan-Vese active contour without edges.
//! - Geodesic active contour (Caselles et al., 1997).

use crate::error::{NdimageError, NdimageResult};
use std::f64::consts::PI;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the Chan-Vese level set evolution.
#[derive(Debug, Clone)]
pub struct LevelSetConfig {
    /// Maximum number of iterations.
    pub iterations: usize,
    /// Time step Δt for explicit Euler update.
    pub dt: f64,
    /// Weight of the length regularisation term (µ).
    pub mu: f64,
    /// Area term weight (ν): positive → shrink, negative → expand.
    pub nu: f64,
    /// Data fidelity weight for region inside φ > 0 (λ₁).
    pub lambda1: f64,
    /// Data fidelity weight for region outside φ < 0 (λ₂).
    pub lambda2: f64,
    /// Smoothing parameter ε for the regularised Heaviside and Dirac functions.
    pub epsilon: f64,
}

impl Default for LevelSetConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            dt: 0.5,
            mu: 1.0,
            nu: 0.0,
            lambda1: 1.0,
            lambda2: 1.0,
            epsilon: 1.0,
        }
    }
}

// ─── Smoothed Heaviside and Dirac delta ──────────────────────────────────────

/// Regularised Heaviside function.
/// `H_ε(z) = ½ · (1 + 2/π · arctan(z/ε))`
#[inline]
fn heaviside(z: f64, eps: f64) -> f64 {
    0.5 * (1.0 + 2.0 / PI * (z / eps).atan())
}

/// Regularised Dirac delta function.
/// `δ_ε(z) = ε / (π · (ε² + z²))`
#[inline]
fn dirac(z: f64, eps: f64) -> f64 {
    eps / (PI * (eps * eps + z * z))
}

// ─── Chan-Vese ────────────────────────────────────────────────────────────────

/// Chan-Vese active contour without edges.
///
/// Evolves the level set function `φ` according to:
///
/// `∂φ/∂t = δ_ε(φ) · [µ · div(∇φ/|∇φ|) - ν - λ₁(u - c₁)² + λ₂(u - c₂)²]`
///
/// where `c₁` = mean intensity inside contour (φ > 0),
/// `c₂` = mean intensity outside contour (φ < 0).
///
/// # Arguments
/// * `image` – 2-D grayscale image (normalised to [0, 1] is recommended)
/// * `phi0`  – initial level set function (same shape as `image`)
/// * `config` – evolution parameters
///
/// # Returns
/// Evolved level set function φ.
///
/// # Errors
/// Returns `DimensionError` if `image` and `phi0` differ in shape.
pub fn chan_vese(
    image: &[Vec<f64>],
    phi0: &[Vec<f64>],
    config: &LevelSetConfig,
) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("image must not be empty".into()));
    }
    let cols = image[0].len();
    if phi0.len() != rows || phi0.iter().any(|r| r.len() != cols)
        || image.iter().any(|r| r.len() != cols)
    {
        return Err(NdimageError::DimensionError(
            "image and phi0 must have the same shape".into(),
        ));
    }

    let mut phi: Vec<Vec<f64>> = phi0.iter().map(|r| r.to_vec()).collect();
    let eps = config.epsilon;

    for _iter in 0..config.iterations {
        // ── Compute means c1, c2 ──
        let mut sum_in = 0.0f64;
        let mut cnt_in = 0.0f64;
        let mut sum_out = 0.0f64;
        let mut cnt_out = 0.0f64;

        for r in 0..rows {
            for c in 0..cols {
                let h = heaviside(phi[r][c], eps);
                sum_in  += image[r][c] * h;
                cnt_in  += h;
                sum_out += image[r][c] * (1.0 - h);
                cnt_out += 1.0 - h;
            }
        }

        let c1 = if cnt_in  > 1e-12 { sum_in  / cnt_in  } else { 0.0 };
        let c2 = if cnt_out > 1e-12 { sum_out / cnt_out } else { 0.0 };

        // ── Update step ──
        let mut new_phi = phi.clone();
        for r in 0..rows {
            for c in 0..cols {
                let p = phi[r][c];
                let d = dirac(p, eps);

                // Curvature term: div(∇φ/|∇φ|)
                let kappa = compute_curvature(&phi, r, c, rows, cols);

                // Data terms
                let diff1 = image[r][c] - c1;
                let diff2 = image[r][c] - c2;
                let data = -config.lambda1 * diff1 * diff1
                    + config.lambda2 * diff2 * diff2;

                // Update
                new_phi[r][c] = p + config.dt * d * (config.mu * kappa - config.nu + data);
            }
        }

        phi = new_phi;
    }

    Ok(phi)
}

/// Compute the mean curvature (κ = div(∇φ/|∇φ|)) at pixel (r, c).
///
/// Uses a central-difference discretisation with reflecting boundary conditions.
fn compute_curvature(phi: &[Vec<f64>], r: usize, c: usize, rows: usize, cols: usize) -> f64 {
    let get = |ri: isize, ci: isize| -> f64 {
        let rr = ri.max(0).min(rows as isize - 1) as usize;
        let cc = ci.max(0).min(cols as isize - 1) as usize;
        phi[rr][cc]
    };

    let ri = r as isize;
    let ci = c as isize;

    let p   = get(ri,     ci);
    let p_n = get(ri - 1, ci);
    let p_s = get(ri + 1, ci);
    let p_e = get(ri,     ci + 1);
    let p_w = get(ri,     ci - 1);
    let p_ne = get(ri - 1, ci + 1);
    let p_nw = get(ri - 1, ci - 1);
    let p_se = get(ri + 1, ci + 1);
    let p_sw = get(ri + 1, ci - 1);

    // Central differences.
    let dx  = (p_e - p_w)  * 0.5;
    let dy  = (p_s - p_n)  * 0.5;
    let dxx = p_e - 2.0 * p + p_w;
    let dyy = p_s - 2.0 * p + p_n;
    let dxy = (p_se - p_sw - p_ne + p_nw) * 0.25;

    let grad_sq = dx * dx + dy * dy;
    let norm = (grad_sq + 1e-10).sqrt();

    // Numerator: dxx*(dy^2) - 2*dxy*dx*dy + dyy*(dx^2)
    let num = dxx * (dy * dy) - 2.0 * dxy * dx * dy + dyy * (dx * dx);
    let denom = (grad_sq + 1e-10).powf(1.5);

    let div_unit = num / denom;

    // Alternative simpler approximation when gradient is very small.
    let laplacian = dxx + dyy;
    let simple = laplacian / norm;

    // Blend: use full formula when gradient is strong, simple otherwise.
    let alpha = (grad_sq / (grad_sq + 1.0)).min(1.0);
    alpha * div_unit + (1.0 - alpha) * simple
}

// ─── Geodesic active contour ──────────────────────────────────────────────────

/// Geodesic active contour (Caselles et al., 1997).
///
/// Evolves the level set according to:
/// `∂φ/∂t = g(x) · κ · |∇φ| + α · g(x) · |∇φ| + β · (∇g · ∇φ)`
///
/// where `g(x) = 1 / (1 + |∇(G_σ * I)|²)` is the edge indicator.
///
/// # Arguments
/// * `image`      – 2-D grayscale image
/// * `phi0`       – initial level set function
/// * `sigma`      – Gaussian blur sigma for gradient computation in `g`
/// * `alpha`      – balloon force weight (positive = expand)
/// * `beta`       – advection term weight
/// * `iterations` – number of evolution steps
///
/// # Returns
/// Evolved level set function φ.
///
/// # Errors
/// Returns `DimensionError` if shapes do not match.
pub fn geodesic_active_contour(
    image: &[Vec<f64>],
    phi0: &[Vec<f64>],
    sigma: f64,
    alpha: f64,
    beta: f64,
    iterations: usize,
) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("image must not be empty".into()));
    }
    let cols = image[0].len();
    if phi0.len() != rows || phi0.iter().any(|r| r.len() != cols)
        || image.iter().any(|r| r.len() != cols)
    {
        return Err(NdimageError::DimensionError(
            "image and phi0 must have the same shape".into(),
        ));
    }

    // Compute edge indicator g(x) = 1 / (1 + |∇(G_σ * I)|²).
    let smoothed = gaussian_blur(image, sigma)?;
    let (gx, gy) = gradient(&smoothed, rows, cols);

    let g: Vec<Vec<f64>> = (0..rows).map(|r| {
        (0..cols).map(|c| {
            let grad_sq = gx[r][c] * gx[r][c] + gy[r][c] * gy[r][c];
            1.0 / (1.0 + grad_sq)
        }).collect()
    }).collect();

    // Gradient of g for advection term.
    let (dgx, dgy) = gradient(&g, rows, cols);

    let dt = 0.5_f64;
    let mut phi: Vec<Vec<f64>> = phi0.iter().map(|r| r.to_vec()).collect();

    for _iter in 0..iterations {
        let mut new_phi = phi.clone();
        for r in 0..rows {
            for c in 0..cols {
                let kappa = compute_curvature(&phi, r, c, rows, cols);
                let (phi_x, phi_y) = central_diff_2d(&phi, r, c, rows, cols);
                let grad_phi_norm = (phi_x * phi_x + phi_y * phi_y + 1e-10).sqrt();

                // Curvature term: g * κ * |∇φ|
                let curvature_term = g[r][c] * kappa * grad_phi_norm;
                // Balloon term: α * g * |∇φ|
                let balloon_term = alpha * g[r][c] * grad_phi_norm;
                // Advection: β * (∇g · ∇φ)
                let advection_term = beta * (dgx[r][c] * phi_x + dgy[r][c] * phi_y);

                new_phi[r][c] = phi[r][c] + dt * (curvature_term + balloon_term + advection_term);
            }
        }
        phi = new_phi;
    }

    Ok(phi)
}

// ─── Internal: image processing helpers ──────────────────────────────────────

/// Simple Gaussian blur via separable convolution.
fn gaussian_blur(image: &[Vec<f64>], sigma: f64) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    let cols = image[0].len();

    if sigma < 1e-10 {
        return Ok(image.iter().map(|r| r.to_vec()).collect());
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;
    let kernel: Vec<f64> = (0..kernel_size).map(|i| {
        let x = i as f64 - radius as f64;
        (-x * x / (2.0 * sigma * sigma)).exp()
    }).collect();
    let kernel_sum: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|&k| k / kernel_sum).collect();

    // Row-wise convolution.
    let mut tmp = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let cc = (c as isize + ki as isize - radius as isize)
                    .max(0).min(cols as isize - 1) as usize;
                acc += image[r][cc] * kv;
            }
            tmp[r][c] = acc;
        }
    }

    // Column-wise convolution.
    let mut result = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let rr = (r as isize + ki as isize - radius as isize)
                    .max(0).min(rows as isize - 1) as usize;
                acc += tmp[rr][c] * kv;
            }
            result[r][c] = acc;
        }
    }

    Ok(result)
}

/// Compute image gradient (Gx, Gy) using central differences.
fn gradient(image: &[Vec<f64>], rows: usize, cols: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut gx = vec![vec![0.0f64; cols]; rows];
    let mut gy = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let (dx, dy) = central_diff_2d(image, r, c, rows, cols);
            gx[r][c] = dx;
            gy[r][c] = dy;
        }
    }
    (gx, gy)
}

/// Central difference at pixel (r, c).
#[inline]
fn central_diff_2d(
    image: &[Vec<f64>],
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
) -> (f64, f64) {
    let get = |ri: isize, ci: isize| -> f64 {
        let rr = ri.max(0).min(rows as isize - 1) as usize;
        let cc = ci.max(0).min(cols as isize - 1) as usize;
        image[rr][cc]
    };
    let ri = r as isize;
    let ci = c as isize;
    let dx = (get(ri, ci + 1) - get(ri, ci - 1)) * 0.5;
    let dy = (get(ri + 1, ci) - get(ri - 1, ci)) * 0.5;
    (dx, dy)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_image(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let cr = rows as f64 / 2.0;
        let cc = cols as f64 / 2.0;
        let r2 = (rows.min(cols) as f64 * 0.3).powi(2);
        (0..rows).map(|r| {
            (0..cols).map(|c| {
                let d2 = (r as f64 - cr).powi(2) + (c as f64 - cc).powi(2);
                if d2 <= r2 { 1.0 } else { 0.0 }
            }).collect()
        }).collect()
    }

    fn checkerboard_phi(rows: usize, cols: usize, sq: usize) -> Vec<Vec<f64>> {
        (0..rows).map(|r| {
            (0..cols).map(|c| {
                let v = ((r / sq) + (c / sq)) % 2;
                if v == 0 { 1.0 } else { -1.0 }
            }).collect()
        }).collect()
    }

    #[test]
    fn test_chan_vese_basic() {
        let image = disk_image(30, 30);
        let phi0  = checkerboard_phi(30, 30, 5);
        let config = LevelSetConfig {
            iterations: 30,
            dt: 0.5,
            ..Default::default()
        };
        let phi = chan_vese(&image, &phi0, &config).expect("chan_vese failed");
        assert_eq!(phi.len(), 30);
        assert_eq!(phi[0].len(), 30);
        // Check that the output contains both positive and negative values.
        let has_pos = phi.iter().flat_map(|r| r.iter()).any(|&v| v > 0.0);
        let has_neg = phi.iter().flat_map(|r| r.iter()).any(|&v| v < 0.0);
        assert!(has_pos, "phi should have positive values");
        assert!(has_neg, "phi should have negative values");
    }

    #[test]
    fn test_chan_vese_shape_mismatch() {
        let image = disk_image(10, 10);
        let phi0  = vec![vec![0.0f64; 10]; 12];
        let config = LevelSetConfig::default();
        assert!(chan_vese(&image, &phi0, &config).is_err());
    }

    #[test]
    fn test_geodesic_active_contour_basic() {
        let image = disk_image(20, 20);
        let phi0  = checkerboard_phi(20, 20, 4);
        let phi = geodesic_active_contour(&image, &phi0, 1.0, 0.0, 1.0, 20)
            .expect("geodesic failed");
        assert_eq!(phi.len(), 20);
        assert_eq!(phi[0].len(), 20);
    }

    #[test]
    fn test_geodesic_active_contour_shape_mismatch() {
        let image = disk_image(10, 10);
        let phi0  = vec![vec![0.0f64; 10]; 8];
        assert!(geodesic_active_contour(&image, &phi0, 1.0, 0.0, 1.0, 10).is_err());
    }

    #[test]
    fn test_heaviside_dirac() {
        let eps = 1.0;
        assert!((heaviside(10.0, eps) - 1.0).abs() < 0.05);
        assert!((heaviside(-10.0, eps)).abs() < 0.05);
        assert!((heaviside(0.0, eps) - 0.5).abs() < 1e-10);
        assert!(dirac(0.0, eps) > dirac(1.0, eps));
        assert!(dirac(1.0, eps) > dirac(5.0, eps));
    }

    #[test]
    fn test_gaussian_blur_identity_when_sigma_zero() {
        let image = disk_image(10, 10);
        let blurred = gaussian_blur(&image, 0.0).expect("blur failed");
        for (r, row) in image.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert!((blurred[r][c] - v).abs() < 1e-10);
            }
        }
    }
}
