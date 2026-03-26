//! Geodesic active contours and Chan-Vese level-set segmentation.
//!
//! Implements:
//! * **Chan-Vese** (Mumford-Shah functional, piecewise constant) for 2D and 3D.
//! * **Geodesic active contours** (Caselles 1997, edge-based) for 2D.
//!
//! References:
//! * Chan & Vese, "Active Contours Without Edges," TIP 2001.
//! * Caselles et al., "Geodesic Active Contours," IJCV 1997.

/// Configuration for level-set segmentation methods.
#[derive(Debug, Clone)]
pub struct LevelSetConfig {
    /// Number of iterations.
    pub n_iter: usize,
    /// Time step for explicit updates.
    pub dt: f64,
    /// Weight for curvature (smoothing) term.
    pub smoothing_weight: f64,
    /// Balloon force coefficient (geodesic method).
    pub balloon: f64,
    /// Chan-Vese length penalty (weight of perimeter term).
    pub mu: f64,
    /// Chan-Vese area penalty (weight of area term).
    pub nu: f64,
    /// Chan-Vese λ₁ (foreground data fidelity weight).
    pub lambda1: f64,
    /// Chan-Vese λ₂ (background data fidelity weight).
    pub lambda2: f64,
}

impl Default for LevelSetConfig {
    fn default() -> Self {
        LevelSetConfig {
            n_iter: 100,
            dt: 0.5,
            smoothing_weight: 1.0,
            balloon: 0.0,
            mu: 0.2,
            nu: 0.0,
            lambda1: 1.0,
            lambda2: 1.0,
        }
    }
}

/// Level-set evolution method selector.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelSetMethod {
    /// Chan-Vese segmentation (Mumford-Shah piecewise constant).
    ChanVese,
    /// Geodesic active contour (edge-based, Caselles 1997).
    GeodesicActiveContour,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Smoothed Heaviside function.
#[inline]
fn heaviside(phi: f64, eps: f64) -> f64 {
    0.5 * (1.0 + (2.0 / std::f64::consts::PI) * (phi / eps).atan())
}

/// Smoothed Dirac delta function.
#[inline]
fn dirac(phi: f64, eps: f64) -> f64 {
    eps / (std::f64::consts::PI * (phi * phi + eps * eps))
}

/// Clamp index to [0, n-1].
#[inline]
fn clamp_idx(i: isize, n: usize) -> usize {
    i.max(0).min(n as isize - 1) as usize
}

// ---------------------------------------------------------------------------
// Gradient magnitude
// ---------------------------------------------------------------------------

/// Compute the gradient magnitude of a 2D image using central differences.
pub fn compute_gradient_magnitude(image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let h = image.len();
    if h == 0 {
        return vec![];
    }
    let w = image[0].len();
    let mut grad = vec![vec![0.0f64; w]; h];
    for i in 0..h {
        for j in 0..w {
            let ip = clamp_idx(i as isize + 1, h);
            let im = clamp_idx(i as isize - 1, h);
            let jp = clamp_idx(j as isize + 1, w);
            let jm = clamp_idx(j as isize - 1, w);
            let gx = (image[ip][j] - image[im][j]) / 2.0;
            let gy = (image[i][jp] - image[i][jm]) / 2.0;
            grad[i][j] = (gx * gx + gy * gy).sqrt();
        }
    }
    grad
}

// ---------------------------------------------------------------------------
// Curvature (divergence of normalised gradient of φ)
// ---------------------------------------------------------------------------

/// Compute the mean curvature of the level-set function φ via central differences.
pub fn compute_curvature(phi: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let h = phi.len();
    if h == 0 {
        return vec![];
    }
    let w = phi[0].len();
    let mut curv = vec![vec![0.0f64; w]; h];
    let eps = 1e-8;
    for i in 0..h {
        for j in 0..w {
            let ip = clamp_idx(i as isize + 1, h);
            let im = clamp_idx(i as isize - 1, h);
            let jp = clamp_idx(j as isize + 1, w);
            let jm = clamp_idx(j as isize - 1, w);

            let phi_c = phi[i][j];
            let phi_ip = phi[ip][j];
            let phi_im = phi[im][j];
            let phi_jp = phi[i][jp];
            let phi_jm = phi[i][jm];

            // Partial derivatives
            let phix = (phi_ip - phi_im) / 2.0;
            let phiy = (phi_jp - phi_jm) / 2.0;
            let phixx = phi_ip - 2.0 * phi_c + phi_im;
            let phiyy = phi_jp - 2.0 * phi_c + phi_jm;
            // Cross derivative
            let phi_ipjp = phi[ip][jp];
            let phi_ipjm = phi[ip][jm];
            let phi_imjp = phi[im][jp];
            let phi_imjm = phi[im][jm];
            let phixy = (phi_ipjp - phi_ipjm - phi_imjp + phi_imjm) / 4.0;

            let norm2 = phix * phix + phiy * phiy + eps;
            curv[i][j] = (phixx * phiy * phiy - 2.0 * phixy * phix * phiy + phiyy * phix * phix)
                / (norm2 * norm2.sqrt());
        }
    }
    curv
}

// ---------------------------------------------------------------------------
// Helpers for phi reinitialization (simplified fast marching via sign * dist)
// ---------------------------------------------------------------------------

fn reinitialize_phi(phi: &mut Vec<Vec<f64>>) {
    let h = phi.len();
    if h == 0 {
        return;
    }
    let w = phi[0].len();
    // Simple approximate reinitialization: signed distance to zero crossing
    // via a 2D distance transform on the sign array.
    let sign: Vec<Vec<f64>> = (0..h)
        .map(|i| {
            (0..w)
                .map(|j| if phi[i][j] > 0.0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();
    // Euclidean distance transform approximation (Rosenfeld-Pfaltz)
    let mut dist = vec![vec![f64::INFINITY; w]; h];
    // Forward pass
    for i in 0..h {
        for j in 0..w {
            // Check if zero crossing nearby: if sign changes relative to neighbor
            let mut is_border = false;
            for (di, dj) in &[(-1isize, 0isize), (0, -1)] {
                let ni = i as isize + di;
                let nj = j as isize + dj;
                if ni >= 0 && ni < h as isize && nj >= 0 && nj < w as isize {
                    if sign[i][j] != sign[ni as usize][nj as usize] {
                        is_border = true;
                    }
                }
            }
            if is_border {
                dist[i][j] = 0.0;
            }
        }
    }
    // BFS-like distance propagation from borders
    let mut changed = true;
    let mut iter = 0;
    while changed && iter < (h + w) {
        changed = false;
        iter += 1;
        for i in 0..h {
            for j in 0..w {
                // Check zero crossing
                let s = sign[i][j];
                let mut min_d = dist[i][j];
                for (di, dj) in &[(-1isize, 0isize), (1, 0), (0, -1isize), (0, 1)] {
                    let ni = i as isize + di;
                    let nj = j as isize + dj;
                    if ni >= 0 && ni < h as isize && nj >= 0 && nj < w as isize {
                        let ns = sign[ni as usize][nj as usize];
                        if ns != s {
                            // Zero crossing between (i,j) and neighbor → distance ~0.5
                            let d = 0.5;
                            if d < min_d {
                                min_d = d;
                                changed = true;
                            }
                        } else {
                            let d = dist[ni as usize][nj as usize] + 1.0;
                            if d < min_d {
                                min_d = d;
                                changed = true;
                            }
                        }
                    }
                }
                dist[i][j] = min_d;
            }
        }
    }

    for i in 0..h {
        for j in 0..w {
            phi[i][j] = sign[i][j] * dist[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Chan-Vese 2D
// ---------------------------------------------------------------------------

/// Chan-Vese level-set segmentation for a 2D image.
///
/// The level set φ evolves to minimise:
///   E = λ₁ ∫(u₀-c₁)²H(φ) + λ₂ ∫(u₀-c₂)²(1-H(φ)) + μ ∫|∇φ| + ν ∫H(φ)
///
/// Returns the evolved φ field; φ > 0 = foreground, φ ≤ 0 = background.
pub fn chan_vese_2d(
    image: &[Vec<f64>],
    phi_init: &[Vec<f64>],
    config: &LevelSetConfig,
) -> Vec<Vec<f64>> {
    let h = image.len();
    if h == 0 {
        return vec![];
    }
    let w = image[0].len();
    let eps = 1.0_f64;

    let mut phi: Vec<Vec<f64>> = phi_init.iter().map(|row| row.to_vec()).collect();

    for iter in 0..config.n_iter {
        // Compute c1 and c2
        let mut sum_fg = 0.0f64;
        let mut cnt_fg = 0.0f64;
        let mut sum_bg = 0.0f64;
        let mut cnt_bg = 0.0f64;
        for i in 0..h {
            for j in 0..w {
                let u = image[i][j];
                let hv = heaviside(phi[i][j], eps);
                sum_fg += u * hv;
                cnt_fg += hv;
                sum_bg += u * (1.0 - hv);
                cnt_bg += 1.0 - hv;
            }
        }
        let c1 = if cnt_fg > 1e-12 { sum_fg / cnt_fg } else { 0.0 };
        let c2 = if cnt_bg > 1e-12 { sum_bg / cnt_bg } else { 0.0 };

        // Curvature
        let curv = compute_curvature(&phi);

        // Update φ
        let mut phi_new = phi.clone();
        for i in 0..h {
            for j in 0..w {
                let u = image[i][j];
                let d_phi = dirac(phi[i][j], eps);
                let kappa = curv[i][j];
                let diff_fg = (u - c1) * (u - c1);
                let diff_bg = (u - c2) * (u - c2);
                let rhs = config.mu * kappa - config.nu - config.lambda1 * diff_fg
                    + config.lambda2 * diff_bg;
                phi_new[i][j] = phi[i][j] + config.dt * d_phi * rhs;
            }
        }
        phi = phi_new;

        // Reinitialize every 20 steps
        if iter % 20 == 19 {
            reinitialize_phi(&mut phi);
        }
    }

    phi
}

// ---------------------------------------------------------------------------
// Chan-Vese 3D
// ---------------------------------------------------------------------------

fn compute_curvature_3d(phi: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
    let nd = phi.len();
    if nd == 0 {
        return vec![];
    }
    let nh = phi[0].len();
    if nh == 0 {
        return vec![vec![]];
    }
    let nw = phi[0][0].len();
    let eps = 1e-8;
    let mut curv = vec![vec![vec![0.0f64; nw]; nh]; nd];

    for d in 0..nd {
        for h in 0..nh {
            for w in 0..nw {
                let dp = clamp_idx(d as isize + 1, nd);
                let dm = clamp_idx(d as isize - 1, nd);
                let hp = clamp_idx(h as isize + 1, nh);
                let hm = clamp_idx(h as isize - 1, nh);
                let wp = clamp_idx(w as isize + 1, nw);
                let wm = clamp_idx(w as isize - 1, nw);

                let c = phi[d][h][w];
                let fx = (phi[dp][h][w] - phi[dm][h][w]) / 2.0;
                let fy = (phi[d][hp][w] - phi[d][hm][w]) / 2.0;
                let fz = (phi[d][h][wp] - phi[d][h][wm]) / 2.0;
                let fxx = phi[dp][h][w] - 2.0 * c + phi[dm][h][w];
                let fyy = phi[d][hp][w] - 2.0 * c + phi[d][hm][w];
                let fzz = phi[d][h][wp] - 2.0 * c + phi[d][h][wm];
                let fxy = (phi[dp][hp][w] - phi[dp][hm][w] - phi[dm][hp][w] + phi[dm][hm][w]) / 4.0;
                let fxz = (phi[dp][h][wp] - phi[dp][h][wm] - phi[dm][h][wp] + phi[dm][h][wm]) / 4.0;
                let fyz = (phi[d][hp][wp] - phi[d][hp][wm] - phi[d][hm][wp] + phi[d][hm][wm]) / 4.0;

                let norm2 = fx * fx + fy * fy + fz * fz + eps;
                let numerator = (fxx * (fy * fy + fz * fz)
                    + fyy * (fx * fx + fz * fz)
                    + fzz * (fx * fx + fy * fy)
                    - 2.0 * fxy * fx * fy
                    - 2.0 * fxz * fx * fz
                    - 2.0 * fyz * fy * fz);
                curv[d][h][w] = numerator / (norm2 * norm2.sqrt());
            }
        }
    }
    curv
}

/// Chan-Vese level-set segmentation for a 3D volume.
///
/// Returns the evolved φ field; φ > 0 = foreground.
pub fn chan_vese_3d(
    image: &[Vec<Vec<f64>>],
    phi_init: &[Vec<Vec<f64>>],
    config: &LevelSetConfig,
) -> Vec<Vec<Vec<f64>>> {
    let nd = image.len();
    if nd == 0 {
        return vec![];
    }
    let nh = image[0].len();
    let nw = if nh > 0 { image[0][0].len() } else { 0 };
    let eps = 1.0_f64;

    let mut phi: Vec<Vec<Vec<f64>>> = phi_init
        .iter()
        .map(|plane| plane.iter().map(|row| row.to_vec()).collect())
        .collect();

    for _iter in 0..config.n_iter {
        // Compute c1, c2
        let mut sum_fg = 0.0f64;
        let mut cnt_fg = 0.0f64;
        let mut sum_bg = 0.0f64;
        let mut cnt_bg = 0.0f64;
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let u = image[d][h][w];
                    let hv = heaviside(phi[d][h][w], eps);
                    sum_fg += u * hv;
                    cnt_fg += hv;
                    sum_bg += u * (1.0 - hv);
                    cnt_bg += 1.0 - hv;
                }
            }
        }
        let c1 = if cnt_fg > 1e-12 { sum_fg / cnt_fg } else { 0.0 };
        let c2 = if cnt_bg > 1e-12 { sum_bg / cnt_bg } else { 0.0 };

        let curv = compute_curvature_3d(&phi);

        let mut phi_new = phi.clone();
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let u = image[d][h][w];
                    let d_phi = dirac(phi[d][h][w], eps);
                    let kappa = curv[d][h][w];
                    let diff_fg = (u - c1) * (u - c1);
                    let diff_bg = (u - c2) * (u - c2);
                    let rhs = config.mu * kappa - config.nu - config.lambda1 * diff_fg
                        + config.lambda2 * diff_bg;
                    phi_new[d][h][w] = phi[d][h][w] + config.dt * d_phi * rhs;
                }
            }
        }
        phi = phi_new;
    }

    phi
}

// ---------------------------------------------------------------------------
// Geodesic active contour 2D
// ---------------------------------------------------------------------------

/// Geodesic active contour level-set evolution (Caselles 1997) for 2D.
///
/// Uses edge stopping function g = 1/(1 + |∇I|²), upwind scheme for advection.
pub fn geodesic_contour_2d(
    image: &[Vec<f64>],
    phi_init: &[Vec<f64>],
    config: &LevelSetConfig,
) -> Vec<Vec<f64>> {
    let h = image.len();
    if h == 0 {
        return vec![];
    }
    let w = image[0].len();

    // Compute gradient magnitude of image
    let grad_mag = compute_gradient_magnitude(image);

    // Edge stopping function g(x) = 1 / (1 + |∇I(x)|²)
    let g: Vec<Vec<f64>> = (0..h)
        .map(|i| {
            (0..w)
                .map(|j| {
                    let gm = grad_mag[i][j];
                    1.0 / (1.0 + gm * gm)
                })
                .collect()
        })
        .collect();

    // Gradient of g (for advection term ∇g · ∇φ)
    let mut gx = vec![vec![0.0f64; w]; h];
    let mut gy = vec![vec![0.0f64; w]; h];
    for i in 0..h {
        for j in 0..w {
            let ip = clamp_idx(i as isize + 1, h);
            let im = clamp_idx(i as isize - 1, h);
            let jp = clamp_idx(j as isize + 1, w);
            let jm = clamp_idx(j as isize - 1, w);
            gx[i][j] = (g[ip][j] - g[im][j]) / 2.0;
            gy[i][j] = (g[i][jp] - g[i][jm]) / 2.0;
        }
    }

    let mut phi: Vec<Vec<f64>> = phi_init.iter().map(|r| r.to_vec()).collect();

    for _iter in 0..config.n_iter {
        let curv = compute_curvature(&phi);

        let mut phi_new = phi.clone();
        for i in 0..h {
            for j in 0..w {
                let ip = clamp_idx(i as isize + 1, h);
                let im = clamp_idx(i as isize - 1, h);
                let jp = clamp_idx(j as isize + 1, w);
                let jm = clamp_idx(j as isize - 1, w);

                // Upwind differences for |∇φ| approximation
                let dpx_fwd = phi[ip][j] - phi[i][j];
                let dpx_bwd = phi[i][j] - phi[im][j];
                let dpy_fwd = phi[i][jp] - phi[i][j];
                let dpy_bwd = phi[i][j] - phi[i][jm];

                // Godunov upwind scheme for |∇φ|
                let grad_phi_norm = {
                    let ax = dpx_bwd.max(0.0).powi(2) + dpx_fwd.min(0.0).powi(2);
                    let ay = dpy_bwd.max(0.0).powi(2) + dpy_fwd.min(0.0).powi(2);
                    (ax + ay).sqrt()
                };

                // Central differences for ∇φ (advection)
                let phix_c = (phi[ip][j] - phi[im][j]) / 2.0;
                let phiy_c = (phi[i][jp] - phi[i][jm]) / 2.0;

                // Advection term: ∇g · ∇φ
                let advection = gx[i][j] * phix_c + gy[i][j] * phiy_c;

                // Smoothing term: g * κ * |∇φ|
                let smoothing = g[i][j] * config.smoothing_weight * curv[i][j] * grad_phi_norm;

                // Balloon term: balloon * g * |∇φ|
                let balloon_term = config.balloon * g[i][j] * grad_phi_norm;

                phi_new[i][j] = phi[i][j] + config.dt * (smoothing + advection + balloon_term);
            }
        }
        phi = phi_new;
    }

    phi
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Convert a level-set function φ to a binary mask: φ > 0 → true.
pub fn binary_from_levelset(phi: &[Vec<f64>]) -> Vec<Vec<bool>> {
    phi.iter()
        .map(|row| row.iter().map(|&v| v > 0.0).collect())
        .collect()
}

/// Initialize a circle-shaped level-set function.
///
/// φ(x,y) = radius - √((x-cx)² + (y-cy)²)
/// Positive inside the circle, negative outside.
pub fn initialize_circle(
    height: usize,
    width: usize,
    center: (usize, usize),
    radius: f64,
) -> Vec<Vec<f64>> {
    let (cy, cx) = center;
    (0..height)
        .map(|i| {
            (0..width)
                .map(|j| {
                    let di = i as f64 - cy as f64;
                    let dj = j as f64 - cx as f64;
                    radius - (di * di + dj * dj).sqrt()
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image_bright_center(h: usize, w: usize) -> Vec<Vec<f64>> {
        let mut img = vec![vec![0.2f64; w]; h];
        let ch = h / 2;
        let cw = w / 2;
        let r = (h.min(w) / 4) as f64;
        for i in 0..h {
            for j in 0..w {
                let di = i as f64 - ch as f64;
                let dj = j as f64 - cw as f64;
                if (di * di + dj * dj).sqrt() < r {
                    img[i][j] = 0.9;
                }
            }
        }
        img
    }

    #[test]
    fn test_initialize_circle() {
        let phi = initialize_circle(20, 20, (10, 10), 5.0);
        assert!(phi[10][10] > 0.0, "Center should be inside (phi > 0)");
        assert!(phi[0][0] < 0.0, "Corner should be outside (phi < 0)");
    }

    #[test]
    fn test_binary_from_levelset() {
        let phi = vec![vec![-1.0, 0.5], vec![0.0, -0.5]];
        let mask = binary_from_levelset(&phi);
        assert!(!mask[0][0]); // -1.0 → false
        assert!(mask[0][1]); //  0.5 → true
        assert!(!mask[1][0]); //  0.0 → false (not strictly > 0)
        assert!(!mask[1][1]); // -0.5 → false
    }

    #[test]
    fn test_compute_gradient_magnitude_flat() {
        // Flat image → gradient magnitude should be ~0 everywhere interior
        let img = vec![vec![1.0f64; 10]; 10];
        let grad = compute_gradient_magnitude(&img);
        for row in &grad {
            for &v in row {
                assert!(
                    v.abs() < 1e-10,
                    "Expected ~0 gradient on flat image, got {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_compute_gradient_magnitude_edge() {
        // Step edge: left half 0, right half 1
        let h = 5;
        let w = 10;
        let img: Vec<Vec<f64>> = (0..h)
            .map(|_| (0..w).map(|j| if j < 5 { 0.0 } else { 1.0 }).collect())
            .collect();
        let grad = compute_gradient_magnitude(&img);
        // At j=4 (boundary), gradient should be > 0
        assert!(
            grad[2][4] > 0.0 || grad[2][5] > 0.0,
            "Expected non-zero gradient at edge"
        );
        // Far from edge, gradient should be ~0
        assert!(grad[2][0].abs() < 1e-10 || grad[2][1].abs() < 1e-10);
    }

    #[test]
    fn test_chan_vese_2d_bright_center() {
        // 10×10 image with bright center; circle initialised around center
        let img = make_image_bright_center(10, 10);
        let phi_init = initialize_circle(10, 10, (5, 5), 3.0);
        let config = LevelSetConfig {
            n_iter: 30,
            dt: 0.3,
            mu: 0.3,
            lambda1: 1.0,
            lambda2: 1.0,
            ..Default::default()
        };
        let phi = chan_vese_2d(&img, &phi_init, &config);
        assert_eq!(phi.len(), 10);
        assert_eq!(phi[0].len(), 10);
        // Center should remain positive (foreground)
        assert!(
            phi[5][5] > 0.0,
            "Center expected inside (phi>0), got {}",
            phi[5][5]
        );
        // binary_from_levelset should work
        let mask = binary_from_levelset(&phi);
        assert!(mask[5][5], "Center should be foreground");
    }

    #[test]
    fn test_chan_vese_3d_basic() {
        // Tiny 4×4×4 volume
        let image: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| (0..4).map(|_| vec![0.5f64; 4]).collect())
            .collect();
        let phi_init: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|d| {
                (0..4)
                    .map(|h| {
                        (0..4)
                            .map(|w| {
                                let dd = d as f64 - 2.0;
                                let dh = h as f64 - 2.0;
                                let dw = w as f64 - 2.0;
                                1.5 - (dd * dd + dh * dh + dw * dw).sqrt()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let config = LevelSetConfig {
            n_iter: 5,
            ..Default::default()
        };
        let phi = chan_vese_3d(&image, &phi_init, &config);
        assert_eq!(phi.len(), 4);
        assert_eq!(phi[0].len(), 4);
        assert_eq!(phi[0][0].len(), 4);
    }

    #[test]
    fn test_geodesic_contour_edge_stopping_function() {
        // Flat image → g should be ~1 everywhere
        let img = vec![vec![0.5f64; 10]; 10];
        let grad = compute_gradient_magnitude(&img);
        for row in &grad {
            for &gm in row {
                let g = 1.0 / (1.0 + gm * gm);
                assert!(
                    (g - 1.0).abs() < 1e-6,
                    "g on flat image should be ~1, got {}",
                    g
                );
            }
        }

        // Edge image → g should be < 1 at boundary
        let img_edge: Vec<Vec<f64>> = (0..10)
            .map(|_| (0..10).map(|j| if j < 5 { 0.0 } else { 1.0 }).collect())
            .collect();
        let grad_edge = compute_gradient_magnitude(&img_edge);
        let has_large_grad = grad_edge.iter().any(|row| {
            row.iter().any(|&gm| {
                let g = 1.0 / (1.0 + gm * gm);
                g < 0.99
            })
        });
        assert!(has_large_grad, "Expected g < 1 near edge");
    }

    #[test]
    fn test_geodesic_contour_2d_runs() {
        let img = make_image_bright_center(12, 12);
        let phi_init = initialize_circle(12, 12, (6, 6), 4.0);
        let config = LevelSetConfig {
            n_iter: 20,
            dt: 0.2,
            smoothing_weight: 1.0,
            balloon: 0.1,
            ..Default::default()
        };
        let phi = geodesic_contour_2d(&img, &phi_init, &config);
        assert_eq!(phi.len(), 12);
        assert_eq!(phi[0].len(), 12);
    }

    #[test]
    fn test_compute_curvature_constant_phi() {
        // Constant φ → curvature numerator involves uniform gradients → should be near zero
        let phi = vec![vec![2.0f64; 8]; 8];
        let curv = compute_curvature(&phi);
        for row in &curv {
            for &c in row {
                assert!(
                    c.abs() < 1e-5,
                    "Expected ~0 curvature on constant phi, got {}",
                    c
                );
            }
        }
    }
}
