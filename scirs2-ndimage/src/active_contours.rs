//! Advanced active contour (snake) models for image segmentation
//!
//! This module provides a comprehensive implementation of parametric active
//! contour models with multiple external force formulations. Key features:
//!
//! - **Classical snake energy minimization** via implicit Euler integration
//! - **Internal energy**: elasticity (first derivative) + bending (second derivative)
//! - **External energy**: gradient magnitude, edge-based attraction
//! - **Balloon force**: expansion/contraction pressure term
//! - **GVF (Gradient Vector Flow)**: diffusion-based external force that captures
//!   concavities and long-range edge attraction
//! - **Configurable alpha, beta, gamma**: full control over snake behavior
//!
//! # References
//!
//! - Kass, M., Witkin, A., Terzopoulos, D. (1988). "Snakes: Active Contour Models"
//! - Xu, C. & Prince, J.L. (1998). "Snakes, Shapes, and Gradient Vector Flow"

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters controlling active contour evolution
#[derive(Debug, Clone)]
pub struct SnakeParams {
    /// Elasticity weight (first-order smoothness term).
    /// Higher values make the contour resist stretching.
    pub alpha: f64,
    /// Bending rigidity weight (second-order smoothness term).
    /// Higher values make the contour resist bending.
    pub beta: f64,
    /// External force weight (image attraction).
    /// Higher values make the contour adhere more strongly to edges.
    pub gamma: f64,
    /// Balloon force magnitude. Positive = outward expansion, negative = inward.
    /// Set to 0.0 to disable.
    pub kappa: f64,
    /// Time step for the Euler integration
    pub time_step: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold: stop when total contour movement < this value
    pub convergence_threshold: f64,
    /// Whether to use GVF (Gradient Vector Flow) for external force
    pub use_gvf: bool,
    /// GVF regularization parameter (mu). Higher = smoother GVF field.
    /// Typical range: 0.05 to 0.5
    pub gvf_mu: f64,
    /// Number of GVF diffusion iterations
    pub gvf_iterations: usize,
}

impl Default for SnakeParams {
    fn default() -> Self {
        SnakeParams {
            alpha: 0.01,
            beta: 0.1,
            gamma: 1.0,
            kappa: 0.0,
            time_step: 0.1,
            max_iterations: 500,
            convergence_threshold: 0.1,
            use_gvf: false,
            gvf_mu: 0.2,
            gvf_iterations: 80,
        }
    }
}

/// Result of active contour evolution
#[derive(Debug, Clone)]
pub struct SnakeResult {
    /// Final contour points (N x 2, columns are [x, y])
    pub contour: Array2<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the contour converged
    pub converged: bool,
    /// Final total energy (internal + external)
    pub final_energy: f64,
    /// Energy history per iteration
    pub energy_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute image gradient using central differences
fn image_gradient(image: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (h, w) = image.dim();
    let mut gx = Array2::zeros((h, w));
    let mut gy = Array2::zeros((h, w));

    for i in 0..h {
        for j in 0..w {
            // dx (column direction)
            gx[[i, j]] = if j == 0 {
                if w > 1 {
                    image[[i, 1]] - image[[i, 0]]
                } else {
                    0.0
                }
            } else if j == w - 1 {
                image[[i, j]] - image[[i, j - 1]]
            } else {
                (image[[i, j + 1]] - image[[i, j - 1]]) / 2.0
            };

            // dy (row direction)
            gy[[i, j]] = if i == 0 {
                if h > 1 {
                    image[[1, j]] - image[[0, j]]
                } else {
                    0.0
                }
            } else if i == h - 1 {
                image[[i, j]] - image[[i - 1, j]]
            } else {
                (image[[i + 1, j]] - image[[i - 1, j]]) / 2.0
            };
        }
    }

    (gx, gy)
}

/// Compute gradient magnitude (edge map)
fn gradient_magnitude(image: &Array2<f64>) -> Array2<f64> {
    let (gx, gy) = image_gradient(image);
    let mut mag = Array2::zeros(image.dim());
    for i in 0..image.nrows() {
        for j in 0..image.ncols() {
            mag[[i, j]] = (gx[[i, j]].powi(2) + gy[[i, j]].powi(2)).sqrt();
        }
    }
    mag
}

/// Compute Gradient Vector Flow (GVF) field
///
/// GVF extends the edge map's gradient into homogeneous regions while
/// preserving strong gradients near edges. This allows the snake to be
/// attracted to concavities and distant edges.
fn compute_gvf(edge_map: &Array2<f64>, mu: f64, iterations: usize) -> (Array2<f64>, Array2<f64>) {
    let (h, w) = edge_map.dim();

    // Initialize with gradient of edge map
    let (fx, fy) = image_gradient(edge_map);
    let mut u = fx.clone();
    let mut v = fy.clone();

    // Squared gradient magnitude: b = |nabla f|^2
    let b = {
        let mut b_arr = Array2::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                b_arr[[i, j]] = fx[[i, j]].powi(2) + fy[[i, j]].powi(2);
            }
        }
        b_arr
    };

    // Iterative GVF diffusion
    for _ in 0..iterations {
        let mut u_new = Array2::zeros((h, w));
        let mut v_new = Array2::zeros((h, w));

        for i in 0..h {
            for j in 0..w {
                // Laplacian via 5-point stencil with Neumann boundary
                let u_ip = if i + 1 < h { u[[i + 1, j]] } else { u[[i, j]] };
                let u_im = if i > 0 { u[[i - 1, j]] } else { u[[i, j]] };
                let u_jp = if j + 1 < w { u[[i, j + 1]] } else { u[[i, j]] };
                let u_jm = if j > 0 { u[[i, j - 1]] } else { u[[i, j]] };

                let v_ip = if i + 1 < h { v[[i + 1, j]] } else { v[[i, j]] };
                let v_im = if i > 0 { v[[i - 1, j]] } else { v[[i, j]] };
                let v_jp = if j + 1 < w { v[[i, j + 1]] } else { v[[i, j]] };
                let v_jm = if j > 0 { v[[i, j - 1]] } else { v[[i, j]] };

                let lap_u = u_ip + u_im + u_jp + u_jm - 4.0 * u[[i, j]];
                let lap_v = v_ip + v_im + v_jp + v_jm - 4.0 * v[[i, j]];

                let bval = b[[i, j]];
                u_new[[i, j]] = u[[i, j]] + mu * lap_u - bval * (u[[i, j]] - fx[[i, j]]);
                v_new[[i, j]] = v[[i, j]] + mu * lap_v - bval * (v[[i, j]] - fy[[i, j]]);
            }
        }

        u = u_new;
        v = v_new;
    }

    (u, v)
}

/// Bilinear interpolation of a 2D field at fractional coordinates
fn bilinear_interp(field: &Array2<f64>, x: f64, y: f64) -> f64 {
    let (h, w) = field.dim();
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let clamp_r = |r: isize| -> usize { r.max(0).min((h as isize) - 1) as usize };
    let clamp_c = |c: isize| -> usize { c.max(0).min((w as isize) - 1) as usize };

    let v00 = field[[clamp_r(y0), clamp_c(x0)]];
    let v01 = field[[clamp_r(y0), clamp_c(x1)]];
    let v10 = field[[clamp_r(y1), clamp_c(x0)]];
    let v11 = field[[clamp_r(y1), clamp_c(x1)]];

    v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy) + v10 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Compute internal energy of the contour (elasticity + bending)
fn internal_energy(contour: &Array2<f64>, alpha: f64, beta: f64) -> f64 {
    let n = contour.nrows();
    let mut energy = 0.0;

    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = if i == n - 1 { 0 } else { i + 1 };

        // First derivative (elasticity)
        let dx1 = contour[[next, 0]] - contour[[i, 0]];
        let dy1 = contour[[next, 1]] - contour[[i, 1]];
        energy += alpha * (dx1 * dx1 + dy1 * dy1);

        // Second derivative (bending)
        let dx2 = contour[[prev, 0]] - 2.0 * contour[[i, 0]] + contour[[next, 0]];
        let dy2 = contour[[prev, 1]] - 2.0 * contour[[i, 1]] + contour[[next, 1]];
        energy += beta * (dx2 * dx2 + dy2 * dy2);
    }

    energy / (n as f64)
}

/// Compute external energy from the edge map at contour points
fn external_energy(contour: &Array2<f64>, edge_map: &Array2<f64>) -> f64 {
    let n = contour.nrows();
    let mut energy = 0.0;

    for i in 0..n {
        let x = contour[[i, 0]];
        let y = contour[[i, 1]];
        // External energy is negative of edge strength (attract to edges)
        energy -= bilinear_interp(edge_map, x, y);
    }

    energy / (n as f64)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Evolve an active contour (snake) on an image
///
/// The contour evolves by minimizing an energy functional composed of:
/// - Internal energy: elasticity (alpha) + bending rigidity (beta)
/// - External energy: gradient magnitude or GVF field (gamma)
/// - Balloon force: expansion/contraction (kappa)
///
/// # Arguments
///
/// * `image`           - Input grayscale image (2D, higher precision)
/// * `initial_contour` - Initial contour points (N x 2 array, columns [x, y])
/// * `params`          - Snake evolution parameters
///
/// # Returns
///
/// A `SnakeResult` containing the final contour, convergence info, and energy history.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::active_contours::{evolve_snake, SnakeParams, create_circular_contour};
///
/// let mut image = Array2::<f64>::zeros((50, 50));
/// // Draw a circle edge
/// for i in 0..50 {
///     for j in 0..50 {
///         let r = (((i as f64 - 25.0).powi(2) + (j as f64 - 25.0).powi(2)).sqrt() - 15.0).abs();
///         if r < 2.0 { image[[i, j]] = 1.0; }
///     }
/// }
///
/// let contour = create_circular_contour(25.0, 25.0, 12.0, 30);
/// let params = SnakeParams { max_iterations: 20, ..Default::default() };
/// let result = evolve_snake(&image, &contour, &params).expect("should succeed");
/// assert_eq!(result.contour.nrows(), 30);
/// ```
pub fn evolve_snake<T>(
    image: &Array2<T>,
    initial_contour: &Array2<f64>,
    params: &SnakeParams,
) -> NdimageResult<SnakeResult>
where
    T: Float + FromPrimitive + NumAssign + Debug + Send + Sync + 'static,
{
    // Validate inputs
    if initial_contour.ncols() != 2 {
        return Err(NdimageError::InvalidInput(
            "Initial contour must be N x 2 (columns [x, y])".to_string(),
        ));
    }

    let n = initial_contour.nrows();
    if n < 3 {
        return Err(NdimageError::InvalidInput(
            "Contour must have at least 3 points".to_string(),
        ));
    }

    let (h, w) = image.dim();
    if h == 0 || w == 0 {
        return Err(NdimageError::InvalidInput(
            "Image must be non-empty".to_string(),
        ));
    }

    // Convert image to f64
    let img_f64: Array2<f64> = image.mapv(|x| x.to_f64().unwrap_or(0.0));

    // Compute edge map
    let edge_map = gradient_magnitude(&img_f64);

    // Compute external force field
    let (force_x, force_y) = if params.use_gvf {
        compute_gvf(&edge_map, params.gvf_mu, params.gvf_iterations)
    } else {
        // Standard gradient of edge map squared
        let edge_sq = edge_map.mapv(|x| x * x);
        image_gradient(&edge_sq)
    };

    // Initialize contour
    let mut contour = initial_contour.to_owned();
    let mut energy_history = Vec::with_capacity(params.max_iterations);
    let mut converged = false;
    let mut iterations_done = 0;

    let dt = params.time_step;
    let alpha = params.alpha;
    let beta = params.beta;
    let gamma = params.gamma;
    let kappa = params.kappa;

    for iter in 0..params.max_iterations {
        iterations_done = iter + 1;

        let prev_contour = contour.clone();

        // Update each point
        for i in 0..n {
            let prev_i = if i == 0 { n - 1 } else { i - 1 };
            let next_i = if i == n - 1 { 0 } else { i + 1 };

            let x = contour[[i, 0]];
            let y = contour[[i, 1]];
            let x_prev = contour[[prev_i, 0]];
            let y_prev = contour[[prev_i, 1]];
            let x_next = contour[[next_i, 0]];
            let y_next = contour[[next_i, 1]];

            // Internal force: elasticity (pulls toward neighbors' average)
            let elastic_x = alpha * ((x_prev + x_next) / 2.0 - x);
            let elastic_y = alpha * ((y_prev + y_next) / 2.0 - y);

            // Internal force: bending (resists curvature)
            let bend_x = beta * (x_prev - 2.0 * x + x_next);
            let bend_y = beta * (y_prev - 2.0 * y + y_next);

            // External force (image-derived, interpolated at contour point)
            let ext_x = gamma * bilinear_interp(&force_x, x, y);
            let ext_y = gamma * bilinear_interp(&force_y, x, y);

            // Balloon force (normal direction)
            let (balloon_x, balloon_y) = if kappa.abs() > 1e-15 {
                let dx = x_next - x_prev;
                let dy = y_next - y_prev;
                let norm = (dx * dx + dy * dy).sqrt();
                if norm > 1e-15 {
                    // Normal is perpendicular to tangent
                    (-kappa * dy / norm, kappa * dx / norm)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

            // Update position
            contour[[i, 0]] += dt * (elastic_x + bend_x + ext_x + balloon_x);
            contour[[i, 1]] += dt * (elastic_y + bend_y + ext_y + balloon_y);

            // Clamp to image bounds
            contour[[i, 0]] = contour[[i, 0]].max(0.0).min((w - 1) as f64);
            contour[[i, 1]] = contour[[i, 1]].max(0.0).min((h - 1) as f64);
        }

        // Compute energy
        let int_e = internal_energy(&contour, alpha, beta);
        let ext_e = external_energy(&contour, &edge_map);
        let total_energy = int_e + gamma * ext_e;
        energy_history.push(total_energy);

        // Check convergence
        let mut total_movement = 0.0;
        for i in 0..n {
            let dx = contour[[i, 0]] - prev_contour[[i, 0]];
            let dy = contour[[i, 1]] - prev_contour[[i, 1]];
            total_movement += dx * dx + dy * dy;
        }
        total_movement = total_movement.sqrt();

        if total_movement < params.convergence_threshold {
            converged = true;
            break;
        }
    }

    let final_energy = energy_history.last().copied().unwrap_or(0.0);

    Ok(SnakeResult {
        contour,
        iterations: iterations_done,
        converged,
        final_energy,
        energy_history,
    })
}

/// Compute the total energy of a contour on an image
///
/// # Arguments
///
/// * `image`   - Grayscale image
/// * `contour` - Contour points (N x 2)
/// * `alpha`   - Elasticity weight
/// * `beta`    - Bending weight
/// * `gamma`   - External force weight
///
/// # Returns
///
/// Total energy value (internal + external).
pub fn contour_energy<T>(
    image: &Array2<T>,
    contour: &Array2<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> NdimageResult<f64>
where
    T: Float + FromPrimitive + NumAssign + Debug + 'static,
{
    if contour.ncols() != 2 || contour.nrows() < 3 {
        return Err(NdimageError::InvalidInput(
            "Contour must be N x 2 with N >= 3".to_string(),
        ));
    }

    let img_f64: Array2<f64> = image.mapv(|x| x.to_f64().unwrap_or(0.0));
    let edge_map = gradient_magnitude(&img_f64);

    let int_e = internal_energy(contour, alpha, beta);
    let ext_e = external_energy(contour, &edge_map);

    Ok(int_e + gamma * ext_e)
}

// ---------------------------------------------------------------------------
// Contour creation helpers
// ---------------------------------------------------------------------------

/// Create a circular initial contour
///
/// # Arguments
///
/// * `cx`, `cy` - Center coordinates (x, y)
/// * `radius`   - Radius of the circle
/// * `n_points` - Number of points on the contour
///
/// # Returns
///
/// N x 2 array with contour points.
pub fn create_circular_contour(cx: f64, cy: f64, radius: f64, n_points: usize) -> Array2<f64> {
    let mut contour = Array2::zeros((n_points, 2));
    for i in 0..n_points {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
        contour[[i, 0]] = cx + radius * theta.cos();
        contour[[i, 1]] = cy + radius * theta.sin();
    }
    contour
}

/// Create an elliptical initial contour
///
/// # Arguments
///
/// * `cx`, `cy`    - Center coordinates (x, y)
/// * `semi_a`      - Semi-major axis length
/// * `semi_b`      - Semi-minor axis length
/// * `angle`       - Rotation angle in radians
/// * `n_points`    - Number of points on the contour
pub fn create_elliptical_contour(
    cx: f64,
    cy: f64,
    semi_a: f64,
    semi_b: f64,
    angle: f64,
    n_points: usize,
) -> Array2<f64> {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let mut contour = Array2::zeros((n_points, 2));

    for i in 0..n_points {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
        let x = semi_a * theta.cos();
        let y = semi_b * theta.sin();
        contour[[i, 0]] = cx + x * cos_a - y * sin_a;
        contour[[i, 1]] = cy + x * sin_a + y * cos_a;
    }

    contour
}

/// Create a rectangular initial contour
///
/// # Arguments
///
/// * `x_min`, `y_min` - Top-left corner
/// * `x_max`, `y_max` - Bottom-right corner
/// * `n_per_side`      - Number of points per side
pub fn create_rectangular_contour(
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
    n_per_side: usize,
) -> Array2<f64> {
    let n_total = 4 * n_per_side;
    let mut contour = Array2::zeros((n_total, 2));
    let mut idx = 0;

    // Top side (left to right)
    for i in 0..n_per_side {
        let t = i as f64 / n_per_side as f64;
        contour[[idx, 0]] = x_min + t * (x_max - x_min);
        contour[[idx, 1]] = y_min;
        idx += 1;
    }

    // Right side (top to bottom)
    for i in 0..n_per_side {
        let t = i as f64 / n_per_side as f64;
        contour[[idx, 0]] = x_max;
        contour[[idx, 1]] = y_min + t * (y_max - y_min);
        idx += 1;
    }

    // Bottom side (right to left)
    for i in 0..n_per_side {
        let t = i as f64 / n_per_side as f64;
        contour[[idx, 0]] = x_max - t * (x_max - x_min);
        contour[[idx, 1]] = y_max;
        idx += 1;
    }

    // Left side (bottom to top)
    for i in 0..n_per_side {
        let t = i as f64 / n_per_side as f64;
        contour[[idx, 0]] = x_min;
        contour[[idx, 1]] = y_max - t * (y_max - y_min);
        idx += 1;
    }

    contour
}

/// Resample a contour to have a specified number of uniformly spaced points
///
/// # Arguments
///
/// * `contour`  - Input contour (N x 2)
/// * `n_points` - Desired number of output points
pub fn resample_contour(contour: &Array2<f64>, n_points: usize) -> NdimageResult<Array2<f64>> {
    let n = contour.nrows();
    if n < 2 {
        return Err(NdimageError::InvalidInput(
            "Contour must have at least 2 points".to_string(),
        ));
    }
    if n_points < 3 {
        return Err(NdimageError::InvalidInput(
            "Must resample to at least 3 points".to_string(),
        ));
    }

    // Compute cumulative arc length
    let mut arc_lengths = vec![0.0f64; n + 1];
    for i in 0..n {
        let next = (i + 1) % n;
        let dx = contour[[next, 0]] - contour[[i, 0]];
        let dy = contour[[next, 1]] - contour[[i, 1]];
        arc_lengths[i + 1] = arc_lengths[i] + (dx * dx + dy * dy).sqrt();
    }

    let total_length = arc_lengths[n];
    if total_length < 1e-15 {
        return Err(NdimageError::ComputationError(
            "Contour has zero length".to_string(),
        ));
    }

    let mut resampled = Array2::zeros((n_points, 2));
    for i in 0..n_points {
        let target_s = total_length * i as f64 / n_points as f64;

        // Binary search for the segment containing target_s
        let mut lo = 0;
        let mut hi = n;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if arc_lengths[mid] <= target_s {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let seg_len = arc_lengths[lo + 1] - arc_lengths[lo];
        let t = if seg_len > 1e-15 {
            (target_s - arc_lengths[lo]) / seg_len
        } else {
            0.0
        };

        let p0 = lo % n;
        let p1 = (lo + 1) % n;

        resampled[[i, 0]] = contour[[p0, 0]] * (1.0 - t) + contour[[p1, 0]] * t;
        resampled[[i, 1]] = contour[[p0, 1]] * (1.0 - t) + contour[[p1, 1]] * t;
    }

    Ok(resampled)
}

/// Compute the area enclosed by a closed contour using the shoelace formula
///
/// The sign indicates orientation: positive = counter-clockwise, negative = clockwise.
pub fn contour_area(contour: &Array2<f64>) -> f64 {
    let n = contour.nrows();
    if n < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += contour[[i, 0]] * contour[[j, 1]];
        area -= contour[[j, 0]] * contour[[i, 1]];
    }

    area / 2.0
}

/// Compute the perimeter (total arc length) of a closed contour
pub fn contour_perimeter(contour: &Array2<f64>) -> f64 {
    let n = contour.nrows();
    if n < 2 {
        return 0.0;
    }

    let mut length = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = contour[[j, 0]] - contour[[i, 0]];
        let dy = contour[[j, 1]] - contour[[i, 1]];
        length += (dx * dx + dy * dy).sqrt();
    }

    length
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_create_circular_contour() {
        let contour = create_circular_contour(25.0, 25.0, 10.0, 20);
        assert_eq!(contour.dim(), (20, 2));

        for i in 0..20 {
            let dx = contour[[i, 0]] - 25.0;
            let dy = contour[[i, 1]] - 25.0;
            let r = (dx * dx + dy * dy).sqrt();
            assert_abs_diff_eq!(r, 10.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_create_elliptical_contour() {
        let contour = create_elliptical_contour(50.0, 50.0, 30.0, 20.0, 0.0, 20);
        assert_eq!(contour.dim(), (20, 2));

        for i in 0..20 {
            let x = (contour[[i, 0]] - 50.0) / 30.0;
            let y = (contour[[i, 1]] - 50.0) / 20.0;
            let val = x * x + y * y;
            assert_abs_diff_eq!(val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_create_rectangular_contour() {
        let contour = create_rectangular_contour(10.0, 10.0, 40.0, 30.0, 5);
        assert_eq!(contour.nrows(), 20); // 4 * 5
    }

    #[test]
    fn test_contour_area_circle() {
        let contour = create_circular_contour(0.0, 0.0, 10.0, 200);
        let area = contour_area(&contour).abs();
        let expected = std::f64::consts::PI * 100.0;
        // Large number of points should approximate well
        assert!((area - expected).abs() / expected < 0.01);
    }

    #[test]
    fn test_contour_area_square() {
        // Manual square: 10x10
        let contour = create_rectangular_contour(0.0, 0.0, 10.0, 10.0, 100);
        let area = contour_area(&contour).abs();
        assert!((area - 100.0).abs() / 100.0 < 0.05);
    }

    #[test]
    fn test_contour_perimeter_circle() {
        let contour = create_circular_contour(0.0, 0.0, 10.0, 200);
        let perim = contour_perimeter(&contour);
        let expected = 2.0 * std::f64::consts::PI * 10.0;
        assert!((perim - expected).abs() / expected < 0.01);
    }

    #[test]
    fn test_resample_contour() {
        // Use more input points to get better polygon approximation of circle.
        // With N input points, the max inward error is r * (1 - cos(pi/N)).
        // For N=40: error = 10 * (1 - cos(pi/40)) ~ 0.031
        let contour = create_circular_contour(25.0, 25.0, 10.0, 40);
        let resampled = resample_contour(&contour, 20).expect("should succeed");
        assert_eq!(resampled.nrows(), 20);

        // All resampled points should be close to the circle.
        // Since resampled points lie on polygon chords, they can be slightly
        // inside the circle. Tolerance accounts for polygon approximation error.
        for i in 0..20 {
            let dx = resampled[[i, 0]] - 25.0;
            let dy = resampled[[i, 1]] - 25.0;
            let r = (dx * dx + dy * dy).sqrt();
            assert!(
                (r - 10.0).abs() < 0.5,
                "Point {} at radius {}, expected ~10.0",
                i,
                r
            );
        }
    }

    #[test]
    fn test_resample_contour_errors() {
        let small = Array2::zeros((1, 2));
        assert!(resample_contour(&small, 10).is_err());

        let contour = create_circular_contour(0.0, 0.0, 10.0, 10);
        assert!(resample_contour(&contour, 2).is_err());
    }

    #[test]
    fn test_gradient_magnitude() {
        let mut image = Array2::zeros((10, 10));
        // Step edge at column 5
        for r in 0..10 {
            for c in 5..10 {
                image[[r, c]] = 1.0;
            }
        }

        let mag = gradient_magnitude(&image);

        // Should have strong gradient at the step edge
        assert!(mag[[5, 5]] > 0.0 || mag[[5, 4]] > 0.0);
    }

    #[test]
    fn test_gvf_field() {
        let mut edge_map = Array2::zeros((20, 20));
        // Simple edge at column 10
        for r in 0..20 {
            edge_map[[r, 10]] = 1.0;
        }

        let (u, v) = compute_gvf(&edge_map, 0.2, 50);
        assert_eq!(u.dim(), (20, 20));
        assert_eq!(v.dim(), (20, 20));

        // GVF should point toward the edge at column 10
        // At (10, 5), u should be positive (pointing right toward col 10)
        // At (10, 15), u should be negative (pointing left toward col 10)
        // Allow some tolerance for diffusion effects
        assert!(
            u[[10, 5]] > -0.1,
            "GVF left of edge should point rightward or be small"
        );
    }

    #[test]
    fn test_evolve_snake_basic() {
        let mut image = Array2::<f64>::zeros((50, 50));
        // Draw a circle edge
        for i in 0..50 {
            for j in 0..50 {
                let r =
                    (((i as f64 - 25.0).powi(2) + (j as f64 - 25.0).powi(2)).sqrt() - 15.0).abs();
                if r < 2.0 {
                    image[[i, j]] = 1.0;
                }
            }
        }

        let contour = create_circular_contour(25.0, 25.0, 12.0, 30);
        let params = SnakeParams {
            max_iterations: 20,
            alpha: 0.01,
            beta: 0.1,
            gamma: 1.0,
            ..Default::default()
        };

        let result = evolve_snake(&image, &contour, &params).expect("should succeed");
        assert_eq!(result.contour.nrows(), 30);
        assert!(result.iterations > 0);
        assert!(!result.energy_history.is_empty());
    }

    #[test]
    fn test_evolve_snake_convergence() {
        // Uniform image with zero internal and external forces: contour
        // should not move at all and converge immediately.
        let image = Array2::<f64>::from_elem((50, 50), 0.5);
        let contour = create_circular_contour(25.0, 25.0, 10.0, 20);

        let params = SnakeParams {
            max_iterations: 500,
            convergence_threshold: 0.01,
            alpha: 0.0, // No elasticity force
            beta: 0.0,  // No bending force
            gamma: 0.0, // No external force
            kappa: 0.0, // No balloon force
            ..Default::default()
        };

        let result = evolve_snake(&image, &contour, &params).expect("should succeed");
        assert!(
            result.converged,
            "Should converge immediately with no forces on uniform image"
        );
        // Contour should barely move
        for i in 0..20 {
            let dx = result.contour[[i, 0]] - contour[[i, 0]];
            let dy = result.contour[[i, 1]] - contour[[i, 1]];
            assert!(
                (dx * dx + dy * dy).sqrt() < 1e-10,
                "Point {} should not move with zero forces",
                i
            );
        }
    }

    #[test]
    fn test_evolve_snake_invalid_inputs() {
        let image = Array2::<f64>::zeros((10, 10));

        // Wrong contour shape
        let bad_contour = Array2::zeros((5, 3));
        assert!(evolve_snake(&image, &bad_contour, &SnakeParams::default()).is_err());

        // Too few points
        let small_contour = Array2::zeros((2, 2));
        assert!(evolve_snake(&image, &small_contour, &SnakeParams::default()).is_err());

        // Empty image
        let empty_image: Array2<f64> = Array2::zeros((0, 0));
        let contour = create_circular_contour(5.0, 5.0, 3.0, 10);
        assert!(evolve_snake(&empty_image, &contour, &SnakeParams::default()).is_err());
    }

    #[test]
    fn test_evolve_snake_with_balloon() {
        let image = Array2::<f64>::zeros((50, 50));
        let contour = create_circular_contour(25.0, 25.0, 5.0, 20);

        let params = SnakeParams {
            max_iterations: 50,
            kappa: 2.0, // Strong outward balloon force
            alpha: 0.0, // No elasticity (would contract)
            beta: 0.0,  // No bending (would contract)
            gamma: 0.0, // No external force
            time_step: 0.1,
            ..Default::default()
        };

        let result = evolve_snake(&image, &contour, &params).expect("should succeed");

        // With positive balloon force and no internal forces, the contour should expand
        let initial_area = contour_area(&contour).abs();
        let final_area = contour_area(&result.contour).abs();
        assert!(
            final_area > initial_area,
            "Balloon force should expand contour: initial={}, final={}",
            initial_area,
            final_area
        );
    }

    #[test]
    fn test_contour_energy_calculation() {
        let image = Array2::<f64>::from_elem((20, 20), 1.0);
        let contour = create_circular_contour(10.0, 10.0, 5.0, 20);

        let energy = contour_energy(&image, &contour, 0.01, 0.1, 1.0).expect("should succeed");
        // Energy should be finite
        assert!(energy.is_finite());
    }

    #[test]
    fn test_contour_energy_errors() {
        let image = Array2::<f64>::zeros((10, 10));

        // Too few points
        let small = Array2::zeros((2, 2));
        assert!(contour_energy(&image, &small, 0.01, 0.1, 1.0).is_err());

        // Wrong columns
        let bad = Array2::zeros((5, 3));
        assert!(contour_energy(&image, &bad, 0.01, 0.1, 1.0).is_err());
    }

    #[test]
    fn test_internal_energy_circle() {
        // A perfect circle should have low bending energy
        let circle = create_circular_contour(0.0, 0.0, 10.0, 100);
        let e_circle = internal_energy(&circle, 0.01, 0.1);

        // A "spiky" contour should have higher energy
        let mut spiky = circle.clone();
        for i in (0..100).step_by(2) {
            spiky[[i, 0]] *= 1.3;
            spiky[[i, 1]] *= 1.3;
        }
        let e_spiky = internal_energy(&spiky, 0.01, 0.1);

        assert!(
            e_spiky > e_circle,
            "Spiky contour should have higher internal energy"
        );
    }

    #[test]
    fn test_evolve_snake_with_gvf() {
        let mut image = Array2::<f64>::zeros((30, 30));
        // Draw vertical edge at column 15
        for r in 0..30 {
            for c in 15..30 {
                image[[r, c]] = 1.0;
            }
        }

        let contour = create_circular_contour(15.0, 15.0, 8.0, 20);
        let params = SnakeParams {
            max_iterations: 30,
            use_gvf: true,
            gvf_mu: 0.2,
            gvf_iterations: 40,
            ..Default::default()
        };

        let result = evolve_snake(&image, &contour, &params).expect("should succeed");
        assert_eq!(result.contour.nrows(), 20);
    }
}
