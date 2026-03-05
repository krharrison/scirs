//! Dense optical flow algorithms.
//!
//! This module provides pixel-level motion estimation between consecutive frames,
//! returning dense vector fields (u, v) that describe the displacement of every
//! pixel from frame 1 to frame 2.
//!
//! # Algorithms
//!
//! - **Horn-Schunck** -- global variational approach with smoothness regularisation
//! - **TV-L1** -- total-variation regularised L1 data fidelity (robust to occlusions)
//! - **Coarse-to-fine (pyramid)** -- multi-scale warping wrapper for any flow method
//!
//! # Utilities
//!
//! - `warp_image` -- apply a (u, v) flow field to warp an image using bilinear interpolation
//! - `flow_to_color` -- HSV colour-coding of a 2D flow field

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{s, Array2, Array3};

// ---------------------------------------------------------------------------
// Horn-Schunck optical flow
// ---------------------------------------------------------------------------

/// Compute the Horn-Schunck dense optical flow between two single-channel frames.
///
/// The method minimises the energy
/// `E(u,v) = ∫∫ (Ix·u + Iy·v + It)² + α²(|∇u|² + |∇v|²) dxdy`
/// using a Gauss-Seidel / Jacobi iterative scheme.
///
/// # Arguments
///
/// * `frame1` – source frame (pixel values in `[0, 1]`)
/// * `frame2` – target frame, same shape
/// * `alpha`  – smoothness weight (larger → smoother field, e.g. 1.0)
/// * `max_iter` – maximum number of iterations (e.g. 100)
///
/// # Returns
///
/// `(u, v)` where `u` is the column displacement and `v` is the row displacement.
pub fn horn_schunck(
    frame1: &Array2<f64>,
    frame2: &Array2<f64>,
    alpha: f64,
    max_iter: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let shape = frame1.dim();
    if shape != frame2.dim() {
        return Err(VisionError::DimensionMismatch(
            "horn_schunck: frames must have identical shapes".into(),
        ));
    }
    if alpha <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "horn_schunck: alpha must be positive".into(),
        ));
    }

    let (rows, cols) = shape;
    let alpha2 = alpha * alpha;

    // Spatial and temporal gradients (central differences with border clamping).
    let ix = spatial_gradient_x(frame1)?;
    let iy = spatial_gradient_y(frame1)?;
    let it = temporal_gradient(frame1, frame2);

    let mut u = Array2::<f64>::zeros((rows, cols));
    let mut v = Array2::<f64>::zeros((rows, cols));

    for _ in 0..max_iter {
        // Local averages (4-connectivity Laplacian kernel).
        let u_avg = laplacian_avg(&u);
        let v_avg = laplacian_avg(&v);

        // Update rule.
        for r in 0..rows {
            for c in 0..cols {
                let ix_rc = ix[[r, c]];
                let iy_rc = iy[[r, c]];
                let it_rc = it[[r, c]];
                let u_a = u_avg[[r, c]];
                let v_a = v_avg[[r, c]];

                let denom = alpha2 + ix_rc * ix_rc + iy_rc * iy_rc;
                let num = ix_rc * u_a + iy_rc * v_a + it_rc;
                let factor = num / denom;

                u[[r, c]] = u_a - ix_rc * factor;
                v[[r, c]] = v_a - iy_rc * factor;
            }
        }
    }

    Ok((u, v))
}

// ---------------------------------------------------------------------------
// TV-L1 optical flow
// ---------------------------------------------------------------------------

/// Compute TV-L1 optical flow between two single-channel frames.
///
/// TV-L1 replaces the quadratic data term of Horn-Schunck with an L1 norm
/// (robust to outliers / occlusions) and total-variation regularisation.
/// The primal-dual scheme follows Zach, Pock & Bischof (2007).
///
/// # Arguments
///
/// * `frame1`  – source frame in `[0, 1]`
/// * `frame2`  – target frame in `[0, 1]`, same shape
/// * `lambda`  – data fidelity weight (higher → flow follows brightness more)
/// * `tau`     – primal step size (e.g. 0.25)
/// * `theta`   – coupling parameter between u and v̄ (e.g. 0.3)
/// * `n_iter`  – number of outer iterations (e.g. 30)
pub fn tvl1_flow(
    frame1: &Array2<f64>,
    frame2: &Array2<f64>,
    lambda: f64,
    tau: f64,
    theta: f64,
    n_iter: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let shape = frame1.dim();
    if shape != frame2.dim() {
        return Err(VisionError::DimensionMismatch(
            "tvl1_flow: frames must have identical shapes".into(),
        ));
    }
    if lambda <= 0.0 || tau <= 0.0 || theta <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "tvl1_flow: lambda, tau, and theta must be positive".into(),
        ));
    }

    let (rows, cols) = shape;

    // Pre-compute image gradients of frame1.
    let ix = spatial_gradient_x(frame1)?;
    let iy = spatial_gradient_y(frame1)?;

    // Initialise flow and auxiliary variables.
    let mut u = Array2::<f64>::zeros((rows, cols));
    let mut v = Array2::<f64>::zeros((rows, cols));
    // Dual variables for TV regularisation.
    let mut p1 = Array2::<f64>::zeros((rows, cols)); // px for u
    let mut p2 = Array2::<f64>::zeros((rows, cols)); // py for u
    let mut q1 = Array2::<f64>::zeros((rows, cols)); // px for v
    let mut q2 = Array2::<f64>::zeros((rows, cols)); // py for v

    let lt = lambda * theta;

    for _ in 0..n_iter {
        // --- Dual ascent step (TV regularisation via gradient of u and v) ---
        for r in 0..rows {
            for c in 0..cols {
                // Forward differences for u.
                let du_dx = if c + 1 < cols {
                    u[[r, c + 1]] - u[[r, c]]
                } else {
                    0.0
                };
                let du_dy = if r + 1 < rows {
                    u[[r + 1, c]] - u[[r, c]]
                } else {
                    0.0
                };
                let dv_dx = if c + 1 < cols {
                    v[[r, c + 1]] - v[[r, c]]
                } else {
                    0.0
                };
                let dv_dy = if r + 1 < rows {
                    v[[r + 1, c]] - v[[r, c]]
                } else {
                    0.0
                };

                let new_p1 = p1[[r, c]] + tau / theta * du_dx;
                let new_p2 = p2[[r, c]] + tau / theta * du_dy;
                let new_q1 = q1[[r, c]] + tau / theta * dv_dx;
                let new_q2 = q2[[r, c]] + tau / theta * dv_dy;

                // Project onto unit ball (TV reprojection).
                let norm_p = (new_p1 * new_p1 + new_p2 * new_p2).sqrt().max(1.0);
                let norm_q = (new_q1 * new_q1 + new_q2 * new_q2).sqrt().max(1.0);

                p1[[r, c]] = new_p1 / norm_p;
                p2[[r, c]] = new_p2 / norm_p;
                q1[[r, c]] = new_q1 / norm_q;
                q2[[r, c]] = new_q2 / norm_q;
            }
        }

        // --- Primal descent (data fidelity update via thresholding) ---
        let u_old = u.clone();
        let v_old = v.clone();

        for r in 0..rows {
            for c in 0..cols {
                // Divergence of dual variable.
                let div_p = divergence_at(&p1, &p2, r, c, rows, cols);
                let div_q = divergence_at(&q1, &q2, r, c, rows, cols);

                let u_tilde = u_old[[r, c]] + theta * div_p;
                let v_tilde = v_old[[r, c]] + theta * div_q;

                // Warped frame2 value at (r + v_tilde, c + u_tilde).
                let warped_val = bilinear_sample(frame2, r as f64 + v_tilde, c as f64 + u_tilde);

                // Linearised brightness constraint.
                let rho = warped_val - frame1[[r, c]]
                    + ix[[r, c]] * (u_tilde - u_old[[r, c]])
                    + iy[[r, c]] * (v_tilde - v_old[[r, c]]);

                let grad_norm2 = ix[[r, c]] * ix[[r, c]] + iy[[r, c]] * iy[[r, c]] + 1e-9;

                // Soft-thresholding.
                let (du, dv) = if rho < -lt * grad_norm2 {
                    (lt * ix[[r, c]], lt * iy[[r, c]])
                } else if rho > lt * grad_norm2 {
                    (-lt * ix[[r, c]], -lt * iy[[r, c]])
                } else {
                    (
                        -rho * ix[[r, c]] / grad_norm2,
                        -rho * iy[[r, c]] / grad_norm2,
                    )
                };

                u[[r, c]] = u_tilde + du;
                v[[r, c]] = v_tilde + dv;
            }
        }
    }

    Ok((u, v))
}

// ---------------------------------------------------------------------------
// Coarse-to-fine pyramid flow
// ---------------------------------------------------------------------------

/// Coarse-to-fine multi-scale optical flow via image pyramid warping.
///
/// Builds a Gaussian image pyramid with `n_levels` levels and runs `flow_fn`
/// at the coarsest level, then iteratively refines the flow by:
///  1. Up-scaling the current flow estimate.
///  2. Warping `frame1` with the up-scaled flow.
///  3. Running `flow_fn` on the residual between the warped frame1 and frame2.
///  4. Accumulating the residual flow.
///
/// # Arguments
///
/// * `frame1`   – source frame
/// * `frame2`   – target frame
/// * `n_levels` – number of pyramid levels (e.g. 3–5)
/// * `flow_fn`  – closure that takes `(&Array2<f64>, &Array2<f64>)` and returns
///   `Result<(Array2<f64>, Array2<f64>)>`
pub fn coarse_to_fine_flow<F>(
    frame1: &Array2<f64>,
    frame2: &Array2<f64>,
    n_levels: usize,
    flow_fn: F,
) -> Result<(Array2<f64>, Array2<f64>)>
where
    F: Fn(&Array2<f64>, &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)>,
{
    if frame1.dim() != frame2.dim() {
        return Err(VisionError::DimensionMismatch(
            "coarse_to_fine_flow: frames must have identical shapes".into(),
        ));
    }
    if n_levels == 0 {
        return Err(VisionError::InvalidParameter(
            "coarse_to_fine_flow: n_levels must be at least 1".into(),
        ));
    }

    // Build pyramids.
    let pyr1 = build_pyramid(frame1, n_levels);
    let pyr2 = build_pyramid(frame2, n_levels);

    // Start at the coarsest level.
    let (mut u, mut v) = flow_fn(&pyr1[n_levels - 1], &pyr2[n_levels - 1])?;

    // Refine from coarse to fine.
    for lvl in (0..n_levels - 1).rev() {
        let (rows_fine, cols_fine) = pyr1[lvl].dim();

        // Up-scale flow.
        u = upsample_flow(&u, rows_fine, cols_fine);
        v = upsample_flow(&v, rows_fine, cols_fine);

        // Scale flow magnitudes (each level halves the resolution).
        u.mapv_inplace(|x| x * 2.0);
        v.mapv_inplace(|x| x * 2.0);

        // Warp frame1 at this level by the current flow estimate.
        let warped = warp_image(&pyr1[lvl], &u, &v)?;

        // Compute residual flow on the warped pair.
        let (du, dv) = flow_fn(&warped, &pyr2[lvl])?;

        // Accumulate.
        u = u + du;
        v = v + dv;
    }

    Ok((u, v))
}

// ---------------------------------------------------------------------------
// Image warping
// ---------------------------------------------------------------------------

/// Warp an image using a dense (u, v) displacement field via bilinear interpolation.
///
/// For each output pixel `(r, c)` the source sample is taken from
/// `(r + v[r,c], c + u[r,c])`.  Out-of-bounds samples are clamped to the
/// image border.
pub fn warp_image(image: &Array2<f64>, u: &Array2<f64>, v: &Array2<f64>) -> Result<Array2<f64>> {
    let shape = image.dim();
    if u.dim() != shape || v.dim() != shape {
        return Err(VisionError::DimensionMismatch(
            "warp_image: image, u, and v must have identical shapes".into(),
        ));
    }

    let (rows, cols) = shape;
    let mut output = Array2::<f64>::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let src_r = r as f64 + v[[r, c]];
            let src_c = c as f64 + u[[r, c]];
            output[[r, c]] = bilinear_sample(image, src_r, src_c);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Flow colour coding
// ---------------------------------------------------------------------------

/// Convert a 2D optical flow field to an HSV colour image (Baker et al. 2011).
///
/// The hue encodes the flow direction (angle), and the saturation/value
/// encodes the magnitude (normalised to the maximum observed magnitude).
/// Returns an `Array3<u8>` with shape `[rows, cols, 3]` (RGB).
pub fn flow_to_color(u: &Array2<f64>, v: &Array2<f64>) -> Result<Array3<u8>> {
    if u.dim() != v.dim() {
        return Err(VisionError::DimensionMismatch(
            "flow_to_color: u and v must have identical shapes".into(),
        ));
    }

    let (rows, cols) = u.dim();

    // Find maximum magnitude for normalisation.
    let mut max_mag: f64 = 1e-6;
    for r in 0..rows {
        for c in 0..cols {
            let mag = (u[[r, c]] * u[[r, c]] + v[[r, c]] * v[[r, c]]).sqrt();
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }

    let mut rgb = Array3::<u8>::zeros((rows, cols, 3));

    for r in 0..rows {
        for c in 0..cols {
            let fu = u[[r, c]];
            let fv = v[[r, c]];
            let mag = (fu * fu + fv * fv).sqrt();
            let angle = fv.atan2(fu); // radians in (-π, π]

            // Map angle to hue in [0, 1].
            let hue = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
            let sat = 1.0_f64;
            let val = (mag / max_mag).min(1.0);

            let (red, green, blue) = hsv_to_rgb(hue, sat, val);
            rgb[[r, c, 0]] = red;
            rgb[[r, c, 1]] = green;
            rgb[[r, c, 2]] = blue;
        }
    }

    Ok(rgb)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Spatial gradient in the x (column) direction via central differences.
fn spatial_gradient_x(frame: &Array2<f64>) -> Result<Array2<f64>> {
    let (rows, cols) = frame.dim();
    let mut gx = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let left = if c > 0 {
                frame[[r, c - 1]]
            } else {
                frame[[r, c]]
            };
            let right = if c + 1 < cols {
                frame[[r, c + 1]]
            } else {
                frame[[r, c]]
            };
            gx[[r, c]] = (right - left) * 0.5;
        }
    }
    Ok(gx)
}

/// Spatial gradient in the y (row) direction via central differences.
fn spatial_gradient_y(frame: &Array2<f64>) -> Result<Array2<f64>> {
    let (rows, cols) = frame.dim();
    let mut gy = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let top = if r > 0 {
                frame[[r - 1, c]]
            } else {
                frame[[r, c]]
            };
            let bot = if r + 1 < rows {
                frame[[r + 1, c]]
            } else {
                frame[[r, c]]
            };
            gy[[r, c]] = (bot - top) * 0.5;
        }
    }
    Ok(gy)
}

/// Temporal gradient: simple difference frame2 - frame1.
fn temporal_gradient(frame1: &Array2<f64>, frame2: &Array2<f64>) -> Array2<f64> {
    frame2 - frame1
}

/// Local 4-connectivity average (used as Horn-Schunck Laplacian).
fn laplacian_avg(arr: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = arr.dim();
    let mut avg = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let top = if r > 0 { arr[[r - 1, c]] } else { arr[[r, c]] };
            let bot = if r + 1 < rows {
                arr[[r + 1, c]]
            } else {
                arr[[r, c]]
            };
            let left = if c > 0 { arr[[r, c - 1]] } else { arr[[r, c]] };
            let right = if c + 1 < cols {
                arr[[r, c + 1]]
            } else {
                arr[[r, c]]
            };
            // Weighted average: corners get weight 1/12, edges 1/6.
            let tl = if r > 0 && c > 0 {
                arr[[r - 1, c - 1]]
            } else {
                arr[[r, c]]
            };
            let tr = if r > 0 && c + 1 < cols {
                arr[[r - 1, c + 1]]
            } else {
                arr[[r, c]]
            };
            let bl = if r + 1 < rows && c > 0 {
                arr[[r + 1, c - 1]]
            } else {
                arr[[r, c]]
            };
            let br = if r + 1 < rows && c + 1 < cols {
                arr[[r + 1, c + 1]]
            } else {
                arr[[r, c]]
            };
            avg[[r, c]] = (top + bot + left + right) / 6.0 + (tl + tr + bl + br) / 12.0;
        }
    }
    avg
}

/// Divergence of a vector field (p1 = x-component, p2 = y-component) at (r, c).
fn divergence_at(
    p1: &Array2<f64>,
    p2: &Array2<f64>,
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
) -> f64 {
    // Backward differences.
    let dp1_dx = p1[[r, c]] - if c > 0 { p1[[r, c - 1]] } else { 0.0 };
    let dp2_dy = p2[[r, c]] - if r > 0 { p2[[r - 1, c]] } else { 0.0 };
    // Suppress unused variable warning
    let _ = (rows, cols);
    dp1_dx + dp2_dy
}

/// Bilinear interpolation from `image` at floating-point coordinates `(fr, fc)`.
/// Clamps to image borders.
fn bilinear_sample(image: &Array2<f64>, fr: f64, fc: f64) -> f64 {
    let (rows, cols) = image.dim();
    let fr = fr.clamp(0.0, (rows - 1) as f64);
    let fc = fc.clamp(0.0, (cols - 1) as f64);

    let r0 = fr.floor() as usize;
    let c0 = fc.floor() as usize;
    let r1 = (r0 + 1).min(rows - 1);
    let c1 = (c0 + 1).min(cols - 1);

    let dr = fr - r0 as f64;
    let dc = fc - c0 as f64;

    let top = image[[r0, c0]] * (1.0 - dc) + image[[r0, c1]] * dc;
    let bot = image[[r1, c0]] * (1.0 - dc) + image[[r1, c1]] * dc;
    top * (1.0 - dr) + bot * dr
}

/// Build a Gaussian image pyramid with `n_levels` levels (level 0 = original).
fn build_pyramid(frame: &Array2<f64>, n_levels: usize) -> Vec<Array2<f64>> {
    let mut pyramid = Vec::with_capacity(n_levels);
    pyramid.push(frame.to_owned());
    for i in 1..n_levels {
        let prev = &pyramid[i - 1];
        let (pr, pc) = prev.dim();
        let nr = pr.div_ceil(2);
        let nc = pc.div_ceil(2);
        let mut down = Array2::<f64>::zeros((nr, nc));
        for r in 0..nr {
            for c in 0..nc {
                // Simple 2x2 average downsampling.
                let r2 = (r * 2).min(pr - 1);
                let c2 = (c * 2).min(pc - 1);
                let r2b = (r2 + 1).min(pr - 1);
                let c2b = (c2 + 1).min(pc - 1);
                down[[r, c]] =
                    (prev[[r2, c2]] + prev[[r2, c2b]] + prev[[r2b, c2]] + prev[[r2b, c2b]]) / 4.0;
            }
        }
        pyramid.push(down);
    }
    pyramid
}

/// Upsample a flow field to `(target_rows, target_cols)` using nearest-neighbour.
fn upsample_flow(flow: &Array2<f64>, target_rows: usize, target_cols: usize) -> Array2<f64> {
    let (src_rows, src_cols) = flow.dim();
    let mut up = Array2::<f64>::zeros((target_rows, target_cols));
    for r in 0..target_rows {
        for c in 0..target_cols {
            let sr = ((r * src_rows) / target_rows).min(src_rows - 1);
            let sc = ((c * src_cols) / target_cols).min(src_cols - 1);
            up[[r, c]] = flow[[sr, sc]];
        }
    }
    up
}

/// Convert HSV (all in [0, 1]) to RGB `(u8, u8, u8)`.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h6 = h * 6.0;
    let i = h6.floor() as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    (
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_frame(rows: usize, cols: usize, val: f64) -> Array2<f64> {
        Array2::from_elem((rows, cols), val)
    }

    #[test]
    fn horn_schunck_identical_frames_zero_flow() {
        let f = make_frame(16, 16, 0.5);
        let (u, v) = horn_schunck(&f, &f, 1.0, 50).expect("horn_schunck failed");
        for &x in u.iter().chain(v.iter()) {
            assert!(x.abs() < 1e-10, "expected zero flow, got {x}");
        }
    }

    #[test]
    fn tvl1_identical_frames_zero_flow() {
        let f = make_frame(16, 16, 0.5);
        let (u, v) = tvl1_flow(&f, &f, 1.0, 0.25, 0.3, 10).expect("tvl1_flow failed");
        for &x in u.iter().chain(v.iter()) {
            assert!(x.abs() < 1e-9, "expected zero flow, got {x}");
        }
    }

    #[test]
    fn warp_image_zero_flow_identity() {
        let mut f = Array2::<f64>::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                f[[i, j]] = (i + j) as f64 / 14.0;
            }
        }
        let u = Array2::zeros((8, 8));
        let v = Array2::zeros((8, 8));
        let warped = warp_image(&f, &u, &v).expect("warp failed");
        for (a, b) in f.iter().zip(warped.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn flow_to_color_shape() {
        let u = Array2::zeros((10, 12));
        let v = Array2::zeros((10, 12));
        let rgb = flow_to_color(&u, &v).expect("flow_to_color failed");
        assert_eq!(rgb.dim(), (10, 12, 3));
    }

    #[test]
    fn coarse_to_fine_shape_preserved() {
        let f1 = make_frame(32, 32, 0.3);
        let f2 = make_frame(32, 32, 0.4);
        let (u, v) = coarse_to_fine_flow(&f1, &f2, 3, |a, b| horn_schunck(a, b, 1.0, 5))
            .expect("coarse_to_fine failed");
        assert_eq!(u.dim(), (32, 32));
        assert_eq!(v.dim(), (32, 32));
    }

    #[test]
    fn horn_schunck_dimension_mismatch_error() {
        let f1 = make_frame(8, 8, 0.0);
        let f2 = make_frame(8, 9, 0.0);
        assert!(horn_schunck(&f1, &f2, 1.0, 5).is_err());
    }
}
