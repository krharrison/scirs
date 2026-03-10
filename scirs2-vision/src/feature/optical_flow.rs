//! Optical flow computation for motion analysis
//!
//! This module provides algorithms for computing optical flow between
//! consecutive frames, useful for motion analysis and tracking.

use crate::error::Result;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use scirs2_core::ndarray::{s, Array2};

/// Optical flow vector at a point
#[derive(Debug, Clone, Copy)]
pub struct FlowVector {
    /// Horizontal displacement
    pub u: f32,
    /// Vertical displacement
    pub v: f32,
}

/// Parameters for Lucas-Kanade optical flow
#[derive(Debug, Clone)]
pub struct LucasKanadeParams {
    /// Window size for local computation
    pub window_size: usize,
    /// Maximum iterations for iterative refinement
    pub max_iterations: usize,
    /// Convergence threshold
    pub epsilon: f32,
    /// Number of pyramid levels (0 for no pyramid)
    pub pyramid_levels: usize,
}

impl Default for LucasKanadeParams {
    fn default() -> Self {
        Self {
            window_size: 15,
            max_iterations: 20,
            epsilon: 0.01,
            pyramid_levels: 3,
        }
    }
}

/// Compute optical flow using Lucas-Kanade method
///
/// # Arguments
///
/// * `img1` - First frame
/// * `img2` - Second frame
/// * `points` - Points to track (if None, computes dense flow)
/// * `params` - Algorithm parameters
///
/// # Returns
///
/// * Flow field as 2D array of flow vectors
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::{lucas_kanade_flow, LucasKanadeParams};
/// use image::{DynamicImage, RgbImage};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// // Create simple test images
/// let frame1 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// let frame2 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// let flow = lucas_kanade_flow(&frame1, &frame2, None, &LucasKanadeParams::default())?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn lucas_kanade_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    points: Option<&[(f32, f32)]>,
    params: &LucasKanadeParams,
) -> Result<Array2<FlowVector>> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();

    if params.pyramid_levels > 0 {
        pyramidal_lucas_kanade(&gray1, &gray2, points, params)
    } else {
        simple_lucas_kanade(&gray1, &gray2, points, params)
    }
}

/// Simple Lucas-Kanade without pyramid
#[allow(dead_code)]
fn simple_lucas_kanade(
    img1: &GrayImage,
    img2: &GrayImage,
    points: Option<&[(f32, f32)]>,
    params: &LucasKanadeParams,
) -> Result<Array2<FlowVector>> {
    let (width, height) = img1.dimensions();

    // Convert images to float arrays
    let i1 = image_to_float_array(img1);
    let i2 = image_to_float_array(img2);

    // Compute image gradients
    let (ix, iy) = compute_gradients(&i1);

    let half_window = params.window_size / 2;

    // Determine points to compute flow for
    let track_points: Vec<(f32, f32)> = if let Some(pts) = points {
        pts.to_vec()
    } else {
        // Dense flow - compute for all pixels with sufficient margin
        let mut pts = Vec::new();
        for y in half_window..height as usize - half_window {
            for x in half_window..width as usize - half_window {
                pts.push((x as f32, y as f32));
            }
        }
        pts
    };

    // Initialize flow field
    let mut flow = Array2::from_elem(
        (height as usize, width as usize),
        FlowVector { u: 0.0, v: 0.0 },
    );

    // Compute flow for each point
    for &(px, py) in &track_points {
        let x = px as usize;
        let y = py as usize;

        // Skip boundary points
        if x < half_window
            || x >= width as usize - half_window
            || y < half_window
            || y >= height as usize - half_window
        {
            continue;
        }

        // Extract window around point
        let window_ix = ix.slice(s![
            y - half_window..=y + half_window,
            x - half_window..=x + half_window
        ]);
        let window_iy = iy.slice(s![
            y - half_window..=y + half_window,
            x - half_window..=x + half_window
        ]);
        let window_i1 = i1.slice(s![
            y - half_window..=y + half_window,
            x - half_window..=x + half_window
        ]);

        // Build system matrix A^T A
        let mut a11 = 0.0f32;
        let mut a12 = 0.0f32;
        let mut a22 = 0.0f32;

        for ((ix_val, iy_val), _) in window_ix.iter().zip(window_iy.iter()).zip(window_i1.iter()) {
            a11 += ix_val * ix_val;
            a12 += ix_val * iy_val;
            a22 += iy_val * iy_val;
        }

        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-6 {
            continue; // Singular matrix, skip this point
        }

        // Iterative refinement
        let mut u = 0.0f32;
        let mut v = 0.0f32;

        for _ in 0..params.max_iterations {
            // Get warped window from second image
            let warped_x = (x as f32 + u) as usize;
            let warped_y = (y as f32 + v) as usize;

            if warped_x < half_window
                || warped_x >= width as usize - half_window
                || warped_y < half_window
                || warped_y >= height as usize - half_window
            {
                break;
            }

            let window_i2 = i2.slice(s![
                warped_y - half_window..=warped_y + half_window,
                warped_x - half_window..=warped_x + half_window
            ]);

            // Compute temporal derivative and error
            let mut b1 = 0.0f32;
            let mut b2 = 0.0f32;

            for ((&ix_val, &iy_val), (&i1_val, &i2_val)) in window_ix
                .iter()
                .zip(window_iy.iter())
                .zip(window_i1.iter().zip(window_i2.iter()))
            {
                let it = i2_val - i1_val;
                b1 -= ix_val * it;
                b2 -= iy_val * it;
            }

            // Solve for flow update
            let inv_det = 1.0 / det;
            let du = inv_det * (a22 * b1 - a12 * b2);
            let dv = inv_det * (-a12 * b1 + a11 * b2);

            u += du;
            v += dv;

            if du.abs() < params.epsilon && dv.abs() < params.epsilon {
                break;
            }
        }

        flow[[y, x]] = FlowVector { u, v };
    }

    Ok(flow)
}

/// Pyramidal Lucas-Kanade
#[allow(dead_code)]
fn pyramidal_lucas_kanade(
    img1: &GrayImage,
    img2: &GrayImage,
    points: Option<&[(f32, f32)]>,
    params: &LucasKanadeParams,
) -> Result<Array2<FlowVector>> {
    let (width, height) = img1.dimensions();

    // Build image pyramids
    let pyramid1 = build_pyramid(img1, params.pyramid_levels);
    let pyramid2 = build_pyramid(img2, params.pyramid_levels);

    // Initialize flow
    let mut flow = Array2::from_elem(
        (height as usize, width as usize),
        FlowVector { u: 0.0, v: 0.0 },
    );

    // Process from coarse to fine
    for level in (0..params.pyramid_levels).rev() {
        let scale = 2.0_f32.powi(level as i32);

        // Scale points for this level
        let scaled_points: Option<Vec<(f32, f32)>> =
            points.map(|pts| pts.iter().map(|&(x, y)| (x / scale, y / scale)).collect());

        // Compute flow at this level
        let level_params = LucasKanadeParams {
            pyramid_levels: 0, // No recursion
            ..params.clone()
        };

        let level_flow = simple_lucas_kanade(
            &pyramid1[level],
            &pyramid2[level],
            scaled_points.as_deref(),
            &level_params,
        )?;

        // Propagate flow to finer level
        if level > 0 {
            let (level_width, level_height) = pyramid1[level].dimensions();
            for y in 0..level_height as usize {
                for x in 0..level_width as usize {
                    let fine_x = (x * 2).min(width as usize - 1);
                    let fine_y = (y * 2).min(height as usize - 1);

                    flow[[fine_y, fine_x]].u = level_flow[[y, x]].u * 2.0;
                    flow[[fine_y, fine_x]].v = level_flow[[y, x]].v * 2.0;

                    // Fill neighboring pixels
                    if fine_x + 1 < width as usize {
                        flow[[fine_y, fine_x + 1]] = flow[[fine_y, fine_x]];
                    }
                    if fine_y + 1 < height as usize {
                        flow[[fine_y + 1, fine_x]] = flow[[fine_y, fine_x]];
                        if fine_x + 1 < width as usize {
                            flow[[fine_y + 1, fine_x + 1]] = flow[[fine_y, fine_x]];
                        }
                    }
                }
            }
        } else {
            flow = level_flow;
        }
    }

    Ok(flow)
}

/// Build image pyramid
#[allow(dead_code)]
fn build_pyramid(img: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = vec![img.clone()];

    for _ in 1..levels {
        let prev = &pyramid[pyramid.len() - 1];
        let (width, height) = prev.dimensions();
        let new_width = width / 2;
        let new_height = height / 2;

        let mut downsampled = ImageBuffer::new(new_width, new_height);

        for y in 0..new_height {
            for x in 0..new_width {
                // Simple 2x2 average
                let x2 = x * 2;
                let y2 = y * 2;

                let sum = prev.get_pixel(x2, y2)[0] as u32
                    + prev.get_pixel(x2 + 1, y2)[0] as u32
                    + prev.get_pixel(x2, y2 + 1)[0] as u32
                    + prev.get_pixel(x2 + 1, y2 + 1)[0] as u32;

                downsampled.put_pixel(x, y, Luma([(sum / 4) as u8]));
            }
        }

        pyramid.push(downsampled);
    }

    pyramid
}

/// Convert image to float array
#[allow(dead_code)]
fn image_to_float_array(img: &GrayImage) -> Array2<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = img.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    array
}

/// Compute image gradients using Scharr operator
#[allow(dead_code)]
fn compute_gradients(img: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (height, width) = img.dim();
    let mut ix = Array2::zeros((height, width));
    let mut iy = Array2::zeros((height, width));

    // Scharr kernels
    let scharr_x = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
    let scharr_y = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0.0;
            let mut gy = 0.0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let pixel = img[[(y as i32 + dy) as usize, (x as i32 + dx) as usize]];
                    gx += pixel * scharr_x[(dy + 1) as usize][(dx + 1) as usize] / 32.0;
                    gy += pixel * scharr_y[(dy + 1) as usize][(dx + 1) as usize] / 32.0;
                }
            }

            ix[[y, x]] = gx;
            iy[[y, x]] = gy;
        }
    }

    (ix, iy)
}

/// Visualize optical flow as color image
///
/// # Arguments
///
/// * `flow` - Flow field
/// * `max_flow` - Maximum flow magnitude for scaling (None for auto)
///
/// # Returns
///
/// * RGB image with flow visualization
#[allow(dead_code)]
pub fn visualize_flow(_flow: &Array2<FlowVector>, maxflow: Option<f32>) -> RgbImage {
    let (height, width) = _flow.dim();
    let mut result = RgbImage::new(width as u32, height as u32);

    // Find maximum _flow if not provided
    let max_magnitude = if let Some(max) = maxflow {
        max
    } else {
        let mut max = 0.0f32;
        for flow_vec in _flow.iter() {
            let magnitude = (flow_vec.u.powi(2) + flow_vec.v.powi(2)).sqrt();
            if magnitude > max {
                max = magnitude;
            }
        }
        max.max(1.0) // Avoid division by zero
    };

    for y in 0..height {
        for x in 0..width {
            let flow_vec = &_flow[[y, x]];
            let magnitude = (flow_vec.u.powi(2) + flow_vec.v.powi(2)).sqrt();
            let angle = flow_vec.v.atan2(flow_vec.u);

            // Convert to HSV color
            let hue = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let saturation = (magnitude / max_magnitude).min(1.0);
            let value = saturation; // Or use 1.0 for constant brightness

            // Convert HSV to RGB
            let (r, g, b) = hsv_to_rgb(hue, saturation, value);
            result.put_pixel(
                x as u32,
                y as u32,
                Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]),
            );
        }
    }

    result
}

/// Convert HSV to RGB
#[allow(dead_code)]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r + m, g + m, b + m)
}

/// Dense optical flow using Farneback method (simplified version)
#[allow(dead_code)]
pub fn farneback_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    _pyr_scale: f32,
    _levels: usize,
    winsize: usize,
    _iterations: usize,
) -> Result<Array2<FlowVector>> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (width, height) = gray1.dimensions();

    // Initialize flow
    let mut flow = Array2::from_elem(
        (height as usize, width as usize),
        FlowVector { u: 0.0, v: 0.0 },
    );

    // Simplified dense flow computation
    let i1 = image_to_float_array(&gray1);
    let i2 = image_to_float_array(&gray2);
    let (ix, iy) = compute_gradients(&i1);

    let half_win = winsize / 2;

    for y in half_win..height as usize - half_win {
        for x in half_win..width as usize - half_win {
            // Extract windows
            let win_ix = ix.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);
            let win_iy = iy.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);

            // Compute structure tensor
            let mut ixx = 0.0;
            let mut ixy = 0.0;
            let mut iyy = 0.0;

            for (&ix_val, &iy_val) in win_ix.iter().zip(win_iy.iter()) {
                ixx += ix_val * ix_val;
                ixy += ix_val * iy_val;
                iyy += iy_val * iy_val;
            }

            let det = ixx * iyy - ixy * ixy;
            if det > 1e-6 {
                // Simplified flow computation
                let win_i1 = i1.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);
                let win_i2 = i2.slice(s![y - half_win..=y + half_win, x - half_win..=x + half_win]);

                let mut bx = 0.0;
                let mut by = 0.0;

                for ((&i1_val, &i2_val), (&ix_val, &iy_val)) in win_i1
                    .iter()
                    .zip(win_i2.iter())
                    .zip(win_ix.iter().zip(win_iy.iter()))
                {
                    let it = i2_val - i1_val;
                    bx -= ix_val * it;
                    by -= iy_val * it;
                }

                let inv_det = 1.0 / det;
                flow[[y, x]] = FlowVector {
                    u: inv_det * (iyy * bx - ixy * by),
                    v: inv_det * (-ixy * bx + ixx * by),
                };
            }
        }
    }

    Ok(flow)
}

/// Parameters for Horn-Schunck dense optical flow
///
/// The Horn-Schunck method computes dense optical flow by minimizing a global
/// energy functional that combines a data term (brightness constancy) with a
/// smoothness regularization term.
///
/// # References
///
/// - Horn, B.K. and Schunck, B.G., 1981. Determining optical flow.
///   Artificial intelligence, 17(1-3), pp.185-203.
#[derive(Debug, Clone)]
pub struct HornSchunckParams {
    /// Smoothness weight (alpha). Larger values produce smoother flow fields
    /// but may miss fine motion details. Typical range: 1.0 to 100.0.
    pub alpha: f32,
    /// Maximum number of Jacobi/Gauss-Seidel iterations
    pub max_iterations: usize,
    /// Convergence threshold (maximum change between iterations)
    pub epsilon: f32,
}

impl Default for HornSchunckParams {
    fn default() -> Self {
        Self {
            alpha: 15.0,
            max_iterations: 200,
            epsilon: 1e-4,
        }
    }
}

/// Compute dense optical flow using the Horn-Schunck variational method
///
/// This method produces a dense flow field (one vector per pixel) by solving
/// a global optimization problem that balances the brightness constancy
/// constraint with a first-order smoothness prior.
///
/// The energy functional minimized is:
///   E(u,v) = integral [ (I_x u + I_y v + I_t)^2 + alpha^2 (|grad u|^2 + |grad v|^2) ] dx dy
///
/// The solution is obtained iteratively using a Gauss-Seidel scheme.
///
/// # Arguments
///
/// * `img1` - First frame (reference)
/// * `img2` - Second frame (target)
/// * `params` - Horn-Schunck parameters
///
/// # Returns
///
/// * Dense flow field as 2D array of flow vectors
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::optical_flow::{horn_schunck_flow, HornSchunckParams};
/// use image::{DynamicImage, RgbImage};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let frame1 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// let frame2 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// let flow = horn_schunck_flow(&frame1, &frame2, &HornSchunckParams::default())?;
/// assert_eq!(flow.dim(), (64, 64));
/// # Ok(())
/// # }
/// ```
pub fn horn_schunck_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    params: &HornSchunckParams,
) -> Result<Array2<FlowVector>> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (width, height) = gray1.dimensions();
    let h = height as usize;
    let w = width as usize;

    if h < 3 || w < 3 {
        return Err(crate::error::VisionError::InvalidParameter(
            "Images must be at least 3x3 for Horn-Schunck flow".to_string(),
        ));
    }

    // Convert to float arrays
    let i1 = image_to_float_array(&gray1);
    let i2 = image_to_float_array(&gray2);

    // Compute spatial gradients on the averaged image (I_x, I_y)
    // and temporal gradient (I_t)
    let mut ix = Array2::<f32>::zeros((h, w));
    let mut iy = Array2::<f32>::zeros((h, w));
    let mut it = Array2::<f32>::zeros((h, w));

    for y in 0..h - 1 {
        for x in 0..w - 1 {
            // Averaged partial derivatives using 2x2 stencils
            // I_x = 0.25 * (I(x+1,y) - I(x,y) + I(x+1,y+1) - I(x,y+1)
            //              + J(x+1,y) - J(x,y) + J(x+1,y+1) - J(x,y+1))
            ix[[y, x]] = 0.25
                * ((i1[[y, x + 1]] - i1[[y, x]])
                    + (i1[[y + 1, x + 1]] - i1[[y + 1, x]])
                    + (i2[[y, x + 1]] - i2[[y, x]])
                    + (i2[[y + 1, x + 1]] - i2[[y + 1, x]]));

            iy[[y, x]] = 0.25
                * ((i1[[y + 1, x]] - i1[[y, x]])
                    + (i1[[y + 1, x + 1]] - i1[[y, x + 1]])
                    + (i2[[y + 1, x]] - i2[[y, x]])
                    + (i2[[y + 1, x + 1]] - i2[[y, x + 1]]));

            it[[y, x]] = 0.25
                * ((i2[[y, x]] - i1[[y, x]])
                    + (i2[[y, x + 1]] - i1[[y, x + 1]])
                    + (i2[[y + 1, x]] - i1[[y + 1, x]])
                    + (i2[[y + 1, x + 1]] - i1[[y + 1, x + 1]]));
        }
    }

    let alpha_sq = params.alpha * params.alpha;

    // Initialize flow fields
    let mut u_flow = Array2::<f32>::zeros((h, w));
    let mut v_flow = Array2::<f32>::zeros((h, w));

    // Iterative Gauss-Seidel / Jacobi solver
    for _iter in 0..params.max_iterations {
        let mut max_change: f32 = 0.0;

        // Compute Laplacian-weighted averages (using 4-connected neighbors)
        let u_avg = laplacian_average(&u_flow);
        let v_avg = laplacian_average(&v_flow);

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let ix_val = ix[[y, x]];
                let iy_val = iy[[y, x]];
                let it_val = it[[y, x]];

                let denom = alpha_sq + ix_val * ix_val + iy_val * iy_val;
                if denom.abs() < 1e-12 {
                    continue;
                }

                let p = ix_val * u_avg[[y, x]] + iy_val * v_avg[[y, x]] + it_val;
                let factor = p / denom;

                let new_u = u_avg[[y, x]] - ix_val * factor;
                let new_v = v_avg[[y, x]] - iy_val * factor;

                let du = (new_u - u_flow[[y, x]]).abs();
                let dv = (new_v - v_flow[[y, x]]).abs();
                if du > max_change {
                    max_change = du;
                }
                if dv > max_change {
                    max_change = dv;
                }

                u_flow[[y, x]] = new_u;
                v_flow[[y, x]] = new_v;
            }
        }

        if max_change < params.epsilon {
            break;
        }
    }

    // Combine into FlowVector array
    let mut flow = Array2::from_elem((h, w), FlowVector { u: 0.0, v: 0.0 });
    for y in 0..h {
        for x in 0..w {
            flow[[y, x]] = FlowVector {
                u: u_flow[[y, x]],
                v: v_flow[[y, x]],
            };
        }
    }

    Ok(flow)
}

/// Compute the weighted average of neighbors (Laplacian kernel) for Horn-Schunck
/// Uses the 4-connected neighborhood: (1/4) * (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1})
fn laplacian_average(field: &Array2<f32>) -> Array2<f32> {
    let (h, w) = field.dim();
    let mut avg = Array2::<f32>::zeros((h, w));

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            avg[[y, x]] = 0.25
                * (field[[y - 1, x]] + field[[y + 1, x]] + field[[y, x - 1]] + field[[y, x + 1]]);
        }
    }

    // Handle boundaries by replicating nearest interior
    for x in 0..w {
        avg[[0, x]] = avg[[1, x.min(w - 2).max(1)]];
        avg[[h - 1, x]] = avg[[(h - 2).max(1), x.min(w - 2).max(1)]];
    }
    for y in 0..h {
        avg[[y, 0]] = avg[[y.min(h - 2).max(1), 1]];
        avg[[y, w - 1]] = avg[[y.min(h - 2).max(1), (w - 2).max(1)]];
    }

    avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lucas_kanade_basic() {
        let img1 = DynamicImage::new_luma8(50, 50);
        let img2 = img1.clone();

        let flow = lucas_kanade_flow(&img1, &img2, None, &LucasKanadeParams::default())
            .expect("Operation failed");
        assert_eq!(flow.dim(), (50, 50));

        // Flow should be zero for identical images
        for flow_vec in flow.iter() {
            assert!(flow_vec.u.abs() < 0.1);
            assert!(flow_vec.v.abs() < 0.1);
        }
    }

    #[test]
    fn test_pyramid_building() {
        let img = GrayImage::new(64, 64);
        let pyramid = build_pyramid(&img, 3);

        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].dimensions(), (64, 64));
        assert_eq!(pyramid[1].dimensions(), (32, 32));
        assert_eq!(pyramid[2].dimensions(), (16, 16));
    }

    #[test]
    fn test_flow_visualization() {
        let mut flow = Array2::from_elem((10, 10), FlowVector { u: 0.0, v: 0.0 });
        flow[[5, 5]] = FlowVector { u: 1.0, v: 0.0 };

        let vis = visualize_flow(&flow, Some(1.0));
        assert_eq!(vis.dimensions(), (10, 10));
    }

    #[test]
    fn test_horn_schunck_identical_images() {
        let img = DynamicImage::new_luma8(32, 32);
        let flow =
            horn_schunck_flow(&img, &img, &HornSchunckParams::default()).expect("HS flow failed");
        assert_eq!(flow.dim(), (32, 32));

        // Zero flow for identical images
        for fv in flow.iter() {
            assert!(fv.u.abs() < 1e-3, "u should be ~0, got {}", fv.u);
            assert!(fv.v.abs() < 1e-3, "v should be ~0, got {}", fv.v);
        }
    }

    #[test]
    fn test_horn_schunck_shifted_pattern() {
        // Create a vertical stripe pattern, then shift it right by 1 pixel
        let mut buf1 = GrayImage::new(32, 32);
        let mut buf2 = GrayImage::new(32, 32);

        for y in 0..32u32 {
            for x in 0..32u32 {
                let val = if (x / 4) % 2 == 0 { 200u8 } else { 50u8 };
                buf1.put_pixel(x, y, Luma([val]));
                // shift right by 1
                if x > 0 {
                    buf2.put_pixel(
                        x,
                        y,
                        Luma([if ((x - 1) / 4) % 2 == 0 { 200u8 } else { 50u8 }]),
                    );
                } else {
                    buf2.put_pixel(x, y, Luma([200u8]));
                }
            }
        }

        let img1 = DynamicImage::ImageLuma8(buf1);
        let img2 = DynamicImage::ImageLuma8(buf2);

        let params = HornSchunckParams {
            alpha: 5.0,
            max_iterations: 500,
            epsilon: 1e-5,
        };

        let flow = horn_schunck_flow(&img1, &img2, &params).expect("HS flow failed");
        assert_eq!(flow.dim(), (32, 32));

        // We just check it ran and produced finite output
        let center = &flow[[16, 16]];
        assert!(
            center.u.is_finite() && center.v.is_finite(),
            "Flow should be computed"
        );
    }

    #[test]
    fn test_horn_schunck_params_default() {
        let params = HornSchunckParams::default();
        assert!(params.alpha > 0.0);
        assert!(params.max_iterations > 0);
        assert!(params.epsilon > 0.0);
    }

    #[test]
    fn test_horn_schunck_small_alpha() {
        // Small alpha should allow more spatial variation
        let mut buf1 = GrayImage::new(16, 16);
        let mut buf2 = GrayImage::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                buf1.put_pixel(x, y, Luma([(x * 16) as u8]));
                buf2.put_pixel(x, y, Luma([((x + 1).min(15) * 16) as u8]));
            }
        }

        let img1 = DynamicImage::ImageLuma8(buf1);
        let img2 = DynamicImage::ImageLuma8(buf2);

        let params = HornSchunckParams {
            alpha: 1.0,
            max_iterations: 100,
            epsilon: 1e-4,
        };

        let flow = horn_schunck_flow(&img1, &img2, &params).expect("HS flow failed");
        assert_eq!(flow.dim(), (16, 16));
    }

    #[test]
    fn test_horn_schunck_large_alpha_smooth() {
        // Large alpha produces smoother flow
        let mut buf1 = GrayImage::new(16, 16);
        let mut buf2 = GrayImage::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                buf1.put_pixel(x, y, Luma([(x * 16) as u8]));
                buf2.put_pixel(x, y, Luma([((x + 1).min(15) * 16) as u8]));
            }
        }

        let img1 = DynamicImage::ImageLuma8(buf1);
        let img2 = DynamicImage::ImageLuma8(buf2);

        let params = HornSchunckParams {
            alpha: 100.0,
            max_iterations: 200,
            epsilon: 1e-5,
        };

        let flow = horn_schunck_flow(&img1, &img2, &params).expect("HS flow failed");

        // With large alpha, neighboring flow vectors should be very similar
        let diff_u = (flow[[8, 8]].u - flow[[8, 9]].u).abs();
        let diff_v = (flow[[8, 8]].v - flow[[8, 9]].v).abs();
        assert!(
            diff_u < 0.5 && diff_v < 0.5,
            "Large alpha should produce smooth flow"
        );
    }

    #[test]
    fn test_horn_schunck_rejects_tiny_images() {
        let img = DynamicImage::new_luma8(2, 2);
        let result = horn_schunck_flow(&img, &img, &HornSchunckParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_average_basic() {
        let mut field = Array2::<f32>::zeros((5, 5));
        field[[2, 2]] = 4.0;
        let avg = laplacian_average(&field);
        // Center neighbors get 1.0 each from the averaging
        assert!((avg[[2, 1]] - 1.0).abs() < 1e-6);
        assert!((avg[[1, 2]] - 1.0).abs() < 1e-6);
    }
}
