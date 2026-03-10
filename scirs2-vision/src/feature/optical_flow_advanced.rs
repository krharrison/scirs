//! Advanced optical flow algorithms
//!
//! This module implements advanced dense optical flow methods including
//! Farneback, TVL1, and DualTVL1 algorithms for motion estimation.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage};
use scirs2_core::ndarray::{s, Array2, Array3};

/// Dense optical flow field
#[derive(Debug, Clone)]
pub struct DenseFlow {
    /// Horizontal flow component (u)
    pub u: Array2<f32>,
    /// Vertical flow component (v)
    pub v: Array2<f32>,
}

impl DenseFlow {
    /// Create a new dense flow field
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            u: Array2::zeros((height, width)),
            v: Array2::zeros((height, width)),
        }
    }

    /// Compute flow magnitude at each pixel
    pub fn magnitude(&self) -> Array2<f32> {
        let (height, width) = self.u.dim();
        let mut mag = Array2::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                let u = self.u[[y, x]];
                let v = self.v[[y, x]];
                mag[[y, x]] = (u * u + v * v).sqrt();
            }
        }

        mag
    }

    /// Compute flow angle at each pixel
    pub fn angle(&self) -> Array2<f32> {
        let (height, width) = self.u.dim();
        let mut ang = Array2::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                ang[[y, x]] = self.v[[y, x]].atan2(self.u[[y, x]]);
            }
        }

        ang
    }
}

/// Farneback optical flow parameters
#[derive(Debug, Clone)]
pub struct FarnebackParams {
    /// Pyramid scale factor
    pub pyr_scale: f32,
    /// Number of pyramid levels
    pub levels: usize,
    /// Window size for polynomial expansion
    pub winsize: usize,
    /// Number of iterations at each pyramid level
    pub iterations: usize,
    /// Size of pixel neighborhood for polynomial expansion
    pub poly_n: usize,
    /// Standard deviation of Gaussian for polynomial expansion
    pub poly_sigma: f32,
}

impl Default for FarnebackParams {
    fn default() -> Self {
        Self {
            pyr_scale: 0.5,
            levels: 3,
            winsize: 15,
            iterations: 3,
            poly_n: 5,
            poly_sigma: 1.2,
        }
    }
}

/// TVL1 optical flow parameters
#[derive(Debug, Clone)]
pub struct TVL1Params {
    /// Regularization parameter
    pub tau: f32,
    /// Data term weight
    pub lambda: f32,
    /// Theta parameter
    pub theta: f32,
    /// Number of warp iterations
    pub warps: usize,
    /// Epsilon convergence threshold
    pub epsilon: f32,
    /// Number of inner iterations
    pub inner_iterations: usize,
    /// Number of outer iterations
    pub outer_iterations: usize,
    /// Pyramid scale factor
    pub scale_step: f32,
    /// Number of scales
    pub scales: usize,
}

impl Default for TVL1Params {
    fn default() -> Self {
        Self {
            tau: 0.25,
            lambda: 0.15,
            theta: 0.3,
            warps: 5,
            epsilon: 0.01,
            inner_iterations: 30,
            outer_iterations: 10,
            scale_step: 0.8,
            scales: 5,
        }
    }
}

/// Compute dense optical flow using Farneback algorithm
///
/// # Arguments
///
/// * `img1` - First frame
/// * `img2` - Second frame
/// * `params` - Algorithm parameters
///
/// # Returns
///
/// * Result containing dense flow field
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_vision::feature::optical_flow_advanced::{farneback_flow, FarnebackParams};
/// use image::open;
///
/// fn main() {
///     let frame1 = open("frame1.jpg").expect("frame1.jpg");
///     let frame2 = open("frame2.jpg").expect("frame2.jpg");
///     let flow = farneback_flow(&frame1, &frame2, &FarnebackParams::default()).expect("flow");
///     println!("Flow computed with magnitude max: {:?}", flow.magnitude().iter().fold(0.0f32, |a, &b| a.max(b)));
/// }
/// ```
pub fn farneback_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    params: &FarnebackParams,
) -> Result<DenseFlow> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (width, height) = gray1.dimensions();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::InvalidParameter(
            "Images must have same dimensions".to_string(),
        ));
    }

    // Build pyramids
    let pyr1 = build_pyramid(&gray1, params.levels, params.pyr_scale)?;
    let pyr2 = build_pyramid(&gray2, params.levels, params.pyr_scale)?;

    // Initialize flow at coarsest level
    let coarsest_level = params.levels - 1;
    let (h, w) = pyr1[coarsest_level].dim();
    let mut flow = DenseFlow::new(h, w);

    // Iterate through pyramid levels from coarse to fine
    for level in (0..params.levels).rev() {
        let img1_level = &pyr1[level];
        let img2_level = &pyr2[level];

        // Upsample flow from previous level
        if level < params.levels - 1 {
            flow = upsample_flow(&flow, img1_level.dim())?;
        }

        // Compute polynomial expansion
        for _iter in 0..params.iterations {
            update_flow_farneback(&mut flow, img1_level, img2_level, params)?;
        }
    }

    // Resize flow to original dimensions if needed
    if flow.u.dim() != (height as usize, width as usize) {
        flow = upsample_flow(&flow, (height as usize, width as usize))?;
    }

    Ok(flow)
}

/// Compute dense optical flow using TVL1 algorithm
///
/// # Arguments
///
/// * `img1` - First frame
/// * `img2` - Second frame
/// * `params` - Algorithm parameters
///
/// # Returns
///
/// * Result containing dense flow field
pub fn tvl1_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    params: &TVL1Params,
) -> Result<DenseFlow> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (width, height) = gray1.dimensions();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::InvalidParameter(
            "Images must have same dimensions".to_string(),
        ));
    }

    // Build pyramids
    let pyr1 = build_pyramid(&gray1, params.scales, params.scale_step)?;
    let pyr2 = build_pyramid(&gray2, params.scales, params.scale_step)?;

    // Initialize flow at coarsest level
    let coarsest_level = params.scales - 1;
    let (h, w) = pyr1[coarsest_level].dim();
    let mut flow = DenseFlow::new(h, w);

    // Iterate through pyramid levels
    for level in (0..params.scales).rev() {
        let i1 = &pyr1[level];
        let i2 = &pyr2[level];

        // Upsample flow from previous level
        if level < params.scales - 1 {
            flow = upsample_flow(&flow, i1.dim())?;
        }

        // TVL1 optimization
        for _warp in 0..params.warps {
            // Warp image
            let i2_warped = warp_image(i2, &flow)?;

            // Compute image gradients
            let (grad_x, grad_y) = compute_gradients(&i2_warped)?;

            // Outer iterations
            for _outer in 0..params.outer_iterations {
                // Compute data term
                let rho = compute_data_term(i1, &i2_warped, &flow, &grad_x, &grad_y)?;

                // Inner iterations - update flow
                for _inner in 0..params.inner_iterations {
                    update_flow_tvl1(&mut flow, &rho, params)?;
                }
            }
        }
    }

    // Resize to original dimensions
    if flow.u.dim() != (height as usize, width as usize) {
        flow = upsample_flow(&flow, (height as usize, width as usize))?;
    }

    Ok(flow)
}

/// Compute dense optical flow using Dual TVL1 algorithm
///
/// This is an improved version of TVL1 with better handling of large displacements
pub fn dual_tvl1_flow(
    img1: &DynamicImage,
    img2: &DynamicImage,
    params: &TVL1Params,
) -> Result<DenseFlow> {
    // Dual TVL1 uses primal-dual optimization
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    let (width, height) = gray1.dimensions();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::InvalidParameter(
            "Images must have same dimensions".to_string(),
        ));
    }

    // Build pyramids
    let pyr1 = build_pyramid(&gray1, params.scales, params.scale_step)?;
    let pyr2 = build_pyramid(&gray2, params.scales, params.scale_step)?;

    // Initialize flow and dual variables
    let coarsest_level = params.scales - 1;
    let (h, w) = pyr1[coarsest_level].dim();
    let mut flow = DenseFlow::new(h, w);
    let mut dual_p = DenseFlow::new(h, w);
    let mut dual_q = DenseFlow::new(h, w);

    // Iterate through pyramid levels
    for level in (0..params.scales).rev() {
        let i1 = &pyr1[level];
        let i2 = &pyr2[level];
        let (h, w) = i1.dim();

        // Upsample from previous level
        if level < params.scales - 1 {
            flow = upsample_flow(&flow, (h, w))?;
            dual_p = upsample_flow(&dual_p, (h, w))?;
            dual_q = upsample_flow(&dual_q, (h, w))?;
        }

        // Dual TVL1 optimization
        for _warp in 0..params.warps {
            let i2_warped = warp_image(i2, &flow)?;
            let (grad_x, grad_y) = compute_gradients(&i2_warped)?;

            for _iter in 0..params.outer_iterations {
                // Primal-dual updates
                update_dual_variables(&mut dual_p, &mut dual_q, &flow, params)?;
                update_primal_flow(
                    &mut flow, &dual_p, &dual_q, i1, &i2_warped, &grad_x, &grad_y, params,
                )?;
            }
        }
    }

    // Resize to original dimensions
    if flow.u.dim() != (height as usize, width as usize) {
        flow = upsample_flow(&flow, (height as usize, width as usize))?;
    }

    Ok(flow)
}

/// Build Gaussian pyramid
fn build_pyramid(img: &GrayImage, levels: usize, scale: f32) -> Result<Vec<Array2<f32>>> {
    let (width, height) = img.dimensions();
    let mut pyramid = Vec::with_capacity(levels);

    // Convert first level to float array
    let mut current = image_to_float_array(img);
    pyramid.push(current.clone());

    // Build remaining levels
    for _ in 1..levels {
        let (h, w) = current.dim();
        let new_h = ((h as f32 * scale) as usize).max(1);
        let new_w = ((w as f32 * scale) as usize).max(1);

        // Gaussian blur before downsampling
        current = gaussian_blur(&current, 1.0)?;

        // Downsample
        current = resize_array(&current, new_h, new_w)?;
        pyramid.push(current.clone());
    }

    Ok(pyramid)
}

/// Convert grayscale image to float array
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

/// Simple Gaussian blur
fn gaussian_blur(img: &Array2<f32>, sigma: f32) -> Result<Array2<f32>> {
    let (height, width) = img.dim();
    let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd
    let radius = kernel_size / 2;

    // Create Gaussian kernel
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f32 - radius as f32;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    for val in &mut kernel {
        *val /= sum;
    }

    let mut result = Array2::zeros((height, width));

    // Horizontal pass
    let mut temp = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k_val) in kernel.iter().enumerate().take(kernel_size) {
                let xi = (x as i32 + i as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum += img[[y, xi]] * k_val;
            }
            temp[[y, x]] = sum;
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k_val) in kernel.iter().enumerate().take(kernel_size) {
                let yi = (y as i32 + i as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum += temp[[yi, x]] * k_val;
            }
            result[[y, x]] = sum;
        }
    }

    Ok(result)
}

/// Resize array using bilinear interpolation
fn resize_array(src: &Array2<f32>, new_h: usize, new_w: usize) -> Result<Array2<f32>> {
    let (src_h, src_w) = src.dim();
    let mut dst = Array2::zeros((new_h, new_w));

    let scale_y = src_h as f32 / new_h as f32;
    let scale_x = src_w as f32 / new_w as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = y as f32 * scale_y;
            let src_x = x as f32 * scale_x;

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let x1 = (x0 + 1).min(src_w - 1);

            let fy = src_y - y0 as f32;
            let fx = src_x - x0 as f32;

            let val = (1.0 - fy) * (1.0 - fx) * src[[y0, x0]]
                + (1.0 - fy) * fx * src[[y0, x1]]
                + fy * (1.0 - fx) * src[[y1, x0]]
                + fy * fx * src[[y1, x1]];

            dst[[y, x]] = val;
        }
    }

    Ok(dst)
}

/// Upsample flow field
fn upsample_flow(flow: &DenseFlow, new_dim: (usize, usize)) -> Result<DenseFlow> {
    let (old_h, old_w) = flow.u.dim();
    let (new_h, new_w) = new_dim;

    let scale_y = new_h as f32 / old_h as f32;
    let scale_x = new_w as f32 / old_w as f32;

    let u = resize_array(&flow.u, new_h, new_w)?;
    let v = resize_array(&flow.v, new_h, new_w)?;

    // Scale flow values
    let u = u * scale_x;
    let v = v * scale_y;

    Ok(DenseFlow { u, v })
}

/// Update flow using Farneback polynomial expansion
fn update_flow_farneback(
    flow: &mut DenseFlow,
    img1: &Array2<f32>,
    img2: &Array2<f32>,
    params: &FarnebackParams,
) -> Result<()> {
    let (height, width) = img1.dim();
    let winsize = params.winsize;
    let radius = winsize / 2;

    // Compute image gradients
    let (gx1, gy1) = compute_gradients(img1)?;
    let (gx2, gy2) = compute_gradients(img2)?;

    // Update flow at each pixel
    for y in radius..height.saturating_sub(radius) {
        for x in radius..width.saturating_sub(radius) {
            // Compute local polynomial coefficients
            let mut a = [[0.0f32; 6]; 6];
            let mut b = [0.0f32; 6];

            for dy in 0..winsize {
                let ny = y + dy - radius;
                for dx in 0..winsize {
                    let nx = x + dx - radius;

                    let i1 = img1[[ny, nx]];
                    let i2 = img2[[ny, nx]];
                    let ix = (gx1[[ny, nx]] + gx2[[ny, nx]]) / 2.0;
                    let iy = (gy1[[ny, nx]] + gy2[[ny, nx]]) / 2.0;
                    let it = i2 - i1;

                    // Build system
                    let features = [ix, iy, ix * ix, iy * iy, ix * iy, 1.0];
                    for i in 0..6 {
                        for j in 0..6 {
                            a[i][j] += features[i] * features[j];
                        }
                        b[i] += -features[i] * it;
                    }
                }
            }

            // Solve 6x6 system (simplified - use only first 2 equations)
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            if det.abs() > 1e-6 {
                let du = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
                let dv = (a[0][0] * b[1] - a[1][0] * b[0]) / det;

                flow.u[[y, x]] += du * 0.1; // Damping factor
                flow.v[[y, x]] += dv * 0.1;
            }
        }
    }

    Ok(())
}

/// Compute image gradients
fn compute_gradients(img: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>)> {
    let (height, width) = img.dim();
    let mut gx = Array2::zeros((height, width));
    let mut gy = Array2::zeros((height, width));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            gx[[y, x]] = (img[[y, x + 1]] - img[[y, x - 1]]) / 2.0;
            gy[[y, x]] = (img[[y + 1, x]] - img[[y - 1, x]]) / 2.0;
        }
    }

    Ok((gx, gy))
}

/// Warp image using flow field
fn warp_image(img: &Array2<f32>, flow: &DenseFlow) -> Result<Array2<f32>> {
    let (height, width) = img.dim();
    let mut warped = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let nx = x as f32 + flow.u[[y, x]];
            let ny = y as f32 + flow.v[[y, x]];

            // Bilinear interpolation
            if nx >= 0.0 && nx < (width - 1) as f32 && ny >= 0.0 && ny < (height - 1) as f32 {
                let x0 = nx.floor() as usize;
                let y0 = ny.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = nx - x0 as f32;
                let fy = ny - y0 as f32;

                warped[[y, x]] = (1.0 - fy) * (1.0 - fx) * img[[y0, x0]]
                    + (1.0 - fy) * fx * img[[y0, x1]]
                    + fy * (1.0 - fx) * img[[y1, x0]]
                    + fy * fx * img[[y1, x1]];
            }
        }
    }

    Ok(warped)
}

/// Compute data term for TVL1
fn compute_data_term(
    i1: &Array2<f32>,
    i2_warped: &Array2<f32>,
    flow: &DenseFlow,
    grad_x: &Array2<f32>,
    grad_y: &Array2<f32>,
) -> Result<Array2<f32>> {
    let (height, width) = i1.dim();
    let mut rho = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let diff = i2_warped[[y, x]] - i1[[y, x]];
            let grad_flow = grad_x[[y, x]] * flow.u[[y, x]] + grad_y[[y, x]] * flow.v[[y, x]];
            rho[[y, x]] = diff + grad_flow;
        }
    }

    Ok(rho)
}

/// Update flow in TVL1 optimization
fn update_flow_tvl1(flow: &mut DenseFlow, rho: &Array2<f32>, params: &TVL1Params) -> Result<()> {
    let (height, width) = flow.u.dim();

    // Compute divergence
    let (div_p_u, div_p_v) = compute_divergence(&flow.u, &flow.v)?;

    // Update flow
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Update u
            let u_bar = flow.u[[y, x]] + params.tau * div_p_u[[y, x]];
            flow.u[[y, x]] = u_bar - params.tau * params.lambda * rho[[y, x]];

            // Update v
            let v_bar = flow.v[[y, x]] + params.tau * div_p_v[[y, x]];
            flow.v[[y, x]] = v_bar - params.tau * params.lambda * rho[[y, x]];
        }
    }

    Ok(())
}

/// Compute divergence
fn compute_divergence(u: &Array2<f32>, v: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>)> {
    let (height, width) = u.dim();
    let mut div_u = Array2::zeros((height, width));
    let mut div_v = Array2::zeros((height, width));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            div_u[[y, x]] =
                (u[[y, x + 1]] - u[[y, x - 1]]) / 2.0 + (u[[y + 1, x]] - u[[y - 1, x]]) / 2.0;
            div_v[[y, x]] =
                (v[[y, x + 1]] - v[[y, x - 1]]) / 2.0 + (v[[y + 1, x]] - v[[y - 1, x]]) / 2.0;
        }
    }

    Ok((div_u, div_v))
}

/// Update dual variables in Dual TVL1
fn update_dual_variables(
    dual_p: &mut DenseFlow,
    dual_q: &mut DenseFlow,
    flow: &DenseFlow,
    params: &TVL1Params,
) -> Result<()> {
    let (height, width) = flow.u.dim();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Compute gradient of flow
            let grad_u_x = flow.u[[y, x + 1]] - flow.u[[y, x]];
            let grad_u_y = flow.u[[y + 1, x]] - flow.u[[y, x]];
            let grad_v_x = flow.v[[y, x + 1]] - flow.v[[y, x]];
            let grad_v_y = flow.v[[y + 1, x]] - flow.v[[y, x]];

            // Update dual variables with projection
            dual_p.u[[y, x]] = (dual_p.u[[y, x]] + params.tau * grad_u_x).clamp(-1.0, 1.0);
            dual_p.v[[y, x]] = (dual_p.v[[y, x]] + params.tau * grad_u_y).clamp(-1.0, 1.0);
            dual_q.u[[y, x]] = (dual_q.u[[y, x]] + params.tau * grad_v_x).clamp(-1.0, 1.0);
            dual_q.v[[y, x]] = (dual_q.v[[y, x]] + params.tau * grad_v_y).clamp(-1.0, 1.0);
        }
    }

    Ok(())
}

/// Update primal flow in Dual TVL1
fn update_primal_flow(
    flow: &mut DenseFlow,
    dual_p: &DenseFlow,
    dual_q: &DenseFlow,
    i1: &Array2<f32>,
    i2_warped: &Array2<f32>,
    grad_x: &Array2<f32>,
    grad_y: &Array2<f32>,
    params: &TVL1Params,
) -> Result<()> {
    let (height, width) = flow.u.dim();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Divergence of dual variables
            let div_p_u = (dual_p.u[[y, x]] - dual_p.u[[y, x - 1]])
                + (dual_p.v[[y, x]] - dual_p.v[[y - 1, x]]);
            let div_q_v = (dual_q.u[[y, x]] - dual_q.u[[y, x - 1]])
                + (dual_q.v[[y, x]] - dual_q.v[[y - 1, x]]);

            // Data term
            let rho = i2_warped[[y, x]] - i1[[y, x]]
                + grad_x[[y, x]] * flow.u[[y, x]]
                + grad_y[[y, x]] * flow.v[[y, x]];

            // Update flow
            flow.u[[y, x]] += params.theta * (div_p_u - params.lambda * grad_x[[y, x]] * rho);
            flow.v[[y, x]] += params.theta * (div_q_v - params.lambda * grad_y[[y, x]] * rho);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_farneback_flow() {
        let img1 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let img2 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));

        let flow = farneback_flow(&img1, &img2, &FarnebackParams::default());
        assert!(flow.is_ok());
    }

    #[test]
    fn test_tvl1_flow() {
        let img1 = DynamicImage::ImageRgb8(RgbImage::new(16, 16));
        let img2 = DynamicImage::ImageRgb8(RgbImage::new(16, 16));

        // Use minimal params to avoid timeout in debug mode
        let params = TVL1Params {
            scales: 1,
            warps: 1,
            inner_iterations: 2,
            outer_iterations: 2,
            ..TVL1Params::default()
        };
        let flow = tvl1_flow(&img1, &img2, &params);
        assert!(flow.is_ok());
    }

    #[test]
    fn test_dual_tvl1_flow() {
        let img1 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let img2 = DynamicImage::ImageRgb8(RgbImage::new(64, 64));

        let flow = dual_tvl1_flow(&img1, &img2, &TVL1Params::default());
        assert!(flow.is_ok());
    }

    #[test]
    fn test_dense_flow_magnitude() {
        let flow = DenseFlow::new(10, 10);
        let mag = flow.magnitude();
        assert_eq!(mag.dim(), (10, 10));
    }

    #[test]
    fn test_dense_flow_angle() {
        let flow = DenseFlow::new(10, 10);
        let ang = flow.angle();
        assert_eq!(ang.dim(), (10, 10));
    }

    #[test]
    fn test_pyramid_building() {
        let img = GrayImage::new(64, 64);
        let pyramid = build_pyramid(&img, 3, 0.5);
        assert!(pyramid.is_ok());
        let pyr = pyramid.expect("pyramid building should succeed");
        assert_eq!(pyr.len(), 3);
    }
}
