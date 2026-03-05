//! Unified optical flow API
//!
//! Provides a single entry point for computing optical flow between two
//! frames using different methods:
//!
//! - **Lucas-Kanade**: Sparse or dense local flow using window-based optimization
//! - **Horn-Schunck**: Dense global flow using variational energy minimization
//! - **Farneback**: Dense polynomial expansion flow (simplified)

use crate::error::Result;
use crate::feature::optical_flow::{FlowVector, HornSchunckParams, LucasKanadeParams};
use image::DynamicImage;
use scirs2_core::ndarray::Array2;

/// Optical flow method selection
#[derive(Debug, Clone)]
pub enum FlowMethod {
    /// Lucas-Kanade sparse/dense optical flow
    LucasKanade {
        /// Window size for local computation
        window_size: usize,
        /// Maximum iterations
        max_iterations: usize,
        /// Number of pyramid levels (0 for no pyramid)
        pyramid_levels: usize,
    },
    /// Horn-Schunck dense variational optical flow
    HornSchunck {
        /// Smoothness weight (larger = smoother flow)
        alpha: f32,
        /// Maximum iterations
        max_iterations: usize,
    },
    /// Farneback dense optical flow (simplified)
    Farneback {
        /// Window size
        window_size: usize,
    },
}

impl Default for FlowMethod {
    fn default() -> Self {
        FlowMethod::LucasKanade {
            window_size: 15,
            max_iterations: 20,
            pyramid_levels: 3,
        }
    }
}

/// Result of optical flow computation
#[derive(Debug, Clone)]
pub struct FlowResult {
    /// Flow field (height x width) with (u, v) displacement per pixel
    pub flow: Array2<FlowVector>,
    /// Method used
    pub method_name: String,
}

impl FlowResult {
    /// Compute the magnitude of the flow at each pixel
    pub fn magnitude(&self) -> Array2<f32> {
        let (h, w) = self.flow.dim();
        let mut mag = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let fv = &self.flow[[y, x]];
                mag[[y, x]] = (fv.u * fv.u + fv.v * fv.v).sqrt();
            }
        }
        mag
    }

    /// Compute the angle of the flow at each pixel (in radians)
    pub fn angle(&self) -> Array2<f32> {
        let (h, w) = self.flow.dim();
        let mut ang = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let fv = &self.flow[[y, x]];
                ang[[y, x]] = fv.v.atan2(fv.u);
            }
        }
        ang
    }

    /// Get the mean flow magnitude
    pub fn mean_magnitude(&self) -> f32 {
        let mag = self.magnitude();
        let total: f32 = mag.iter().sum();
        let count = mag.len();
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }
}

/// Compute optical flow between two frames
///
/// This is the unified API for optical flow computation.
///
/// # Arguments
///
/// * `prev` - Previous (reference) frame
/// * `next` - Next (target) frame
/// * `method` - Optical flow method and parameters
///
/// # Returns
///
/// * Flow result containing the dense flow field
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::flow_unified::{calc_optical_flow, FlowMethod};
/// use image::{DynamicImage, RgbImage};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let frame1 = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
/// let frame2 = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
///
/// let result = calc_optical_flow(&frame1, &frame2, FlowMethod::default())?;
/// assert_eq!(result.flow.dim(), (32, 32));
/// # Ok(())
/// # }
/// ```
pub fn calc_optical_flow(
    prev: &DynamicImage,
    next: &DynamicImage,
    method: FlowMethod,
) -> Result<FlowResult> {
    match method {
        FlowMethod::LucasKanade {
            window_size,
            max_iterations,
            pyramid_levels,
        } => {
            let params = LucasKanadeParams {
                window_size,
                max_iterations,
                epsilon: 0.01,
                pyramid_levels,
            };

            let flow = crate::feature::optical_flow::lucas_kanade_flow(prev, next, None, &params)?;

            Ok(FlowResult {
                flow,
                method_name: "LucasKanade".to_string(),
            })
        }
        FlowMethod::HornSchunck {
            alpha,
            max_iterations,
        } => {
            let params = HornSchunckParams {
                alpha,
                max_iterations,
                epsilon: 1e-4,
            };

            let flow = crate::feature::optical_flow::horn_schunck_flow(prev, next, &params)?;

            Ok(FlowResult {
                flow,
                method_name: "HornSchunck".to_string(),
            })
        }
        FlowMethod::Farneback { window_size } => {
            let flow =
                crate::feature::optical_flow::farneback_flow(prev, next, 0.5, 3, window_size, 3)?;

            Ok(FlowResult {
                flow,
                method_name: "Farneback".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, RgbImage};

    #[test]
    fn test_calc_optical_flow_lucas_kanade() {
        let frame1 = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
        let frame2 = DynamicImage::ImageRgb8(RgbImage::new(32, 32));

        let result = calc_optical_flow(
            &frame1,
            &frame2,
            FlowMethod::LucasKanade {
                window_size: 15,
                max_iterations: 10,
                pyramid_levels: 2,
            },
        )
        .expect("LK flow failed");

        assert_eq!(result.flow.dim(), (32, 32));
        assert_eq!(result.method_name, "LucasKanade");
    }

    #[test]
    fn test_calc_optical_flow_horn_schunck() {
        let frame1 = DynamicImage::ImageRgb8(RgbImage::new(16, 16));
        let frame2 = DynamicImage::ImageRgb8(RgbImage::new(16, 16));

        let result = calc_optical_flow(
            &frame1,
            &frame2,
            FlowMethod::HornSchunck {
                alpha: 15.0,
                max_iterations: 50,
            },
        )
        .expect("HS flow failed");

        assert_eq!(result.flow.dim(), (16, 16));
        assert_eq!(result.method_name, "HornSchunck");
    }

    #[test]
    fn test_calc_optical_flow_farneback() {
        let frame1 = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
        let frame2 = DynamicImage::ImageRgb8(RgbImage::new(32, 32));

        let result = calc_optical_flow(&frame1, &frame2, FlowMethod::Farneback { window_size: 5 })
            .expect("Farneback flow failed");

        assert_eq!(result.flow.dim(), (32, 32));
        assert_eq!(result.method_name, "Farneback");
    }

    #[test]
    fn test_flow_result_magnitude() {
        let mut flow = Array2::from_elem((5, 5), FlowVector { u: 0.0, v: 0.0 });
        flow[[2, 2]] = FlowVector { u: 3.0, v: 4.0 };

        let result = FlowResult {
            flow,
            method_name: "test".to_string(),
        };

        let mag = result.magnitude();
        assert!((mag[[2, 2]] - 5.0).abs() < 1e-5);
        assert!(mag[[0, 0]].abs() < 1e-5);
    }

    #[test]
    fn test_flow_result_angle() {
        let mut flow = Array2::from_elem((5, 5), FlowVector { u: 0.0, v: 0.0 });
        flow[[2, 2]] = FlowVector { u: 1.0, v: 0.0 };

        let result = FlowResult {
            flow,
            method_name: "test".to_string(),
        };

        let ang = result.angle();
        assert!(ang[[2, 2]].abs() < 1e-5); // atan2(0, 1) = 0
    }

    #[test]
    fn test_flow_result_mean_magnitude() {
        let flow = Array2::from_elem((4, 4), FlowVector { u: 3.0, v: 4.0 });

        let result = FlowResult {
            flow,
            method_name: "test".to_string(),
        };

        let mean = result.mean_magnitude();
        assert!((mean - 5.0).abs() < 1e-5); // All vectors have magnitude 5
    }

    #[test]
    fn test_identical_frames_zero_flow() {
        let frame = DynamicImage::ImageRgb8(RgbImage::new(16, 16));

        let result = calc_optical_flow(
            &frame,
            &frame,
            FlowMethod::HornSchunck {
                alpha: 15.0,
                max_iterations: 100,
            },
        )
        .expect("HS flow failed");

        let mean_mag = result.mean_magnitude();
        assert!(
            mean_mag < 0.01,
            "Identical frames should have near-zero flow, got {}",
            mean_mag
        );
    }

    #[test]
    fn test_flow_method_default() {
        let method = FlowMethod::default();
        match method {
            FlowMethod::LucasKanade {
                window_size,
                pyramid_levels,
                ..
            } => {
                assert!(window_size > 0);
                assert!(pyramid_levels > 0);
            }
            _ => panic!("Default should be LucasKanade"),
        }
    }
}
