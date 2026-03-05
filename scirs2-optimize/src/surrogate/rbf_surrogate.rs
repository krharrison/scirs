//! Radial Basis Function (RBF) Surrogate Model
//!
//! RBF surrogates interpolate a set of data points using radial basis functions.
//! They are well-suited for expensive black-box optimization because they provide
//! smooth interpolation and can handle high-dimensional problems.
//!
//! ## Kernel Types
//!
//! - **Polyharmonic**: r^k (k=1: linear, k=3: cubic, k=5: quintic)
//! - **Multiquadric**: sqrt(r^2 + c^2)
//! - **InverseMultiquadric**: 1 / sqrt(r^2 + c^2)
//! - **ThinPlateSpline**: r^2 * ln(r)
//! - **Gaussian**: exp(-r^2 / (2 * sigma^2))
//!
//! ## References
//!
//! - Buhmann, M.D. (2003). Radial Basis Functions: Theory and Implementations.
//! - Gutmann, H.-M. (2001). A Radial Basis Function Method for Global Optimization.

use super::{pairwise_sq_distances, solve_general, SurrogateModel};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

/// RBF kernel type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    /// Polyharmonic spline: r^k
    /// k=1: linear, k=3: cubic, k=5: quintic
    Polyharmonic(u32),
    /// Multiquadric: sqrt(r^2 + c^2)
    Multiquadric {
        /// Shape parameter c
        shape_param: f64,
    },
    /// Inverse multiquadric: 1 / sqrt(r^2 + c^2)
    InverseMultiquadric {
        /// Shape parameter c
        shape_param: f64,
    },
    /// Thin-plate spline: r^2 * ln(r)
    ThinPlateSpline,
    /// Gaussian: exp(-r^2 / (2 * sigma^2))
    Gaussian {
        /// Bandwidth parameter sigma
        sigma: f64,
    },
}

impl Default for RbfKernel {
    fn default() -> Self {
        RbfKernel::Polyharmonic(3) // cubic
    }
}

impl RbfKernel {
    /// Evaluate the kernel function for a given squared distance
    fn evaluate(&self, sq_dist: f64) -> f64 {
        let r = sq_dist.sqrt();
        match *self {
            RbfKernel::Polyharmonic(k) => {
                if r < 1e-30 {
                    0.0
                } else {
                    r.powi(k as i32)
                }
            }
            RbfKernel::Multiquadric { shape_param } => (sq_dist + shape_param * shape_param).sqrt(),
            RbfKernel::InverseMultiquadric { shape_param } => {
                1.0 / (sq_dist + shape_param * shape_param).sqrt()
            }
            RbfKernel::ThinPlateSpline => {
                if r < 1e-30 {
                    0.0
                } else {
                    sq_dist * r.ln()
                }
            }
            RbfKernel::Gaussian { sigma } => (-sq_dist / (2.0 * sigma * sigma)).exp(),
        }
    }

    /// Whether this kernel requires a polynomial tail for well-posedness
    fn needs_polynomial_tail(&self) -> bool {
        matches!(
            self,
            RbfKernel::Polyharmonic(_) | RbfKernel::ThinPlateSpline
        )
    }

    /// Degree of polynomial tail needed
    fn polynomial_degree(&self) -> usize {
        match *self {
            RbfKernel::Polyharmonic(k) => {
                // For polyharmonic splines of order k, need polynomial of degree >= floor(k/2)
                (k as usize) / 2
            }
            RbfKernel::ThinPlateSpline => 1,
            _ => 0,
        }
    }
}

/// Options for RBF surrogate
#[derive(Debug, Clone)]
pub struct RbfOptions {
    /// RBF kernel to use
    pub kernel: RbfKernel,
    /// Regularization parameter (nugget) for numerical stability
    pub regularization: f64,
    /// Whether to normalize the training data
    pub normalize: bool,
}

impl Default for RbfOptions {
    fn default() -> Self {
        Self {
            kernel: RbfKernel::default(),
            regularization: 1e-10,
            normalize: true,
        }
    }
}

/// RBF Surrogate Model
pub struct RbfSurrogate {
    options: RbfOptions,
    /// Training points, shape (n_samples, n_features)
    x_train: Option<Array2<f64>>,
    /// Training values, shape (n_samples,)
    y_train: Option<Array1<f64>>,
    /// RBF weights (alpha)
    weights: Option<Array1<f64>>,
    /// Polynomial coefficients (if polynomial tail is used)
    poly_coeffs: Option<Array1<f64>>,
    /// Normalization parameters
    x_mean: Option<Array1<f64>>,
    x_std: Option<Array1<f64>>,
    y_mean: f64,
    y_std: f64,
    /// Cached kernel matrix for uncertainty estimation
    kernel_matrix: Option<Array2<f64>>,
}

impl RbfSurrogate {
    /// Create a new RBF surrogate
    pub fn new(options: RbfOptions) -> Self {
        Self {
            options,
            x_train: None,
            y_train: None,
            weights: None,
            poly_coeffs: None,
            x_mean: None,
            x_std: None,
            y_mean: 0.0,
            y_std: 1.0,
            kernel_matrix: None,
        }
    }

    /// Compute the kernel matrix for given points
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let sq_dists = pairwise_sq_distances(x, x);
        let mut kernel = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                kernel[[i, j]] = self.options.kernel.evaluate(sq_dists[[i, j]]);
            }
        }
        kernel
    }

    /// Compute kernel vector between a point and training points
    fn compute_kernel_vector(&self, x: &Array1<f64>, x_train: &Array2<f64>) -> Array1<f64> {
        let n = x_train.nrows();
        let mut k_vec = Array1::zeros(n);
        for i in 0..n {
            let mut sq_dist = 0.0;
            for j in 0..x.len() {
                let diff = x[j] - x_train[[i, j]];
                sq_dist += diff * diff;
            }
            k_vec[i] = self.options.kernel.evaluate(sq_dist);
        }
        k_vec
    }

    /// Build the polynomial matrix for the polynomial tail
    fn build_polynomial_matrix(&self, x: &Array2<f64>, degree: usize) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();

        if degree == 0 {
            // Just a constant term
            Array2::ones((n, 1))
        } else if degree == 1 {
            // Constant + linear terms
            let ncols = 1 + d;
            let mut p = Array2::zeros((n, ncols));
            for i in 0..n {
                p[[i, 0]] = 1.0;
                for j in 0..d {
                    p[[i, j + 1]] = x[[i, j]];
                }
            }
            p
        } else {
            // For higher degrees, just use up to linear (simplification)
            let ncols = 1 + d;
            let mut p = Array2::zeros((n, ncols));
            for i in 0..n {
                p[[i, 0]] = 1.0;
                for j in 0..d {
                    p[[i, j + 1]] = x[[i, j]];
                }
            }
            p
        }
    }

    /// Normalize x data
    fn normalize_x(&self, x: &Array2<f64>) -> Array2<f64> {
        if let (Some(ref mean), Some(ref std)) = (&self.x_mean, &self.x_std) {
            let mut normalized = x.clone();
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    let s = if std[j] > 1e-30 { std[j] } else { 1.0 };
                    normalized[[i, j]] = (x[[i, j]] - mean[j]) / s;
                }
            }
            normalized
        } else {
            x.clone()
        }
    }

    /// Normalize a single x point
    fn normalize_x_point(&self, x: &Array1<f64>) -> Array1<f64> {
        if let (Some(ref mean), Some(ref std)) = (&self.x_mean, &self.x_std) {
            let mut normalized = x.clone();
            for j in 0..x.len() {
                let s = if std[j] > 1e-30 { std[j] } else { 1.0 };
                normalized[j] = (x[j] - mean[j]) / s;
            }
            normalized
        } else {
            x.clone()
        }
    }
}

impl SurrogateModel for RbfSurrogate {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> OptimizeResult<()> {
        let n = x.nrows();
        let d = x.ncols();

        if n < d + 1 {
            return Err(OptimizeError::InvalidInput(format!(
                "Need at least {} data points for {} dimensions, got {}",
                d + 1,
                d,
                n
            )));
        }

        // Compute normalization parameters
        if self.options.normalize {
            let mut x_mean = Array1::zeros(d);
            let mut x_std = Array1::zeros(d);
            for j in 0..d {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x[[i, j]];
                }
                x_mean[j] = sum / n as f64;

                let mut sq_sum = 0.0;
                for i in 0..n {
                    let diff = x[[i, j]] - x_mean[j];
                    sq_sum += diff * diff;
                }
                x_std[j] = (sq_sum / n as f64).sqrt();
                if x_std[j] < 1e-30 {
                    x_std[j] = 1.0;
                }
            }

            self.x_mean = Some(x_mean);
            self.x_std = Some(x_std);

            let y_sum: f64 = y.iter().sum();
            self.y_mean = y_sum / n as f64;
            let y_var: f64 = y.iter().map(|yi| (yi - self.y_mean).powi(2)).sum::<f64>() / n as f64;
            self.y_std = y_var.sqrt();
            if self.y_std < 1e-30 {
                self.y_std = 1.0;
            }
        }

        // Normalize training data
        let x_norm = self.normalize_x(x);
        let y_norm: Array1<f64> = if self.options.normalize {
            y.mapv(|yi| (yi - self.y_mean) / self.y_std)
        } else {
            y.clone()
        };

        // Compute kernel matrix
        let mut kernel = self.compute_kernel_matrix(&x_norm);

        // Add regularization
        for i in 0..n {
            kernel[[i, i]] += self.options.regularization;
        }

        self.kernel_matrix = Some(kernel.clone());

        if self.options.kernel.needs_polynomial_tail() {
            let degree = self.options.kernel.polynomial_degree();
            let p = self.build_polynomial_matrix(&x_norm, degree);
            let m = p.ncols();

            // Solve the augmented system:
            // [K  P] [alpha]   [y]
            // [P' 0] [beta ] = [0]
            let total = n + m;
            let mut aug = Array2::zeros((total, total));
            for i in 0..n {
                for j in 0..n {
                    aug[[i, j]] = kernel[[i, j]];
                }
                for j in 0..m {
                    aug[[i, n + j]] = p[[i, j]];
                    aug[[n + j, i]] = p[[i, j]];
                }
            }

            let mut rhs = Array1::zeros(total);
            for i in 0..n {
                rhs[i] = y_norm[i];
            }

            let solution = solve_general(&aug, &rhs)?;
            self.weights = Some(solution.slice(scirs2_core::ndarray::s![..n]).to_owned());
            self.poly_coeffs = Some(solution.slice(scirs2_core::ndarray::s![n..]).to_owned());
        } else {
            // Solve K * alpha = y
            let weights = solve_general(&kernel, &y_norm)?;
            self.weights = Some(weights);
            self.poly_coeffs = None;
        }

        self.x_train = Some(x_norm);
        self.y_train = Some(y_norm);

        Ok(())
    }

    fn predict(&self, x: &Array1<f64>) -> OptimizeResult<f64> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;

        let x_norm = self.normalize_x_point(x);
        let k_vec = self.compute_kernel_vector(&x_norm, x_train);

        let mut prediction = k_vec.dot(weights);

        // Add polynomial tail contribution
        if let Some(ref poly_coeffs) = self.poly_coeffs {
            let d = x_norm.len();
            // Constant term
            prediction += poly_coeffs[0];
            // Linear terms
            for j in 0..d.min(poly_coeffs.len() - 1) {
                prediction += poly_coeffs[j + 1] * x_norm[j];
            }
        }

        // Denormalize
        if self.options.normalize {
            prediction = prediction * self.y_std + self.y_mean;
        }

        Ok(prediction)
    }

    fn predict_with_uncertainty(&self, x: &Array1<f64>) -> OptimizeResult<(f64, f64)> {
        let mean = self.predict(x)?;

        // Estimate uncertainty using leave-one-out cross-validation approximation
        // Simple heuristic: distance-based uncertainty
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;

        let x_norm = self.normalize_x_point(x);
        let n = x_train.nrows();

        // Compute minimum distance to training points
        let mut min_dist = f64::INFINITY;
        let mut sum_inv_dist = 0.0;
        for i in 0..n {
            let mut sq_dist = 0.0;
            for j in 0..x_norm.len() {
                let diff = x_norm[j] - x_train[[i, j]];
                sq_dist += diff * diff;
            }
            let dist = sq_dist.sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
            if dist > 1e-30 {
                sum_inv_dist += 1.0 / dist;
            }
        }

        // Uncertainty is proportional to distance from training data
        // Normalize by the average distance scale
        let avg_inv_dist = if n > 0 { sum_inv_dist / n as f64 } else { 1.0 };
        let uncertainty = if avg_inv_dist > 1e-30 {
            min_dist * avg_inv_dist
        } else {
            min_dist
        };

        // Scale by y_std
        let scaled_uncertainty = uncertainty * self.y_std;

        Ok((mean, scaled_uncertainty.max(1e-10)))
    }

    fn n_samples(&self) -> usize {
        self.x_train.as_ref().map_or(0, |x| x.nrows())
    }

    fn n_features(&self) -> usize {
        self.x_train.as_ref().map_or(0, |x| x.ncols())
    }

    fn update(&mut self, x: &Array1<f64>, y: f64) -> OptimizeResult<()> {
        // Refit with the new point added
        let (new_x, new_y) = if let (Some(ref x_train), Some(ref y_train)) =
            (&self.x_train, &self.y_train)
        {
            // Denormalize existing data
            let d = x_train.ncols();
            let n = x_train.nrows();
            let mut x_denorm = Array2::zeros((n, d));
            for i in 0..n {
                for j in 0..d {
                    if self.options.normalize {
                        let s =
                            self.x_std
                                .as_ref()
                                .map_or(1.0, |s| if s[j] > 1e-30 { s[j] } else { 1.0 });
                        let m = self.x_mean.as_ref().map_or(0.0, |m| m[j]);
                        x_denorm[[i, j]] = x_train[[i, j]] * s + m;
                    } else {
                        x_denorm[[i, j]] = x_train[[i, j]];
                    }
                }
            }
            let y_denorm: Array1<f64> = if self.options.normalize {
                y_train.mapv(|yi| yi * self.y_std + self.y_mean)
            } else {
                y_train.clone()
            };

            // Append new point
            let mut new_x = Array2::zeros((n + 1, d));
            for i in 0..n {
                for j in 0..d {
                    new_x[[i, j]] = x_denorm[[i, j]];
                }
            }
            for j in 0..d {
                new_x[[n, j]] = x[j];
            }

            let mut new_y = Array1::zeros(n + 1);
            for i in 0..n {
                new_y[i] = y_denorm[i];
            }
            new_y[n] = y;

            (new_x, new_y)
        } else {
            let d = x.len();
            let mut new_x = Array2::zeros((1, d));
            for j in 0..d {
                new_x[[0, j]] = x[j];
            }
            let new_y = Array1::from_vec(vec![y]);
            (new_x, new_y)
        };

        self.fit(&new_x, &new_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_cubic_interpolation() {
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions {
            kernel: RbfKernel::Polyharmonic(3),
            regularization: 1e-8,
            normalize: false,
        });

        let result = rbf.fit(&x_train, &y_train);
        assert!(result.is_ok(), "RBF fit failed: {:?}", result.err());

        // Predict at training points (should interpolate)
        for i in 0..5 {
            let x = Array1::from_vec(vec![i as f64]);
            let pred = rbf.predict(&x).expect("Prediction failed");
            assert!(
                (pred - y_train[i]).abs() < 0.5,
                "Interpolation error at {}: pred={}, actual={}",
                i,
                pred,
                y_train[i]
            );
        }
    }

    #[test]
    fn test_rbf_gaussian() {
        let x_train = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions {
            kernel: RbfKernel::Gaussian { sigma: 1.0 },
            regularization: 1e-6,
            normalize: true,
        });

        let result = rbf.fit(&x_train, &y_train);
        assert!(result.is_ok());

        // Predict at a middle point
        let x = Array1::from_vec(vec![0.5, 0.5]);
        let pred = rbf.predict(&x);
        assert!(pred.is_ok());
        let val = pred.expect("Gaussian RBF prediction failed");
        // Should be roughly 1.0 (average of corner values)
        assert!(val > -1.0 && val < 3.0, "Gaussian RBF prediction: {}", val);
    }

    #[test]
    fn test_rbf_multiquadric() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![1.0, 2.0, 5.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions {
            kernel: RbfKernel::Multiquadric { shape_param: 1.0 },
            regularization: 1e-8,
            normalize: false,
        });

        assert!(rbf.fit(&x_train, &y_train).is_ok());

        let pred = rbf.predict(&Array1::from_vec(vec![1.0]));
        assert!(pred.is_ok());
    }

    #[test]
    fn test_rbf_thin_plate_spline() {
        let x_train = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions {
            kernel: RbfKernel::ThinPlateSpline,
            regularization: 1e-6,
            normalize: false,
        });

        assert!(rbf.fit(&x_train, &y_train).is_ok());

        let pred = rbf.predict(&Array1::from_vec(vec![0.5, 0.5]));
        assert!(pred.is_ok());
    }

    #[test]
    fn test_rbf_uncertainty() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 4.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions {
            kernel: RbfKernel::Gaussian { sigma: 1.0 },
            regularization: 1e-6,
            normalize: true,
        });
        rbf.fit(&x_train, &y_train).expect("Fit failed");

        // Uncertainty at a training point should be lower than far away
        let (_, unc_near) = rbf
            .predict_with_uncertainty(&Array1::from_vec(vec![1.0]))
            .expect("Prediction failed");
        let (_, unc_far) = rbf
            .predict_with_uncertainty(&Array1::from_vec(vec![5.0]))
            .expect("Prediction failed");
        assert!(
            unc_far > unc_near,
            "Far point uncertainty ({}) should be greater than near point ({})",
            unc_far,
            unc_near
        );
    }

    #[test]
    fn test_rbf_update() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 4.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions::default());
        rbf.fit(&x_train, &y_train).expect("Fit failed");
        assert_eq!(rbf.n_samples(), 3);

        // Add a new point
        rbf.update(&Array1::from_vec(vec![3.0]), 9.0)
            .expect("Update failed");
        assert_eq!(rbf.n_samples(), 4);
    }

    #[test]
    fn test_rbf_inverse_multiquadric() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![1.0, 2.0, 5.0]);

        let mut rbf = RbfSurrogate::new(RbfOptions {
            kernel: RbfKernel::InverseMultiquadric { shape_param: 1.0 },
            regularization: 1e-6,
            normalize: false,
        });

        assert!(rbf.fit(&x_train, &y_train).is_ok());
        let pred = rbf.predict(&Array1::from_vec(vec![1.0]));
        assert!(pred.is_ok());
    }
}
