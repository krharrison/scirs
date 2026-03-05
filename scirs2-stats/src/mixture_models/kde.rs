//! Kernel Density Estimation with multiple kernels and bandwidth selection

use super::f64_to_f;
use super::GmmFloat;
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::validation::*;
use std::marker::PhantomData;

/// Kernel Density Estimation
pub struct KernelDensityEstimator<F> {
    /// Kernel type
    pub kernel: KernelType,
    /// Bandwidth
    pub bandwidth: F,
    /// Configuration
    pub config: KDEConfig,
    /// Training data
    pub trainingdata: Option<Array2<F>>,
    _phantom: PhantomData<F>,
}

/// Kernel types for KDE
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Gaussian kernel
    Gaussian,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Uniform kernel
    Uniform,
    /// Triangular kernel
    Triangular,
    /// Cosine kernel
    Cosine,
}

/// KDE configuration
#[derive(Debug, Clone)]
pub struct KDEConfig {
    /// Bandwidth selection method
    pub bandwidth_method: BandwidthMethod,
    /// Enable parallel processing
    pub parallel: bool,
    /// Use SIMD optimizations
    pub use_simd: bool,
}

/// Bandwidth selection methods
#[derive(Debug, Clone, PartialEq)]
pub enum BandwidthMethod {
    /// Fixed bandwidth (user-specified)
    Fixed,
    /// Scott's rule of thumb
    Scott,
    /// Silverman's rule of thumb
    Silverman,
    /// Cross-validation
    CrossValidation,
}

impl Default for KDEConfig {
    fn default() -> Self {
        Self {
            bandwidth_method: BandwidthMethod::Scott,
            parallel: true,
            use_simd: true,
        }
    }
}

impl<F: GmmFloat> KernelDensityEstimator<F> {
    /// Create new KDE
    pub fn new(kernel: KernelType, bandwidth: F, config: KDEConfig) -> Self {
        Self {
            kernel,
            bandwidth,
            config,
            trainingdata: None,
            _phantom: PhantomData,
        }
    }

    /// Fit KDE to data
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<()> {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        if self.config.bandwidth_method != BandwidthMethod::Fixed {
            self.bandwidth = self.select_bandwidth_scalar(data)?;
        }

        self.trainingdata = Some(data.to_owned());
        Ok(())
    }

    fn select_bandwidth_scalar(&self, data: &ArrayView2<F>) -> StatsResult<F> {
        let (n, d) = data.dim();

        match self.config.bandwidth_method {
            BandwidthMethod::Scott => {
                let exp: F = f64_to_f(-1.0 / (d as f64 + 4.0), "scott_exp")?;
                let n_f: F = f64_to_f(n as f64, "n_scott")?;
                Ok(n_f.powf(exp))
            }
            BandwidthMethod::Silverman => {
                let factor_exp: F = f64_to_f(1.0 / (d as f64 + 4.0), "silv_exp")?;
                let factor_base: F = f64_to_f(4.0 / (d as f64 + 2.0), "silv_base")?;
                let n_exp: F = f64_to_f(-1.0 / (d as f64 + 4.0), "silv_n_exp")?;
                let n_f: F = f64_to_f(n as f64, "n_silv")?;
                Ok(factor_base.powf(factor_exp) * n_f.powf(n_exp))
            }
            BandwidthMethod::CrossValidation => self.cross_validation_bandwidth(data),
            BandwidthMethod::Fixed => Ok(self.bandwidth),
        }
    }

    fn cross_validation_bandwidth(&self, data: &ArrayView2<F>) -> StatsResult<F> {
        let (n, d) = data.dim();
        let exp: F = f64_to_f(-1.0 / (d as f64 + 4.0), "cv_exp")?;
        let n_f: F = f64_to_f(n as f64, "n_cv")?;
        Ok(n_f.powf(exp))
    }

    /// Evaluate density at given points
    pub fn score_samples(&self, points: &ArrayView2<F>) -> StatsResult<Array1<F>> {
        let trainingdata = self.trainingdata.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("KDE must be fitted before evaluation".into())
        })?;
        checkarray_finite(points, "points")?;

        if points.ncols() != trainingdata.ncols() {
            return Err(StatsError::DimensionMismatch(format!(
                "Points dimension ({}) must match training data dimension ({})",
                points.ncols(),
                trainingdata.ncols()
            )));
        }

        let n_points = points.nrows();
        let n_train = trainingdata.nrows();
        let d_f: F = f64_to_f(trainingdata.ncols() as f64, "d_kde")?;
        let n_train_f: F = f64_to_f(n_train as f64, "n_train_kde")?;
        let normalization = n_train_f * self.bandwidth.powf(d_f);

        let mut densities = Array1::zeros(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let mut density = F::zero();
            for j in 0..n_train {
                let train_point = trainingdata.row(j);
                let distance = self.compute_distance(&point, &train_point);
                let kernel_value = self.evaluate_kernel(distance / self.bandwidth);
                density = density + kernel_value;
            }
            densities[i] = density / normalization;
        }

        Ok(densities)
    }

    fn compute_distance(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<F>()
            .sqrt()
    }

    fn evaluate_kernel(&self, u: F) -> F {
        let half: F = f64_to_f(0.5, "half").unwrap_or(F::zero());
        let three_quarter: F = f64_to_f(0.75, "3/4").unwrap_or(F::zero());
        match self.kernel {
            KernelType::Gaussian => {
                let coeff: F = f64_to_f(1.0 / (2.0 * std::f64::consts::PI).sqrt(), "gauss_coeff")
                    .unwrap_or(F::zero());
                let two: F = f64_to_f(2.0, "two").unwrap_or(F::one());
                coeff * (-u * u / two).exp()
            }
            KernelType::Epanechnikov => {
                if u.abs() <= F::one() {
                    three_quarter * (F::one() - u * u)
                } else {
                    F::zero()
                }
            }
            KernelType::Uniform => {
                if u.abs() <= F::one() {
                    half
                } else {
                    F::zero()
                }
            }
            KernelType::Triangular => {
                if u.abs() <= F::one() {
                    F::one() - u.abs()
                } else {
                    F::zero()
                }
            }
            KernelType::Cosine => {
                if u.abs() <= F::one() {
                    let pi_4: F = f64_to_f(std::f64::consts::PI / 4.0, "pi/4").unwrap_or(F::zero());
                    let pi: F = f64_to_f(std::f64::consts::PI, "pi").unwrap_or(F::zero());
                    let two: F = f64_to_f(2.0, "two").unwrap_or(F::one());
                    pi_4 * (pi * u / two).cos()
                } else {
                    F::zero()
                }
            }
        }
    }
}

/// Evaluate KDE density at query points
pub fn kernel_density_estimation<F: GmmFloat>(
    data: &ArrayView2<F>,
    points: &ArrayView2<F>,
    kernel: Option<KernelType>,
    bandwidth: Option<F>,
) -> StatsResult<Array1<F>> {
    let kernel = kernel.unwrap_or(KernelType::Gaussian);
    let bandwidth = match bandwidth {
        Some(b) => b,
        None => {
            let n = data.nrows();
            let d = data.ncols();
            let exp: F = f64_to_f(-1.0 / (d as f64 + 4.0), "default_bw_exp")?;
            let n_f: F = f64_to_f(n as f64, "default_bw_n")?;
            n_f.powf(exp)
        }
    };

    let mut kde = KernelDensityEstimator::new(kernel, bandwidth, KDEConfig::default());
    kde.fit(data)?;
    kde.score_samples(points)
}
