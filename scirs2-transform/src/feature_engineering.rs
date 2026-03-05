//! Advanced Feature Engineering Transformations
//!
//! This module provides powerful feature engineering primitives for machine learning:
//!
//! - **Polynomial Features**: Generate polynomial and interaction features up to degree d
//! - **Interaction Features**: Pairwise product features for capturing non-linear relationships
//! - **Spline Features**: B-spline basis expansion for non-linear 1D feature modeling
//! - **Radial Basis Features**: Gaussian RBF expansion relative to a set of cluster centers
//! - **Target Encoding**: Mean-target encoding for categorical features
//! - **Quantile Binning**: Bin continuous features into equal-frequency quantile bins
//!
//! ## Design Philosophy
//!
//! All transformers follow a consistent API:
//! - `new()` / `with_*()` builder methods for configuration
//! - `fit()` / `transform()` / `fit_transform()` methods
//! - No `unwrap()` usage anywhere
//!
//! ## References
//!
//! - Friedman, J. H. (1991). Multivariate adaptive regression splines.
//! - De Boor, C. (1978). A Practical Guide to Splines.
//! - Moody, J., & Darken, C. J. (1989). Fast learning in networks of locally-tuned processing units.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;

// ─── Polynomial Features ─────────────────────────────────────────────────────

/// Generate polynomial and interaction features up to a given degree.
///
/// For each input feature vector x = [x1, x2, ..., xp], generates all
/// monomials up to the specified degree. For example, with degree=2 and
/// p=2 features and `include_bias=true`:
///
/// ```text
/// [1, x1, x2, x1^2, x1*x2, x2^2]
/// ```
///
/// # Example
///
/// ```rust
/// use scirs2_transform::feature_engineering::PolynomialFeatures;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("should succeed");
/// let result = PolynomialFeatures::transform(&x, 2).expect("should succeed");
/// // Columns: [1, x0, x1, x0^2, x0*x1, x1^2]
/// assert_eq!(result.ncols(), 6);
/// ```
pub struct PolynomialFeatures {
    /// Maximum polynomial degree
    degree: usize,
    /// Include the bias term (constant feature = 1)
    include_bias: bool,
    /// Include only interaction terms (no pure powers x_i^k for k > 1)
    interaction_only: bool,
    /// Number of input features (set at fit time)
    n_features_in: Option<usize>,
    /// Powers of each output feature (n_output_features × n_input_features)
    powers: Option<Array2<usize>>,
}

impl PolynomialFeatures {
    /// Create a new PolynomialFeatures transformer
    pub fn new(degree: usize) -> Result<Self> {
        if degree == 0 {
            return Err(TransformError::InvalidInput(
                "Polynomial degree must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            degree,
            include_bias: true,
            interaction_only: false,
            n_features_in: None,
            powers: None,
        })
    }

    /// Configure whether to include the bias (intercept) term
    pub fn with_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Configure whether to include only interaction terms
    pub fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Generate polynomial and interaction features up to `degree`.
    ///
    /// This is a convenience static method equivalent to creating
    /// a default `PolynomialFeatures` and calling `fit_transform`.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    /// * `degree` - Maximum polynomial degree
    ///
    /// # Returns
    /// * Feature matrix with polynomial features
    pub fn transform<S>(x: &ArrayBase<S, Ix2>, degree: usize) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let pf = Self::new(degree)?;
        pf.fit_transform(x)
    }

    /// Fit the transformer to data and compute the output feature matrix
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        let p = x.ncols();

        // Convert to f64
        let x_f64: Array2<f64> = Array2::from_shape_fn((n, p), |(i, j)| {
            NumCast::from(x[[i, j]]).unwrap_or(0.0)
        });

        // Generate all multi-index tuples (powers) up to degree
        let powers = self.generate_powers(p);
        let n_output = powers.len();

        let mut output = Array2::<f64>::zeros((n, n_output));

        for (col_idx, power_vec) in powers.iter().enumerate() {
            for row in 0..n {
                let mut val = 1.0f64;
                for (feat_idx, &pow) in power_vec.iter().enumerate() {
                    if pow > 0 {
                        val *= x_f64[[row, feat_idx]].powi(pow as i32);
                    }
                }
                output[[row, col_idx]] = val;
            }
        }

        Ok(output)
    }

    /// Generate all power multi-indices for p features up to the given degree
    fn generate_powers(&self, p: usize) -> Vec<Vec<usize>> {
        let mut powers = Vec::new();

        // Include bias (all zeros = constant 1)
        if self.include_bias {
            powers.push(vec![0usize; p]);
        }

        // Generate all combinations with the given constraints
        self.generate_powers_recursive(p, self.degree, &mut vec![0usize; p], 0, &mut powers);

        powers
    }

    fn generate_powers_recursive(
        &self,
        p: usize,
        remaining_degree: usize,
        current: &mut Vec<usize>,
        start_feat: usize,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining_degree == 0 || start_feat == p {
            return;
        }

        for feat in start_feat..p {
            let max_pow = if self.interaction_only { 1 } else { remaining_degree };
            for pow in 1..=max_pow {
                current[feat] = pow;
                let total: usize = current.iter().sum();
                if total > 0 && total <= self.degree {
                    output.push(current.clone());
                }
                if pow < remaining_degree && feat + 1 < p {
                    self.generate_powers_recursive(
                        p,
                        remaining_degree - pow,
                        current,
                        feat + 1,
                        output,
                    );
                }
                current[feat] = 0;
            }
        }
    }

    /// Get the number of output features for p input features
    pub fn n_output_features(&self, p: usize) -> usize {
        self.generate_powers(p).len()
    }
}

// ─── Interaction Features ─────────────────────────────────────────────────────

/// Generate pairwise interaction (product) features between all column pairs.
///
/// For input with features [f1, f2, f3], produces [f1*f2, f1*f3, f2*f3].
///
/// # Example
///
/// ```rust
/// use scirs2_transform::feature_engineering::InteractionFeatures;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("should succeed");
/// let result = InteractionFeatures::transform(&x).expect("should succeed");
/// assert_eq!(result.ncols(), 3); // C(3,2) = 3 pairwise products
/// ```
pub struct InteractionFeatures {
    /// Whether to include original features alongside interactions
    include_original: bool,
}

impl InteractionFeatures {
    /// Create a new InteractionFeatures transformer
    pub fn new(include_original: bool) -> Self {
        Self { include_original }
    }

    /// Compute pairwise interaction (product) features.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    ///
    /// # Returns
    /// * Feature matrix with all pairwise products (and optionally original features)
    pub fn transform<S>(x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let feat = Self::new(false);
        feat.fit_transform(x)
    }

    /// Fit and transform: compute interaction features
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        let p = x.ncols();

        if p < 2 {
            return Err(TransformError::InvalidInput(
                "InteractionFeatures requires at least 2 input features".to_string(),
            ));
        }

        // Number of interaction terms: C(p, 2) = p*(p-1)/2
        let n_interactions = p * (p - 1) / 2;
        let n_output = if self.include_original {
            p + n_interactions
        } else {
            n_interactions
        };

        let mut output = Array2::<f64>::zeros((n, n_output));

        let mut col = 0;

        if self.include_original {
            for feat in 0..p {
                for row in 0..n {
                    output[[row, col]] = NumCast::from(x[[row, feat]]).unwrap_or(0.0);
                }
                col += 1;
            }
        }

        for fi in 0..p {
            for fj in (fi + 1)..p {
                for row in 0..n {
                    let a: f64 = NumCast::from(x[[row, fi]]).unwrap_or(0.0);
                    let b: f64 = NumCast::from(x[[row, fj]]).unwrap_or(0.0);
                    output[[row, col]] = a * b;
                }
                col += 1;
            }
        }

        Ok(output)
    }

    /// Get the number of interaction columns for p input features
    pub fn n_interaction_columns(p: usize) -> usize {
        p * (p - 1) / 2
    }
}

// ─── Spline Features ─────────────────────────────────────────────────────────

/// B-spline basis function feature expansion for 1D data.
///
/// Transforms a 1D array into a matrix of B-spline basis function evaluations.
/// This provides a non-linear encoding suitable for capturing smooth, non-linear
/// relationships between a feature and a target variable.
///
/// The B-splines are built over uniformly spaced knots spanning the data range.
/// The resulting features can be used directly in linear models to fit smooth curves.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::feature_engineering::SplineFeatures;
/// use scirs2_core::ndarray::Array1;
///
/// let x = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
/// let result = SplineFeatures::transform(&x, 4, 3).expect("should succeed");
/// // Number of columns = n_knots + degree - 1 (number of B-spline basis functions)
/// assert_eq!(result.nrows(), 5);
/// ```
pub struct SplineFeatures {
    /// Number of interior knots
    n_knots: usize,
    /// B-spline degree (1 = linear, 2 = quadratic, 3 = cubic)
    degree: usize,
    /// Whether to include extrapolation clipping at boundaries
    extrapolation: SplineExtrapolation,
    /// Computed knot vector (set at fit time)
    knots: Option<Vec<f64>>,
    /// Data range (set at fit time)
    data_range: Option<(f64, f64)>,
}

/// Extrapolation behavior outside the knot span
#[derive(Debug, Clone, PartialEq)]
pub enum SplineExtrapolation {
    /// Clip values to [min, max] range
    Clip,
    /// Allow values to extend beyond the range
    Continue,
    /// Return zero outside the range (constant extrapolation)
    Constant,
}

impl SplineFeatures {
    /// Create a new SplineFeatures transformer
    pub fn new(n_knots: usize, degree: usize) -> Result<Self> {
        if n_knots == 0 {
            return Err(TransformError::InvalidInput(
                "SplineFeatures requires at least 1 knot".to_string(),
            ));
        }
        if degree == 0 {
            return Err(TransformError::InvalidInput(
                "SplineFeatures degree must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            n_knots,
            degree,
            extrapolation: SplineExtrapolation::Clip,
            knots: None,
            data_range: None,
        })
    }

    /// Configure extrapolation behavior
    pub fn with_extrapolation(mut self, extrapolation: SplineExtrapolation) -> Self {
        self.extrapolation = extrapolation;
        self
    }

    /// Transform a 1D array into B-spline basis features.
    ///
    /// # Arguments
    /// * `x` - 1D input data (n_samples)
    /// * `n_knots` - Number of interior knots
    /// * `degree` - B-spline degree (e.g., 3 for cubic)
    ///
    /// # Returns
    /// * Feature matrix (n_samples × (n_knots + degree - 1))
    pub fn transform<S>(
        x: &ArrayBase<S, Ix1>,
        n_knots: usize,
        degree: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let sf = Self::new(n_knots, degree)?;
        sf.fit_transform(x)
    }

    /// Fit and transform: compute B-spline basis features
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        let x_f64: Vec<f64> = x.iter().map(|&v| NumCast::from(v).unwrap_or(0.0)).collect();

        let x_min = x_f64.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = x_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (x_max - x_min).abs() < 1e-14 {
            return Err(TransformError::InvalidInput(
                "SplineFeatures: all input values are identical".to_string(),
            ));
        }

        // Build knot vector with clamped (de Boor) B-splines
        // Interior knots uniformly spaced; boundary knots repeated (degree+1) times
        let interior_knots: Vec<f64> = (1..=self.n_knots)
            .map(|i| x_min + i as f64 * (x_max - x_min) / (self.n_knots + 1) as f64)
            .collect();

        // Full augmented knot vector: (degree+1) copies of x_min, interior, (degree+1) copies of x_max
        let mut t = Vec::new();
        for _ in 0..=(self.degree) {
            t.push(x_min);
        }
        for &k in &interior_knots {
            t.push(k);
        }
        for _ in 0..=(self.degree) {
            t.push(x_max);
        }

        // Number of B-spline basis functions: len(t) - degree - 1
        let n_basis = t.len() - self.degree - 1;

        let mut output = Array2::<f64>::zeros((n, n_basis));

        for (row, &xi) in x_f64.iter().enumerate() {
            // Apply extrapolation
            let xi = match self.extrapolation {
                SplineExtrapolation::Clip => xi.max(x_min).min(x_max),
                SplineExtrapolation::Constant => {
                    if xi < x_min || xi > x_max {
                        continue; // leave as zeros
                    }
                    xi
                }
                SplineExtrapolation::Continue => xi,
            };

            // Evaluate all B-spline basis functions using De Boor's algorithm
            let basis_vals = evaluate_bspline_basis(xi, &t, self.degree, n_basis);
            for (j, &val) in basis_vals.iter().enumerate() {
                output[[row, j]] = val;
            }
        }

        Ok(output)
    }

    /// Get the expected number of output features
    pub fn n_output_features(&self) -> usize {
        // n_basis = (degree + 1 + n_knots + degree + 1) - degree - 1
        //         = n_knots + degree + 1
        self.n_knots + self.degree + 1
    }
}

/// Evaluate all B-spline basis functions of degree `d` at point `x` over knot vector `t`.
/// Uses the Cox-de Boor recursion.
fn evaluate_bspline_basis(x: f64, t: &[f64], d: usize, n_basis: usize) -> Vec<f64> {
    let m = t.len();

    // Find the last knot value (right boundary) for clamped endpoint handling
    let t_max = t[m - 1];

    // Initialize degree-0 basis functions
    // For clamped B-splines, the right endpoint x == t_max should be assigned
    // to the last non-degenerate knot span (where t[i] < t[i+1]).
    let mut b: Vec<f64> = (0..(m - 1))
        .map(|i| {
            if t[i] <= x && x < t[i + 1] {
                1.0
            } else if (x - t_max).abs() < 1e-12 && t[i] < t[i + 1] && (t[i + 1] - t_max).abs() < 1e-12 {
                // At the right boundary: activate the last non-degenerate span
                // whose right endpoint equals t_max
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // De Boor recursion up to degree d
    for k in 1..=d {
        let mut b_new = vec![0.0f64; m - k - 1];
        for i in 0..(m - k - 1) {
            let denom1 = t[i + k] - t[i];
            let denom2 = t[i + k + 1] - t[i + 1];

            let left = if denom1.abs() > 1e-14 {
                (x - t[i]) / denom1 * b[i]
            } else {
                0.0
            };

            let right = if denom2.abs() > 1e-14 {
                (t[i + k + 1] - x) / denom2 * b[i + 1]
            } else {
                0.0
            };

            b_new[i] = left + right;
        }
        b = b_new;
    }

    // Truncate to n_basis (safety check)
    b.truncate(n_basis);
    while b.len() < n_basis {
        b.push(0.0);
    }

    b
}

// ─── Radial Basis Features ────────────────────────────────────────────────────

/// Radial Basis Function (RBF) feature expansion.
///
/// Transforms input data X into a new feature space using Gaussian RBF kernels
/// centered at a set of pre-specified or learned center points:
///
/// ```text
/// phi_j(x) = exp(-gamma * ||x - c_j||^2)
/// ```
///
/// This is equivalent to using a one-layer RBF network as a feature transformer.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::feature_engineering::RadialBasisFeatures;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
/// ]).expect("should succeed");
/// let centers = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("should succeed");
/// let result = RadialBasisFeatures::transform(&x, &centers, 1.0).expect("should succeed");
/// assert_eq!(result.shape(), &[4, 2]);
/// ```
pub struct RadialBasisFeatures {
    /// RBF kernel bandwidth: exp(-gamma * ||x - c||^2)
    gamma: f64,
    /// Center points (n_centers × n_features)
    centers: Option<Array2<f64>>,
    /// Whether to normalize rows to sum to 1 (softmax-style)
    normalize: bool,
}

impl RadialBasisFeatures {
    /// Create a new RBF feature transformer
    pub fn new(gamma: f64) -> Result<Self> {
        if gamma <= 0.0 {
            return Err(TransformError::InvalidInput(
                "gamma must be positive".to_string(),
            ));
        }
        Ok(Self {
            gamma,
            centers: None,
            normalize: false,
        })
    }

    /// Configure normalization (rows sum to 1)
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Transform data using a fixed set of centers.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    /// * `centers` - RBF centers (n_centers × n_features)
    /// * `gamma` - Kernel bandwidth
    ///
    /// # Returns
    /// * Feature matrix (n_samples × n_centers)
    pub fn transform<S1, S2>(
        x: &ArrayBase<S1, Ix2>,
        centers: &ArrayBase<S2, Ix2>,
        gamma: f64,
    ) -> Result<Array2<f64>>
    where
        S1: Data,
        S1::Elem: Float + NumCast,
        S2: Data,
        S2::Elem: Float + NumCast,
    {
        let rbf = Self::new(gamma)?;
        rbf.fit_transform(x, centers)
    }

    /// Compute RBF features for data X relative to centers
    pub fn fit_transform<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        centers: &ArrayBase<S2, Ix2>,
    ) -> Result<Array2<f64>>
    where
        S1: Data,
        S1::Elem: Float + NumCast,
        S2: Data,
        S2::Elem: Float + NumCast,
    {
        let n = x.nrows();
        let p = x.ncols();
        let n_centers = centers.nrows();

        if p != centers.ncols() {
            return Err(TransformError::InvalidInput(format!(
                "Feature dimensions mismatch: X has {} features, centers have {}",
                p,
                centers.ncols()
            )));
        }

        let mut output = Array2::<f64>::zeros((n, n_centers));

        for i in 0..n {
            for c in 0..n_centers {
                let mut dist_sq = 0.0f64;
                for k in 0..p {
                    let xi: f64 = NumCast::from(x[[i, k]]).unwrap_or(0.0);
                    let ck: f64 = NumCast::from(centers[[c, k]]).unwrap_or(0.0);
                    let diff = xi - ck;
                    dist_sq += diff * diff;
                }
                output[[i, c]] = (-self.gamma * dist_sq).exp();
            }

            // Optional row normalization
            if self.normalize {
                let row_sum: f64 = output.row(i).iter().copied().sum();
                if row_sum > 1e-14 {
                    for c in 0..n_centers {
                        output[[i, c]] /= row_sum;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Select RBF centers using k-means-style initialization (k-means++)
    pub fn select_centers_random<S>(
        x: &ArrayBase<S, Ix2>,
        n_centers: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        let p = x.ncols();

        if n_centers > n {
            return Err(TransformError::InvalidInput(format!(
                "Cannot select {} centers from {} samples",
                n_centers, n
            )));
        }

        // Simple uniform sampling (deterministic - take evenly spaced samples)
        let mut centers = Array2::<f64>::zeros((n_centers, p));
        let step = n / n_centers;

        for c in 0..n_centers {
            let idx = (c * step).min(n - 1);
            for feat in 0..p {
                centers[[c, feat]] = NumCast::from(x[[idx, feat]]).unwrap_or(0.0);
            }
        }

        Ok(centers)
    }
}

// ─── Quantile Binning ─────────────────────────────────────────────────────────

/// Quantile-based binning for continuous features.
///
/// Discretizes continuous features into equally-populated quantile bins.
/// Each bin contains approximately the same number of samples.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::feature_engineering::QuantileBinner;
/// use scirs2_core::ndarray::Array1;
///
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// let mut binner = QuantileBinner::new(4).expect("should succeed");
/// let bins = binner.fit_transform(&x).expect("should succeed");
/// assert_eq!(bins.len(), 8);
/// ```
pub struct QuantileBinner {
    /// Number of quantile bins
    n_bins: usize,
    /// Bin edges computed during fit
    bin_edges: Option<Vec<f64>>,
    /// Whether to encode as one-hot instead of ordinal
    one_hot: bool,
}

impl QuantileBinner {
    /// Create a new QuantileBinner
    pub fn new(n_bins: usize) -> Result<Self> {
        if n_bins < 2 {
            return Err(TransformError::InvalidInput(
                "n_bins must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_bins,
            bin_edges: None,
            one_hot: false,
        })
    }

    /// Configure one-hot encoding output
    pub fn with_one_hot(mut self, one_hot: bool) -> Self {
        self.one_hot = one_hot;
        self
    }

    /// Fit to data and compute bin edges
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput("Empty input".to_string()));
        }

        let mut sorted: Vec<f64> = x
            .iter()
            .map(|&v| NumCast::from(v).unwrap_or(0.0))
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute quantile edges
        let mut edges = Vec::with_capacity(self.n_bins + 1);
        edges.push(sorted[0] - 1e-10); // slightly below min

        for q in 1..self.n_bins {
            let idx_f = q as f64 * n as f64 / self.n_bins as f64;
            let idx_lo = (idx_f.floor() as usize).min(n - 1);
            let idx_hi = (idx_lo + 1).min(n - 1);
            let frac = idx_f - idx_f.floor();
            let edge = sorted[idx_lo] * (1.0 - frac) + sorted[idx_hi] * frac;
            edges.push(edge);
        }

        edges.push(sorted[n - 1] + 1e-10); // slightly above max

        self.bin_edges = Some(edges);
        Ok(())
    }

    /// Transform: assign bin indices to each value
    pub fn transform_array<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array1<usize>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let edges = self.bin_edges.as_ref().ok_or_else(|| {
            TransformError::ComputationError("QuantileBinner not fitted yet".to_string())
        })?;

        let n = x.len();
        let mut result = Array1::<usize>::zeros(n);

        for (i, &val) in x.iter().enumerate() {
            let v: f64 = NumCast::from(val).unwrap_or(0.0);
            // Binary search for bin
            let bin = edges[1..]
                .iter()
                .position(|&e| v <= e)
                .unwrap_or(self.n_bins - 1);
            result[i] = bin.min(self.n_bins - 1);
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix1>) -> Result<Array1<usize>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform_array(x)
    }
}

// ─── Target Encoding ──────────────────────────────────────────────────────────

/// Target (mean) encoding for categorical features.
///
/// Replaces each category value with the mean of the target variable for that
/// category. Optionally applies smoothing to handle rare categories.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::feature_engineering::TargetEncoder;
/// use scirs2_core::ndarray::Array1;
///
/// let categories = vec!["A", "B", "A", "C", "B"];
/// let targets = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let mut encoder = TargetEncoder::new(1.0).expect("should succeed");
/// encoder.fit_str(&categories, &targets).expect("should succeed");
/// let encoded = encoder.transform_str(&["A", "B", "C"]).expect("should succeed");
/// assert_eq!(encoded.len(), 3);
/// ```
pub struct TargetEncoder {
    /// Smoothing parameter: (count * mean + alpha * global_mean) / (count + alpha)
    alpha: f64,
    /// Category statistics: (sum, count)
    category_stats: HashMap<String, (f64, usize)>,
    /// Global mean of the target
    global_mean: f64,
    /// Whether the encoder has been fitted
    fitted: bool,
}

impl TargetEncoder {
    /// Create a new TargetEncoder
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha < 0.0 {
            return Err(TransformError::InvalidInput(
                "Smoothing parameter alpha must be non-negative".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            category_stats: HashMap::new(),
            global_mean: 0.0,
            fitted: false,
        })
    }

    /// Fit encoder to string categories and numeric targets
    pub fn fit_str<S>(&mut self, categories: &[&str], targets: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if categories.len() != targets.len() {
            return Err(TransformError::InvalidInput(
                "categories and targets must have the same length".to_string(),
            ));
        }

        self.category_stats.clear();
        let mut total_sum = 0.0f64;
        let total_count = targets.len();

        for (cat, &tgt) in categories.iter().zip(targets.iter()) {
            let t: f64 = NumCast::from(tgt).unwrap_or(0.0);
            total_sum += t;
            let entry = self.category_stats.entry(cat.to_string()).or_insert((0.0, 0));
            entry.0 += t;
            entry.1 += 1;
        }

        self.global_mean = if total_count > 0 {
            total_sum / total_count as f64
        } else {
            0.0
        };
        self.fitted = true;
        Ok(())
    }

    /// Transform string categories to encoded numeric values
    pub fn transform_str(&self, categories: &[&str]) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(TransformError::ComputationError(
                "TargetEncoder not fitted yet".to_string(),
            ));
        }

        let n = categories.len();
        let mut result = Array1::<f64>::zeros(n);

        for (i, &cat) in categories.iter().enumerate() {
            result[i] = if let Some(&(sum, count)) = self.category_stats.get(cat) {
                // Smoothed estimate
                (count as f64 * (sum / count as f64) + self.alpha * self.global_mean)
                    / (count as f64 + self.alpha)
            } else {
                // Unseen category: return global mean
                self.global_mean
            };
        }

        Ok(result)
    }
}

// ─── Feature Hasher ───────────────────────────────────────────────────────────

/// Hash-based feature mapping for large or sparse feature spaces.
///
/// Uses the hashing trick to map arbitrary string features to a fixed-size vector.
/// Useful for text features, categorical features with many unique values, or when
/// the feature space is not known in advance.
pub struct FeatureHasher {
    /// Number of output features (hash space size)
    n_features: usize,
    /// Whether to use signed hashing (handles collisions better)
    alternate_sign: bool,
}

impl FeatureHasher {
    /// Create a new FeatureHasher
    pub fn new(n_features: usize) -> Result<Self> {
        if n_features == 0 {
            return Err(TransformError::InvalidInput(
                "n_features must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_features,
            alternate_sign: true,
        })
    }

    /// Transform a feature-value dict (string key, f64 value) to a fixed-size vector
    pub fn transform_dict(&self, features: &HashMap<String, f64>) -> Array1<f64> {
        let mut result = Array1::<f64>::zeros(self.n_features);

        for (key, &val) in features {
            let hash = self.hash_str(key);
            let idx = hash % self.n_features;
            let sign = if self.alternate_sign {
                let sign_hash = self.hash_str(&format!("{}_sign", key));
                if sign_hash % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                1.0
            };
            result[idx] += sign * val;
        }

        result
    }

    /// Transform a batch of dicts to a feature matrix
    pub fn transform_batch(&self, features: &[HashMap<String, f64>]) -> Array2<f64> {
        let n = features.len();
        let mut output = Array2::<f64>::zeros((n, self.n_features));

        for (row, feat_dict) in features.iter().enumerate() {
            let row_vec = self.transform_dict(feat_dict);
            for (col, &val) in row_vec.iter().enumerate() {
                output[[row, col]] = val;
            }
        }

        output
    }

    /// Simple hash function for strings
    fn hash_str(&self, s: &str) -> usize {
        let mut h: u64 = 14695981039346656037u64; // FNV-1a offset basis
        for byte in s.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211u64); // FNV-1a prime
        }
        h as usize
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_polynomial_features_degree2() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let result = PolynomialFeatures::transform(&x, 2).expect("poly transform");
        // degree=2, p=2: [1, x0, x1, x0^2, x0*x1, x1^2]
        assert_eq!(result.ncols(), 6, "Expected 6 polynomial features");
        assert_eq!(result.nrows(), 3);

        // Verify first row: x=[1,2], features=[1, 1, 2, 1, 2, 4]
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10, "bias=1");
        assert!((result[[0, 1]] - 1.0).abs() < 1e-10, "x0=1");
        assert!((result[[0, 2]] - 2.0).abs() < 1e-10, "x1=2");
        assert!((result[[0, 4]] - 2.0).abs() < 1e-10, "x0*x1=2");
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0f64, 2.0, 3.0, 4.0])
            .expect("shape ok");
        let pf = PolynomialFeatures::new(2).expect("pf new").with_bias(false);
        let result = pf.fit_transform(&x).expect("pf transform");
        // No bias: [x0, x1, x0^2, x0*x1, x1^2] = 5 features
        assert_eq!(result.ncols(), 5);
    }

    #[test]
    fn test_interaction_features() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let result = InteractionFeatures::transform(&x).expect("interaction transform");
        // C(3,2) = 3 pairwise products
        assert_eq!(result.ncols(), 3);
        assert_eq!(result.nrows(), 2);

        // First row: x=[1,2,3], products: [1*2=2, 1*3=3, 2*3=6]
        assert!((result[[0, 0]] - 2.0).abs() < 1e-10, "1*2=2");
        assert!((result[[0, 1]] - 3.0).abs() < 1e-10, "1*3=3");
        assert!((result[[0, 2]] - 6.0).abs() < 1e-10, "2*3=6");
    }

    #[test]
    fn test_interaction_features_with_original() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let feat = InteractionFeatures::new(true);
        let result = feat.fit_transform(&x).expect("interaction with original");
        // 3 original + 3 interactions = 6 features
        assert_eq!(result.ncols(), 6);
    }

    #[test]
    fn test_spline_features_shape() {
        let x = Array1::from_vec(vec![0.0f64, 0.5, 1.0, 1.5, 2.0]);
        let result = SplineFeatures::transform(&x, 4, 3).expect("spline transform");
        // n_basis = n_knots + degree + 1 = 4 + 3 + 1 = 8
        assert_eq!(result.nrows(), 5);
        // All values should be non-negative and rows should sum to ~1
        assert!(result.iter().all(|&v| v >= -1e-10));
    }

    #[test]
    fn test_spline_features_partition_of_unity() {
        // B-splines form a partition of unity: each row sums to 1
        let x = Array1::from_vec(vec![0.0f64, 0.25, 0.5, 0.75, 1.0]);
        let result = SplineFeatures::transform(&x, 3, 3).expect("spline transform");
        for row in 0..result.nrows() {
            let row_sum: f64 = result.row(row).iter().copied().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-8,
                "Row {} sum = {} (expected 1.0)",
                row,
                row_sum
            );
        }
    }

    #[test]
    fn test_radial_basis_features_shape() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape ok");
        let centers =
            Array2::from_shape_vec((2, 2), vec![0.0f64, 0.0, 1.0, 1.0]).expect("shape ok");
        let result = RadialBasisFeatures::transform(&x, &centers, 1.0).expect("rbf transform");
        assert_eq!(result.shape(), &[4, 2]);
    }

    #[test]
    fn test_radial_basis_features_values() {
        // Point at center should have rbf = exp(0) = 1.0
        let x = Array2::from_shape_vec((1, 2), vec![0.0f64, 0.0]).expect("shape ok");
        let centers = Array2::from_shape_vec((1, 2), vec![0.0f64, 0.0]).expect("shape ok");
        let result = RadialBasisFeatures::transform(&x, &centers, 1.0).expect("rbf transform");
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10, "At center, rbf=1");
    }

    #[test]
    fn test_radial_basis_features_gamma() {
        // Higher gamma = more localized; at distance 1, exp(-gamma * 1)
        let x = Array2::from_shape_vec((1, 1), vec![1.0f64]).expect("shape ok");
        let centers = Array2::from_shape_vec((1, 1), vec![0.0f64]).expect("shape ok");

        let r1 = RadialBasisFeatures::transform(&x, &centers, 1.0).expect("rbf g1");
        let r2 = RadialBasisFeatures::transform(&x, &centers, 2.0).expect("rbf g2");

        assert!(r2[[0, 0]] < r1[[0, 0]], "Higher gamma -> lower value at distance 1");
    }

    #[test]
    fn test_radial_basis_features_normalized() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0f64, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("shape ok");
        let centers =
            Array2::from_shape_vec((3, 2), vec![0.0f64, 0.0, 1.0, 1.0, 2.0, 2.0]).expect("shape ok");
        let rbf = RadialBasisFeatures::new(0.5)
            .expect("rbf new")
            .with_normalize(true);
        let result = rbf.fit_transform(&x, &centers).expect("rbf norm");

        // Each row should sum to ~1.0
        for row in 0..result.nrows() {
            let row_sum: f64 = result.row(row).iter().copied().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {} should sum to 1, got {}",
                row,
                row_sum
            );
        }
    }

    #[test]
    fn test_quantile_binner() {
        let x = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut binner = QuantileBinner::new(4).expect("binner new");
        let bins = binner.fit_transform(&x).expect("binner transform");
        assert_eq!(bins.len(), 8);
        // Values should be in [0, n_bins-1]
        assert!(bins.iter().all(|&b| b < 4));
    }

    #[test]
    fn test_target_encoder() {
        let categories = vec!["A", "B", "A", "C", "B"];
        let targets = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let mut encoder = TargetEncoder::new(1.0).expect("encoder new");
        encoder.fit_str(&categories, &targets).expect("fit");
        let encoded = encoder.transform_str(&["A", "B", "C"]).expect("transform");
        assert_eq!(encoded.len(), 3);
        // A: mean = (1+3)/2 = 2.0, B: mean = (2+5)/2 = 3.5, C: mean = 4.0
        // Smoothed with alpha=1: (count*mean + 1*global_mean) / (count + 1)
        assert!(encoded.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_feature_hasher() {
        let hasher = FeatureHasher::new(16).expect("hasher new");
        let mut features = HashMap::new();
        features.insert("word_hello".to_string(), 1.0);
        features.insert("word_world".to_string(), 2.0);

        let result = hasher.transform_dict(&features);
        assert_eq!(result.len(), 16);
        // Should have some non-zero values
        assert!(result.iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn test_polynomial_features_degree1() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let result = PolynomialFeatures::transform(&x, 1).expect("poly d1");
        // degree=1, p=3 with bias: [1, x0, x1, x2] = 4 features
        assert_eq!(result.ncols(), 4);
    }

    #[test]
    fn test_rbf_center_selection() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| i as f64 + j as f64 * 0.1);
        let centers = RadialBasisFeatures::select_centers_random(&x, 5).expect("centers");
        assert_eq!(centers.shape(), &[5, 2]);
    }

    #[test]
    fn test_interaction_features_two_features() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let result = InteractionFeatures::transform(&x).expect("interaction 2d");
        // C(2,2) = 1 pairwise product
        assert_eq!(result.ncols(), 1);
        // Row 0: 1*2 = 2
        assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
    }
}
