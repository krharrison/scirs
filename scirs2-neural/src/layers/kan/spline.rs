//! B-spline basis functions and activations for KAN layers.
//!
//! Implements Cox-de Boor recursion for B-spline evaluation and learnable
//! spline activations used in Kolmogorov-Arnold Networks.

use scirs2_core::ndarray::{Array1, Array2};

use crate::NeuralError;

/// Configuration for B-spline activations.
#[derive(Debug, Clone)]
pub struct SplineConfig {
    /// Number of basis functions (controls expressiveness).
    pub n_basis: usize,
    /// Polynomial order (default 3 = cubic).
    pub order: usize,
    /// Lower bound of input domain.
    pub domain_low: f64,
    /// Upper bound of input domain.
    pub domain_high: f64,
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            n_basis: 8,
            order: 3,
            domain_low: -1.0,
            domain_high: 1.0,
        }
    }
}

impl SplineConfig {
    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), NeuralError> {
        if self.n_basis == 0 {
            return Err(NeuralError::InvalidArgument(
                "n_basis must be > 0".to_string(),
            ));
        }
        if self.order == 0 {
            return Err(NeuralError::InvalidArgument(
                "order must be > 0".to_string(),
            ));
        }
        if self.n_basis <= self.order {
            return Err(NeuralError::InvalidArgument(format!(
                "n_basis ({}) must be > order ({})",
                self.n_basis, self.order
            )));
        }
        if self.domain_low >= self.domain_high {
            return Err(NeuralError::InvalidArgument(
                "domain_low must be < domain_high".to_string(),
            ));
        }
        Ok(())
    }
}

/// B-spline basis evaluator using Cox-de Boor recursion.
///
/// Constructs a clamped uniform B-spline basis on [domain_low, domain_high].
/// The knot vector is clamped at both ends (first `order+1` knots equal domain_low,
/// last `order+1` knots equal domain_high) with uniformly spaced interior knots.
#[derive(Debug, Clone)]
pub struct BSplineBasis {
    knots: Vec<f64>,
    n_basis: usize,
    order: usize,
}

impl BSplineBasis {
    /// Create a clamped uniform B-spline basis from a [`SplineConfig`].
    ///
    /// Returns an error if the config is invalid.
    pub fn new(config: &SplineConfig) -> Result<Self, NeuralError> {
        config.validate()?;
        let n = config.n_basis;
        let k = config.order;
        // Total knots = n + k + 1
        // Interior knots (not clamped) = n + k + 1 - 2*(k+1) = n - k - 1
        let n_interior = n - k - 1;
        let mut knots = vec![config.domain_low; k + 1];
        if n_interior > 0 {
            let step = (config.domain_high - config.domain_low) / (n_interior + 1) as f64;
            for i in 1..=n_interior {
                knots.push(config.domain_low + i as f64 * step);
            }
        }
        for _ in 0..=k {
            knots.push(config.domain_high);
        }
        debug_assert_eq!(knots.len(), n + k + 1, "Knot vector length mismatch");
        Ok(Self {
            knots,
            n_basis: n,
            order: k,
        })
    }

    /// Evaluate all `n_basis` B-spline basis functions at a single point `x`.
    ///
    /// Uses the Cox-de Boor recursion. Returns an `Array1<f64>` of length `n_basis`.
    pub fn evaluate(&self, x: f64) -> Array1<f64> {
        let k = self.order;
        let n = self.n_basis;
        // Clamp x to be strictly inside the domain to handle right boundary.
        let x_c = x.clamp(
            self.knots[0],
            self.knots[self.knots.len() - 1] - f64::EPSILON * 1e3,
        );

        // Working buffer of length n + k (order-0 basis for n+k spans)
        let mut b = vec![0.0f64; n + k];

        // Order-0: indicator on each knot span
        for i in 0..n + k {
            if self.knots[i] <= x_c && x_c < self.knots[i + 1] {
                b[i] = 1.0;
            }
        }

        // Recursion up to desired order
        for d in 1..=k {
            // Process in-place from left to right; result has n + k - d entries
            for i in 0..n + k - d {
                let denom1 = self.knots[i + d] - self.knots[i];
                let t1 = if denom1.abs() > 1e-14 {
                    (x_c - self.knots[i]) / denom1 * b[i]
                } else {
                    0.0
                };
                let denom2 = self.knots[i + d + 1] - self.knots[i + 1];
                let t2 = if denom2.abs() > 1e-14 {
                    (self.knots[i + d + 1] - x_c) / denom2 * b[i + 1]
                } else {
                    0.0
                };
                b[i] = t1 + t2;
            }
        }

        Array1::from_vec(b[..n].to_vec())
    }

    /// Evaluate basis functions at multiple points.
    ///
    /// Returns `Array2<f64>` of shape `[n_points, n_basis]`.
    pub fn evaluate_batch(&self, xs: &Array1<f64>) -> Array2<f64> {
        let n_pts = xs.len();
        let mut result = Array2::zeros((n_pts, self.n_basis));
        for (i, &x) in xs.iter().enumerate() {
            let row = self.evaluate(x);
            result.row_mut(i).assign(&row);
        }
        result
    }

    /// Number of basis functions.
    pub fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Polynomial order of the spline.
    pub fn order(&self) -> usize {
        self.order
    }
}

/// Learnable B-spline activation function.
///
/// Computes `phi(x) = sum_i coeff_i * B_{i,k}(x)` where `B_{i,k}` are B-spline
/// basis functions of order `k`. The coefficients are the learnable parameters.
#[derive(Debug, Clone)]
pub struct BSplineActivation {
    basis: BSplineBasis,
    /// Learnable coefficients, shape `[n_basis]`.
    pub coefficients: Array1<f64>,
}

impl BSplineActivation {
    /// Create a new `BSplineActivation` with zero-initialized coefficients.
    pub fn new(config: &SplineConfig) -> Result<Self, NeuralError> {
        let basis = BSplineBasis::new(config)?;
        let n = basis.n_basis();
        let coefficients = Array1::zeros(n);
        Ok(Self { basis, coefficients })
    }

    /// Evaluate the activation at a single point.
    pub fn evaluate(&self, x: f64) -> f64 {
        let basis_vals = self.basis.evaluate(x);
        basis_vals.dot(&self.coefficients)
    }

    /// Evaluate the activation at multiple points.
    ///
    /// Returns `Array1<f64>` of the same length as `xs`.
    pub fn evaluate_batch(&self, xs: &Array1<f64>) -> Array1<f64> {
        let b = self.basis.evaluate_batch(xs);
        b.dot(&self.coefficients)
    }

    /// Gradient of the output w.r.t. the coefficients at a single point.
    ///
    /// This equals the vector of basis function values at `x`, since
    /// `d/d(coeff_i) phi(x) = B_{i,k}(x)`.
    pub fn grad_coefficients(&self, x: f64) -> Array1<f64> {
        self.basis.evaluate(x)
    }

    /// Number of learnable parameters (= `n_basis`).
    pub fn n_params(&self) -> usize {
        self.coefficients.len()
    }

    /// Reference to the underlying basis evaluator.
    pub fn basis(&self) -> &BSplineBasis {
        &self.basis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SplineConfig {
        SplineConfig::default()
    }

    /// Partition of unity: all basis functions should sum to 1 at any point.
    #[test]
    fn basis_sums_to_one() {
        let config = default_config();
        let basis = BSplineBasis::new(&config).expect("valid config");
        for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let vals = basis.evaluate(x);
            let total: f64 = vals.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-10,
                "Partition of unity failed at x={x}: sum={total}"
            );
        }
    }

    /// Basis values at both endpoints should be non-negative and sum to 1.
    #[test]
    fn evaluate_at_endpoints() {
        let config = default_config();
        let basis = BSplineBasis::new(&config).expect("valid config");
        for &x in &[config.domain_low, config.domain_high - 1e-10] {
            let vals = basis.evaluate(x);
            for &v in vals.iter() {
                assert!(v >= -1e-12, "Negative basis value at x={x}: {v}");
            }
            let total: f64 = vals.iter().sum();
            assert!((total - 1.0).abs() < 1e-10, "Sum != 1 at x={x}: {total}");
        }
    }

    /// Zero coefficients should yield zero output everywhere.
    #[test]
    fn activation_zero_coefficients() {
        let config = default_config();
        let act = BSplineActivation::new(&config).expect("valid config");
        for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0 - 1e-10] {
            let val = act.evaluate(x);
            assert!(val.abs() < 1e-14, "Expected 0 at x={x}, got {val}");
        }
    }

    /// With appropriate coefficients, should recover an approximately linear function.
    ///
    /// For a linear function f(x) = x on [-1, 1], the spline coefficients are the
    /// Greville abscissae (averages of consecutive knots). We verify that the spline
    /// with these coefficients is close to the identity.
    #[test]
    fn activation_linear_approximation() {
        let config = default_config();
        let basis = BSplineBasis::new(&config).expect("valid config");
        let n = basis.n_basis();
        let k = config.order;
        // Greville abscissae: xi_i = (t_{i+1} + ... + t_{i+k}) / k
        // Using the basis's knot vector via evaluating at Greville points
        // For simplicity, use least-squares fit to identity function
        let test_pts: Vec<f64> = (0..100)
            .map(|i| -0.99 + i as f64 * (1.98 / 99.0))
            .collect();
        let xs = Array1::from_vec(test_pts.clone());
        let b_mat = basis.evaluate_batch(&xs); // [100, n_basis]

        // Solve B @ c = xs (least squares) via normal equations
        let bt_b = b_mat.t().dot(&b_mat); // [n, n]
        let bt_y = b_mat.t().dot(&xs); // [n]

        // Simple diagonal-regularized solve (for test robustness)
        let reg = 1e-8;
        let mut coeffs = Array1::zeros(n);
        // Gauss-Seidel / Jacobi iteration for n iterations
        for _ in 0..1000 {
            for i in 0..n {
                let diag = bt_b[(i, i)] + reg;
                let mut rhs = bt_y[i];
                for j in 0..n {
                    if j != i {
                        rhs -= bt_b[(i, j)] * coeffs[j];
                    }
                }
                coeffs[i] = rhs / diag;
            }
        }

        let mut act = BSplineActivation::new(&config).expect("valid config");
        act.coefficients = coeffs;

        let max_err = test_pts
            .iter()
            .map(|&x| (act.evaluate(x) - x).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 0.05, "Linear approx error too large: {max_err}");
    }

    /// Batch evaluation should match element-wise evaluation.
    #[test]
    fn batch_evaluation_consistent() {
        let config = default_config();
        let mut act = BSplineActivation::new(&config).expect("valid config");
        // Set some non-trivial coefficients
        for (i, c) in act.coefficients.iter_mut().enumerate() {
            *c = (i as f64 * 0.3).sin();
        }
        let xs = Array1::from_vec(vec![-0.8, -0.3, 0.0, 0.4, 0.9]);
        let batch_out = act.evaluate_batch(&xs);
        for (i, &x) in xs.iter().enumerate() {
            let single = act.evaluate(x);
            assert!(
                (single - batch_out[i]).abs() < 1e-14,
                "Mismatch at i={i}: single={single}, batch={}",
                batch_out[i]
            );
        }
    }

    /// Order-1 splines (hat/tent functions) should still satisfy partition of unity.
    #[test]
    fn bspline_basis_order_1() {
        let config = SplineConfig {
            n_basis: 4,
            order: 1,
            domain_low: 0.0,
            domain_high: 1.0,
        };
        let basis = BSplineBasis::new(&config).expect("valid config");
        for i in 0..20 {
            let x = i as f64 / 20.0;
            let vals = basis.evaluate(x);
            let total: f64 = vals.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-10,
                "Order-1 partition of unity failed at x={x}: sum={total}"
            );
        }
    }

    /// Invalid config should return an error.
    #[test]
    fn invalid_config_n_basis_le_order() {
        let config = SplineConfig {
            n_basis: 2,
            order: 3,
            ..SplineConfig::default()
        };
        assert!(BSplineBasis::new(&config).is_err());
    }
}
