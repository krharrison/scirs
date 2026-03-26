//! Padé-type rational activation functions for KAN layers.
//!
//! A rational activation computes `phi(x) = P(x) / Q(x)` where `P` and `Q` are
//! polynomials with learnable coefficients. The denominator is constrained to be
//! strictly positive by using `Q(x) = 1 + |Q_raw(x)|`.

use scirs2_core::ndarray::Array1;

use crate::NeuralError;

/// Configuration for rational activations.
#[derive(Debug, Clone)]
pub struct RationalConfig {
    /// Degree of the numerator polynomial (default 4).
    pub p_degree: usize,
    /// Degree of the denominator polynomial raw part (default 4).
    pub q_degree: usize,
}

impl Default for RationalConfig {
    fn default() -> Self {
        Self {
            p_degree: 4,
            q_degree: 4,
        }
    }
}

impl RationalConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), NeuralError> {
        if self.p_degree == 0 && self.q_degree == 0 {
            return Err(NeuralError::InvalidArgument(
                "At least one of p_degree or q_degree must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Rational activation function: `phi(x) = P(x) / (1 + |Q_raw(x)|)`.
///
/// `P(x)` and `Q_raw(x)` are evaluated via Horner's method. The denominator
/// `1 + |Q_raw(x)|` is always strictly positive, guaranteeing a well-defined
/// output for all finite inputs.
#[derive(Debug, Clone)]
pub struct RationalActivation {
    /// Learnable numerator coefficients `[p_0, p_1, ..., p_{p_degree}]`.
    pub p_coeffs: Array1<f64>,
    /// Learnable denominator raw coefficients `[q_0, q_1, ..., q_{q_degree}]`.
    pub q_coeffs: Array1<f64>,
    config: RationalConfig,
}

impl RationalActivation {
    /// Create a new `RationalActivation` with zero-initialized coefficients.
    pub fn new(config: &RationalConfig) -> Result<Self, NeuralError> {
        config.validate()?;
        let p = Array1::zeros(config.p_degree + 1);
        let q = Array1::zeros(config.q_degree + 1);
        Ok(Self {
            p_coeffs: p,
            q_coeffs: q,
            config: config.clone(),
        })
    }

    /// Evaluate a polynomial with the given coefficients at `x` using Horner's method.
    ///
    /// Coefficients are ordered `[c_0, c_1, ..., c_n]` (ascending power).
    fn eval_poly(coeffs: &Array1<f64>, x: f64) -> f64 {
        let mut result = 0.0f64;
        for &c in coeffs.iter().rev() {
            result = result * x + c;
        }
        result
    }

    /// Evaluate the rational activation at a single point `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        let p = Self::eval_poly(&self.p_coeffs, x);
        let q_raw = Self::eval_poly(&self.q_coeffs, x);
        // Denominator stays strictly positive: 1 + |q_raw|
        let q = 1.0 + q_raw.abs();
        p / q
    }

    /// Evaluate the rational activation at multiple points.
    pub fn evaluate_batch(&self, xs: &Array1<f64>) -> Array1<f64> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Number of learnable parameters (`p_degree + 1 + q_degree + 1`).
    pub fn n_params(&self) -> usize {
        self.p_coeffs.len() + self.q_coeffs.len()
    }

    /// Read-only reference to the config.
    pub fn config(&self) -> &RationalConfig {
        &self.config
    }

    /// Gradient of `phi(x)` with respect to `p_coeffs`.
    ///
    /// `d phi / d p_i = x^i / (1 + |Q_raw(x)|)`
    pub fn grad_p_coeffs(&self, x: f64) -> Array1<f64> {
        let q_raw = Self::eval_poly(&self.q_coeffs, x);
        let denom = 1.0 + q_raw.abs();
        let n = self.p_coeffs.len();
        let mut grad = Array1::zeros(n);
        let mut xi = 1.0f64;
        for i in 0..n {
            grad[i] = xi / denom;
            xi *= x;
        }
        grad
    }

    /// Gradient of `phi(x)` with respect to `q_coeffs`.
    ///
    /// Using the chain rule:
    /// `d phi / d q_i = -P(x) * sign(Q_raw(x)) * x^i / (1 + |Q_raw(x)|)^2`
    pub fn grad_q_coeffs(&self, x: f64) -> Array1<f64> {
        let p = Self::eval_poly(&self.p_coeffs, x);
        let q_raw = Self::eval_poly(&self.q_coeffs, x);
        let denom = 1.0 + q_raw.abs();
        let sign_q = if q_raw >= 0.0 { 1.0 } else { -1.0 };
        let factor = -p * sign_q / (denom * denom);
        let m = self.q_coeffs.len();
        let mut grad = Array1::zeros(m);
        let mut xi = 1.0f64;
        for i in 0..m {
            grad[i] = factor * xi;
            xi *= x;
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RationalConfig {
        RationalConfig::default()
    }

    /// With all-zero coefficients, P(x) = 0, so phi(x) = 0 everywhere.
    #[test]
    fn zero_coefficients_output_zero() {
        let act = RationalActivation::new(&default_config()).expect("valid config");
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let val = act.evaluate(x);
            assert!(val.abs() < 1e-14, "Expected 0 at x={x}, got {val}");
        }
    }

    /// Setting p_coeffs = [1, 0, 0, 0, 0] yields phi(x) = 1 / (1 + |Q_raw(x)|).
    /// With q_coeffs = 0 too, we get phi(x) = 1 everywhere.
    #[test]
    fn constant_numerator() {
        let mut act = RationalActivation::new(&default_config()).expect("valid config");
        act.p_coeffs[0] = 1.0; // p(x) = 1
        // q_coeffs stay zero => q_raw = 0 => denom = 1
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let val = act.evaluate(x);
            assert!(
                (val - 1.0).abs() < 1e-14,
                "Expected 1 at x={x}, got {val}"
            );
        }
    }

    /// Denominator must always be strictly positive.
    #[test]
    fn denominator_stays_positive() {
        let mut act = RationalActivation::new(&default_config()).expect("valid config");
        // Set some non-trivial q coefficients
        for (i, c) in act.q_coeffs.iter_mut().enumerate() {
            *c = (i as f64 + 1.0) * 0.7;
        }
        act.p_coeffs[0] = 1.0;
        for i in -50i64..=50 {
            let x = i as f64 * 0.1;
            let q_raw = RationalActivation::eval_poly(&act.q_coeffs, x);
            let denom = 1.0 + q_raw.abs();
            assert!(
                denom > 0.0,
                "Denominator non-positive at x={x}: {denom}"
            );
            // Also verify the output is finite
            let val = act.evaluate(x);
            assert!(val.is_finite(), "Non-finite output at x={x}: {val}");
        }
    }

    /// Batch evaluation must match element-wise evaluation exactly.
    #[test]
    fn batch_matches_element_wise() {
        let mut act = RationalActivation::new(&default_config()).expect("valid config");
        for (i, c) in act.p_coeffs.iter_mut().enumerate() {
            *c = (i as f64 * 0.4 + 0.1).sin();
        }
        for (i, c) in act.q_coeffs.iter_mut().enumerate() {
            *c = (i as f64 * 0.2 - 0.3).cos() * 0.5;
        }
        let xs = Array1::from_vec(vec![-1.5, -0.7, 0.0, 0.3, 1.1, 2.0]);
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
}
