//! Bijective parameter transforms for ADVI.
//!
//! Maps constrained parameters to unconstrained real space so that
//! the variational Gaussian is defined over the entire real line.
//! Each transform comes with its inverse and log-absolute-Jacobian
//! (change-of-variables adjustment to the ELBO).

use crate::error::{StatsError, StatsResult};

use super::types::ConstraintType;

// ============================================================================
// Standalone transform functions (convenient API)
// ============================================================================

/// Map a positive real θ > 0 to the unconstrained real line: η = log(θ).
///
/// Inverse: θ = exp(η).
#[inline]
pub fn log_transform(x: f64) -> f64 {
    x.ln()
}

/// Map a bounded parameter θ ∈ (lo, hi) to the unconstrained real line
/// via the scaled logit: η = logit((θ − lo) / (hi − lo)).
///
/// Inverse: θ = lo + (hi − lo) · sigmoid(η).
#[inline]
pub fn logit_transform(x: f64, lo: f64, hi: f64) -> f64 {
    let s = (x - lo) / (hi - lo);
    (s / (1.0 - s)).ln()
}

/// Map a real-valued vector to the probability simplex via softmax:
/// p_i = exp(x_i) / Σ_j exp(x_j).
///
/// Numerically stable: subtract max before exponentiating.
pub fn softmax_transform(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-absolute Jacobian for the log transform.
///
/// θ = exp(η) ⟹ |dθ/dη| = exp(η) = θ ⟹ log|J| = η = log(θ).
///
/// Equivalently, the *correction* term for the ELBO is log|dη/dθ| = -log(θ).
#[inline]
pub fn log_jacobian_positive(x: f64) -> f64 {
    // Correction added to log p(θ): log|dη/dθ| where η = log(θ).
    // dη/dθ = 1/θ  ⟹  log|dη/dθ| = -log(θ)
    -x.ln()
}

/// Log-absolute Jacobian for the bounded (logit) transform.
///
/// η = logit((θ − lo)/(hi − lo))
/// |dη/dθ| = 1 / [(θ − lo)(hi − θ)] · (hi − lo)
/// log|dη/dθ| = -log(θ − lo) - log(hi − θ) + log(hi − lo)
#[inline]
pub fn log_jacobian_bounded(x: f64, lo: f64, hi: f64) -> f64 {
    let range = hi - lo;
    -(x - lo).ln() - (hi - x).ln() + range.ln()
}

// ============================================================================
// TransformSpec — per-parameter specification
// ============================================================================

/// Per-parameter transform specification: constraint type + bijective map.
///
/// A `TransformSpec` pairs a `ConstraintType` with the forward/inverse
/// transform functions and log-Jacobian for use inside ADVI.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformSpec {
    /// The constraint type for this parameter
    pub constraint: ConstraintType,
}

impl TransformSpec {
    /// Create a `TransformSpec` from a `ConstraintType`.
    pub fn new(constraint: ConstraintType) -> Self {
        Self { constraint }
    }

    /// Create an unconstrained (identity) spec.
    pub fn unconstrained() -> Self {
        Self::new(ConstraintType::Unconstrained)
    }

    /// Create a positive-valued spec (log transform).
    pub fn positive() -> Self {
        Self::new(ConstraintType::Positive)
    }

    /// Create a bounded spec for θ ∈ (lo, hi).
    pub fn bounded(lo: f64, hi: f64) -> Self {
        Self::new(ConstraintType::Bounded { lo, hi })
    }

    /// Forward transform: constrained θ → unconstrained η.
    ///
    /// Returns an error if the value violates the constraint.
    pub fn to_unconstrained(&self, theta: f64) -> StatsResult<f64> {
        match &self.constraint {
            ConstraintType::Unconstrained => Ok(theta),
            ConstraintType::Positive => {
                if theta <= 0.0 {
                    return Err(StatsError::invalid_argument(format!(
                        "Positive constraint violated: θ = {} must be > 0",
                        theta
                    )));
                }
                Ok(log_transform(theta))
            }
            ConstraintType::Bounded { lo, hi } => {
                if theta <= *lo || theta >= *hi {
                    return Err(StatsError::invalid_argument(format!(
                        "Bounded constraint violated: θ = {} must lie in ({}, {})",
                        theta, lo, hi
                    )));
                }
                Ok(logit_transform(theta, *lo, *hi))
            }
            ConstraintType::Simplex => {
                // For simplex, we use the additive log-ratio (ALR) of the last
                // element as reference; but for a single scalar we just return identity.
                Ok(theta)
            }
        }
    }

    /// Inverse transform: unconstrained η → constrained θ.
    pub fn to_constrained(&self, eta: f64) -> f64 {
        match &self.constraint {
            ConstraintType::Unconstrained => eta,
            ConstraintType::Positive => eta.exp(),
            ConstraintType::Bounded { lo, hi } => {
                let s = sigmoid(eta);
                lo + (hi - lo) * s
            }
            ConstraintType::Simplex => eta,
        }
    }

    /// Log-absolute-Jacobian of the inverse transform (η → θ),
    /// i.e., log|dθ/dη|.  Added to log p(θ) when computing the ELBO.
    pub fn log_jacobian_inverse(&self, eta: f64) -> f64 {
        match &self.constraint {
            ConstraintType::Unconstrained => 0.0,
            ConstraintType::Positive => {
                // θ = exp(η) ⟹ dθ/dη = exp(η) ⟹ log|J| = η
                eta
            }
            ConstraintType::Bounded { lo, hi } => {
                // θ = lo + (hi - lo) σ(η)
                // dθ/dη = (hi - lo) σ(η)(1 − σ(η))
                let range = hi - lo;
                let s = sigmoid(eta);
                range.ln() + s.ln() + (1.0 - s).ln()
            }
            ConstraintType::Simplex => 0.0,
        }
    }
}

/// Numerically stable sigmoid: σ(x) = 1 / (1 + exp(−x))
#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_log_transform_roundtrip() {
        for x in [0.001, 0.1, 1.0, 10.0, 1000.0] {
            let eta = log_transform(x);
            let recovered = eta.exp();
            assert!(
                (recovered - x).abs() < EPS * x.max(1.0),
                "Roundtrip failed for x={}: got {}",
                x,
                recovered
            );
        }
    }

    #[test]
    fn test_logit_transform_range() {
        let lo = -2.0;
        let hi = 5.0;
        // For x in (lo, hi), output should be finite real number
        for x in [-1.5, 0.0, 1.0, 3.0, 4.5] {
            let eta = logit_transform(x, lo, hi);
            assert!(
                eta.is_finite(),
                "logit_transform({}, {}, {}) = {} is not finite",
                x,
                lo,
                hi,
                eta
            );
        }
        // Edge: x approaching lo should give -∞, x approaching hi should give +∞
        let near_lo = logit_transform(lo + 1e-10, lo, hi);
        let near_hi = logit_transform(hi - 1e-10, lo, hi);
        assert!(near_lo < -20.0, "Near lo should give large negative value");
        assert!(near_hi > 20.0, "Near hi should give large positive value");
    }

    #[test]
    fn test_softmax_sums_one() {
        let x = vec![1.0, 2.0, 3.0, -1.0, 0.5];
        let p = softmax_transform(&x);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "Softmax sum = {} ≠ 1", sum);
        for &pi in &p {
            assert!(pi >= 0.0 && pi <= 1.0, "Probability {} out of [0,1]", pi);
        }
    }

    #[test]
    fn test_softmax_empty() {
        let p = softmax_transform(&[]);
        assert!(p.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let p = softmax_transform(&[3.7]);
        assert!((p[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_log_jacobian_positive() {
        // log|dη/dθ| = -log(θ) at various θ > 0
        for theta in [0.1, 1.0, 5.0] {
            let jac = log_jacobian_positive(theta);
            assert!((jac - (-theta.ln())).abs() < EPS);
        }
    }

    #[test]
    fn test_log_jacobian_bounded() {
        let lo = 0.0;
        let hi = 1.0;
        let theta = 0.3;
        let jac = log_jacobian_bounded(theta, lo, hi);
        // Expected: -ln(θ - lo) - ln(hi - θ) + ln(hi - lo)
        let expected = -(theta - lo).ln() - (hi - theta).ln() + (hi - lo).ln();
        assert!((jac - expected).abs() < EPS);
    }

    #[test]
    fn test_transform_spec_unconstrained_roundtrip() {
        let spec = TransformSpec::unconstrained();
        for val in [-3.0, 0.0, 7.0] {
            let eta = spec.to_unconstrained(val).expect("unconstrained ok");
            let theta = spec.to_constrained(eta);
            assert!((theta - val).abs() < EPS);
        }
    }

    #[test]
    fn test_transform_spec_positive_roundtrip() {
        let spec = TransformSpec::positive();
        for val in [0.01, 1.0, 100.0] {
            let eta = spec.to_unconstrained(val).expect("positive ok");
            let theta = spec.to_constrained(eta);
            assert!(
                (theta - val).abs() < EPS * val,
                "Roundtrip failed: {val} -> {eta} -> {theta}"
            );
        }
    }

    #[test]
    fn test_transform_spec_positive_error() {
        let spec = TransformSpec::positive();
        assert!(spec.to_unconstrained(0.0).is_err());
        assert!(spec.to_unconstrained(-1.0).is_err());
    }

    #[test]
    fn test_transform_spec_bounded_roundtrip() {
        let spec = TransformSpec::bounded(2.0, 8.0);
        for val in [2.5, 5.0, 7.9] {
            let eta = spec.to_unconstrained(val).expect("bounded ok");
            let theta = spec.to_constrained(eta);
            assert!(
                (theta - val).abs() < 1e-8,
                "Roundtrip failed: {val} -> {eta} -> {theta}"
            );
        }
    }

    #[test]
    fn test_transform_spec_bounded_error() {
        let spec = TransformSpec::bounded(0.0, 1.0);
        assert!(spec.to_unconstrained(0.0).is_err()); // boundary (excluded)
        assert!(spec.to_unconstrained(1.0).is_err()); // boundary (excluded)
        assert!(spec.to_unconstrained(-0.5).is_err()); // outside
    }

    #[test]
    fn test_log_jacobian_inverse_identity() {
        let spec = TransformSpec::unconstrained();
        assert!((spec.log_jacobian_inverse(3.14) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_log_jacobian_inverse_positive() {
        let spec = TransformSpec::positive();
        for eta in [-2.0, 0.0, 1.5] {
            let jac = spec.log_jacobian_inverse(eta);
            // log|dθ/dη| = η  (since θ = exp(η))
            assert!(
                (jac - eta).abs() < EPS,
                "log_jacobian_inverse({eta}) = {jac} ≠ {eta}"
            );
        }
    }

    #[test]
    fn test_log_jacobian_inverse_bounded() {
        let spec = TransformSpec::bounded(0.0, 1.0);
        let eta = 0.0; // sigmoid(0) = 0.5
        let jac = spec.log_jacobian_inverse(eta);
        // dθ/dη = (hi-lo) σ(1-σ) = 1 · 0.25
        let expected = (1.0_f64).ln() + 0.5_f64.ln() + 0.5_f64.ln();
        assert!((jac - expected).abs() < EPS);
    }
}
