//! Information-theoretic special functions
//!
//! This module provides functions from information theory that are commonly
//! used in machine learning, statistics, and communications, including:
//!
//! - Shannon entropy and variants (Rényi, Tsallis)
//! - Divergence measures (KL, Jensen-Shannon)
//! - Activation functions (sigmoid, softmax, softplus)
//! - Numerically stable log-sum-exp and log-softmax
//! - Gumbel-softmax (differentiable categorical approximation)
//!
//! All probability inputs are validated to be finite non-negative reals.
//! Conventions:
//! - `entropy` / `kl_divergence` etc. use **natural logarithms** (nats) by default;
//!   pass `base` = 2.0 for bits or 10.0 for dits.
//! - 0 · log 0 is defined as 0 by continuity.

use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::LN_2;

// ── Validation helpers ────────────────────────────────────────────────────────

/// Verify that all elements are finite and non-negative.
fn validate_probs(probs: &[f64], name: &str) -> SpecialResult<()> {
    for (i, &p) in probs.iter().enumerate() {
        if !p.is_finite() {
            return Err(SpecialError::ValueError(format!(
                "{name}[{i}] = {p} is not finite"
            )));
        }
        if p < 0.0 {
            return Err(SpecialError::ValueError(format!(
                "{name}[{i}] = {p} is negative"
            )));
        }
    }
    Ok(())
}

/// Verify that two slices have the same length.
fn same_length(a: &[f64], b: &[f64]) -> SpecialResult<()> {
    if a.len() != b.len() {
        Err(SpecialError::ValueError(format!(
            "slice lengths differ: {} vs {}",
            a.len(),
            b.len()
        )))
    } else {
        Ok(())
    }
}

// ── Entropy functions ─────────────────────────────────────────────────────────

/// Binary entropy H(p) in **bits**:  H(p) = −p·log₂(p) − (1−p)·log₂(1−p).
///
/// Returns 0 for p = 0 or p = 1, and the maximum 1.0 bit for p = 0.5.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::binary_entropy;
/// assert_relative_eq!(binary_entropy(0.5), 1.0, epsilon = 1e-12);
/// assert_eq!(binary_entropy(0.0), 0.0);
/// assert_eq!(binary_entropy(1.0), 0.0);
/// ```
pub fn binary_entropy(p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p == 0.0 || p == 1.0 {
        return 0.0;
    }
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

/// Binary entropy in **nats** (natural log base).
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::binary_entropy_nats;
/// use std::f64::consts::LN_2;
/// assert_relative_eq!(binary_entropy_nats(0.5), LN_2, epsilon = 1e-12);
/// ```
pub fn binary_entropy_nats(p: f64) -> f64 {
    binary_entropy(p) * LN_2
}

/// Shannon entropy of a discrete distribution.
///
/// H(p) = −Σ p_i · log_{base}(p_i)
///
/// Pass `base = std::f64::consts::E` for nats, `base = 2.0` for bits,
/// `base = 10.0` for dits.
///
/// # Errors
/// Returns an error if any probability is negative or non-finite.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::entropy;
/// // Uniform over 4 outcomes → log2(4) = 2 bits
/// let h = entropy(&[0.25, 0.25, 0.25, 0.25], 2.0).unwrap();
/// assert_relative_eq!(h, 2.0, epsilon = 1e-12);
/// ```
pub fn entropy(probs: &[f64], base: f64) -> SpecialResult<f64> {
    if base <= 0.0 || base == 1.0 {
        return Err(SpecialError::ValueError(
            "entropy: base must be > 0 and ≠ 1".to_string(),
        ));
    }
    validate_probs(probs, "probs")?;
    let ln_base = base.ln();
    let h = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln() / ln_base)
        .sum();
    Ok(h)
}

/// KL divergence D_KL(p ‖ q) = Σ p_i · ln(p_i / q_i)  (nats).
///
/// Convention:
/// - 0 · ln(0/q) = 0
/// - p_i > 0 and q_i = 0  → +∞ (returns infinity)
///
/// # Errors
/// Returns an error if the slices differ in length or contain invalid values.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::kl_divergence;
/// // Identical distributions → 0
/// let d = kl_divergence(&[0.5, 0.5], &[0.5, 0.5]).unwrap();
/// assert_relative_eq!(d, 0.0, epsilon = 1e-12);
/// // Degenerate p=[1,0] against q=[1,0] → 0
/// let d2 = kl_divergence(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
/// assert_relative_eq!(d2, 0.0, epsilon = 1e-12);
/// ```
pub fn kl_divergence(p: &[f64], q: &[f64]) -> SpecialResult<f64> {
    same_length(p, q)?;
    validate_probs(p, "p")?;
    validate_probs(q, "q")?;
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi == 0.0 {
            continue;
        }
        if qi == 0.0 {
            return Ok(f64::INFINITY);
        }
        kl += pi * (pi / qi).ln();
    }
    Ok(kl)
}

/// Jensen-Shannon divergence (symmetric, bounded [0, ln 2] in nats).
///
/// JSD(p ‖ q) = ½ · D_KL(p ‖ m) + ½ · D_KL(q ‖ m)   where m = (p+q)/2.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::js_divergence;
/// // Symmetry: JSD(p,q) == JSD(q,p)
/// let p = vec![0.7, 0.3];
/// let q = vec![0.4, 0.6];
/// let pq = js_divergence(&p, &q).unwrap();
/// let qp = js_divergence(&q, &p).unwrap();
/// assert_relative_eq!(pq, qp, epsilon = 1e-12);
/// assert!(pq >= 0.0 && pq <= 1.0);  // [0, ln2] ≤ 1.0
/// ```
pub fn js_divergence(p: &[f64], q: &[f64]) -> SpecialResult<f64> {
    same_length(p, q)?;
    validate_probs(p, "p")?;
    validate_probs(q, "q")?;
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| 0.5 * (pi + qi)).collect();
    let kl_pm = kl_divergence(p, &m)?;
    let kl_qm = kl_divergence(q, &m)?;
    Ok(0.5 * (kl_pm + kl_qm))
}

/// Rényi entropy of order α.
///
/// H_α(p) = 1/(1−α) · log(Σ p_i^α)
///
/// Special cases:
/// - α → 1 : Shannon entropy (computed via L'Hôpital)
/// - α = 0 : Hartley/max-entropy = log(|support|)
/// - α → ∞ : min-entropy = −log(max p_i)
///
/// Uses natural logarithm.
///
/// # Errors
/// Returns an error if α is negative, probabilities are invalid, or all
/// probabilities are zero.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::renyi_entropy;
/// use std::f64::consts::LN_2;
/// // For uniform over 4: Renyi = ln(4) regardless of α
/// let h = renyi_entropy(&[0.25, 0.25, 0.25, 0.25], 2.0).unwrap();
/// assert_relative_eq!(h, (4.0f64).ln(), epsilon = 1e-12);
/// ```
pub fn renyi_entropy(probs: &[f64], alpha: f64) -> SpecialResult<f64> {
    if alpha < 0.0 {
        return Err(SpecialError::DomainError(
            "renyi_entropy: alpha must be ≥ 0".to_string(),
        ));
    }
    validate_probs(probs, "probs")?;

    if (alpha - 1.0).abs() < 1e-12 {
        // L'Hôpital limit → Shannon entropy in nats
        return Ok(probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum());
    }
    if alpha == 0.0 {
        // Hartley entropy = log(support size)
        let support = probs.iter().filter(|&&p| p > 0.0).count();
        return Ok((support as f64).ln());
    }
    if alpha == f64::INFINITY {
        let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
        if max_p == 0.0 {
            return Err(SpecialError::DomainError(
                "renyi_entropy: all probabilities are zero".to_string(),
            ));
        }
        return Ok(-max_p.ln());
    }

    let sum_pow: f64 = probs.iter().map(|&p| p.powf(alpha)).sum();
    if sum_pow == 0.0 {
        return Err(SpecialError::DomainError(
            "renyi_entropy: Σ p_i^α = 0".to_string(),
        ));
    }
    Ok(sum_pow.ln() / (1.0 - alpha))
}

/// Tsallis entropy (non-extensive entropy of order q).
///
/// S_q(p) = (1 − Σ p_i^q) / (q − 1)  for q ≠ 1
/// S_1(p) = Shannon entropy (L'Hôpital limit)
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::tsallis_entropy;
/// let h = tsallis_entropy(&[0.5, 0.5], 2.0).unwrap();
/// assert_relative_eq!(h, 0.5, epsilon = 1e-12);
/// ```
pub fn tsallis_entropy(probs: &[f64], q: f64) -> SpecialResult<f64> {
    if q < 0.0 {
        return Err(SpecialError::DomainError(
            "tsallis_entropy: q must be ≥ 0".to_string(),
        ));
    }
    validate_probs(probs, "probs")?;

    if (q - 1.0).abs() < 1e-12 {
        // Shannon entropy
        return Ok(probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum());
    }

    let sum_pow: f64 = probs.iter().map(|&p| p.powf(q)).sum();
    Ok((1.0 - sum_pow) / (q - 1.0))
}

// ── Activation functions ──────────────────────────────────────────────────────

/// Logistic sigmoid σ(x) = 1 / (1 + e^{−x}).
///
/// Numerically stable for large |x|.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::sigmoid;
/// assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-14);
/// assert!(sigmoid(100.0) > 0.9999);
/// assert!(sigmoid(-100.0) < 1e-10);
/// ```
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        // Avoid overflow for large negative x
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Derivative of the sigmoid: σ'(x) = σ(x) · (1 − σ(x)).
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::sigmoid_derivative;
/// assert_relative_eq!(sigmoid_derivative(0.0), 0.25, epsilon = 1e-14);
/// ```
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Numerically stable log-sum-exp:  log(Σ exp(x_i)).
///
/// Computes max(x) + log(Σ exp(x_i − max(x))).
///
/// Returns −∞ for an empty slice.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::log_sum_exp;
/// // log(e^1 + e^2) ≈ 2.3132617…
/// let v = log_sum_exp(&[1.0, 2.0]);
/// assert_relative_eq!(v, (1.0f64.exp() + 2.0f64.exp()).ln(), epsilon = 1e-12);
/// ```
pub fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Softplus: log(1 + e^x).
///
/// Numerically stable for large |x|.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::softplus;
/// // For large x, softplus(x) ≈ x
/// assert!(softplus(100.0) > 99.9);
/// // For large negative x, softplus(x) ≈ 0
/// assert!(softplus(-100.0) < 1e-10);
/// ```
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x // log(1 + e^x) ≈ x for large x
    } else if x < -20.0 {
        x.exp() // log(1 + e^x) ≈ e^x for large negative x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Softmax distribution: exp(x_i) / Σ exp(x_j).
///
/// Numerically stable via the log-sum-exp trick.
///
/// Returns an empty Vec for empty input.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::softmax;
/// let s = softmax(&[1.0, 2.0, 3.0]);
/// assert!(s.iter().all(|&v| v > 0.0));
/// let sum: f64 = s.iter().sum();
/// assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
/// ```
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Log-softmax:  x_i − log(Σ exp(x_j)).
///
/// Numerically stable, equivalent to `log(softmax(x))` but avoids underflow.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::information_theoretic::{softmax, log_softmax};
/// let xs = &[1.0, 2.0, 3.0];
/// let ls = log_softmax(xs);
/// let s = softmax(xs);
/// for (l, sv) in ls.iter().zip(s.iter()) {
///     assert_relative_eq!(*l, sv.ln(), epsilon = 1e-12);
/// }
/// ```
pub fn log_softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let lse = log_sum_exp(xs);
    xs.iter().map(|&x| x - lse).collect()
}

/// Gumbel-softmax (concrete distribution / straight-through estimator).
///
/// Differentiable approximation to discrete sampling using the Gumbel-max trick:
///
/// y_i = softmax((log(π_i) + g_i) / τ)
///
/// where g_i ~ Gumbel(0,1) = −log(−log(U_i)) with U_i ~ Uniform(0,1),
/// π_i are the (unnormalized) logits, and τ > 0 is the temperature.
///
/// As τ → 0 the samples become one-hot; as τ → ∞ they approach uniform.
///
/// The `seed` parameter initializes a simple linear-congruential generator
/// so the function is deterministic given a seed.
///
/// # Errors
/// Returns an error if logits is empty or temperature ≤ 0.
///
/// # Examples
/// ```
/// use scirs2_special::information_theoretic::gumbel_softmax;
/// let gs = gumbel_softmax(&[1.0, 2.0, 3.0], 1.0, 42).unwrap();
/// assert_eq!(gs.len(), 3);
/// let sum: f64 = gs.iter().sum();
/// // Sum ≈ 1.0 (softmax output)
/// assert!((sum - 1.0).abs() < 1e-12);
/// ```
pub fn gumbel_softmax(logits: &[f64], temperature: f64, seed: u64) -> SpecialResult<Vec<f64>> {
    if logits.is_empty() {
        return Err(SpecialError::ValueError(
            "gumbel_softmax: logits must not be empty".to_string(),
        ));
    }
    if temperature <= 0.0 {
        return Err(SpecialError::DomainError(
            "gumbel_softmax: temperature must be > 0".to_string(),
        ));
    }

    // Simple LCG for reproducible Gumbel noise
    let gumbel_samples: Vec<f64> = {
        let mut state = seed.wrapping_add(0x853c49e6748fea9b);
        (0..logits.len())
            .map(|_| {
                // xorshift64*
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to (0,1) open interval
                let u = (state as f64) / (u64::MAX as f64);
                let u_clamped = u.clamp(1e-38, 1.0 - 1e-15);
                // Gumbel(0,1) = -log(-log(U))
                -(-u_clamped.ln()).ln()
            })
            .collect()
    };

    let perturbed: Vec<f64> = logits
        .iter()
        .zip(gumbel_samples.iter())
        .map(|(&l, &g)| (l + g) / temperature)
        .collect();

    Ok(softmax(&perturbed))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binary_entropy_bits() {
        assert_relative_eq!(binary_entropy(0.5), 1.0, epsilon = 1e-12);
        assert_eq!(binary_entropy(0.0), 0.0);
        assert_eq!(binary_entropy(1.0), 0.0);
        assert!(binary_entropy(0.3) > 0.0);
        assert!(binary_entropy(0.3) < 1.0);
    }

    #[test]
    fn test_binary_entropy_nats() {
        use std::f64::consts::LN_2;
        assert_relative_eq!(binary_entropy_nats(0.5), LN_2, epsilon = 1e-12);
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform over n outcomes → log_b(n)
        let h_bits = entropy(&[0.25, 0.25, 0.25, 0.25], 2.0).expect("ok");
        assert_relative_eq!(h_bits, 2.0, epsilon = 1e-12);

        let h_nats = entropy(&[0.25, 0.25, 0.25, 0.25], std::f64::consts::E).expect("ok");
        assert_relative_eq!(h_nats, (4.0f64).ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_entropy_certain() {
        let h = entropy(&[1.0, 0.0, 0.0], 2.0).expect("ok");
        assert_relative_eq!(h, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_entropy_invalid_base() {
        assert!(entropy(&[0.5, 0.5], 1.0).is_err());
        assert!(entropy(&[0.5, 0.5], -2.0).is_err());
    }

    #[test]
    fn test_kl_divergence_zero() {
        let d = kl_divergence(&[0.5, 0.5], &[0.5, 0.5]).expect("ok");
        assert_relative_eq!(d, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_kl_divergence_degenerate() {
        let d = kl_divergence(&[1.0, 0.0], &[1.0, 0.0]).expect("ok");
        assert_relative_eq!(d, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_kl_divergence_nonneg() {
        let p = [0.7, 0.3];
        let q = [0.4, 0.6];
        let kl = kl_divergence(&p, &q).expect("ok");
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_kl_divergence_infinity() {
        // p has support where q = 0
        let kl = kl_divergence(&[0.5, 0.5], &[1.0, 0.0]).expect("ok");
        assert!(kl.is_infinite());
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = [0.7, 0.3];
        let q = [0.4, 0.6];
        let pq = js_divergence(&p, &q).expect("ok");
        let qp = js_divergence(&q, &p).expect("ok");
        assert_relative_eq!(pq, qp, epsilon = 1e-12);
        assert!(pq >= 0.0);
        assert!(pq <= 1.0); // bounded by ln(2) ≈ 0.693
    }

    #[test]
    fn test_js_divergence_identical() {
        let p = [0.3, 0.4, 0.3];
        let jsd = js_divergence(&p, &p).expect("ok");
        assert_relative_eq!(jsd, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_renyi_entropy_uniform() {
        // For uniform distribution, Renyi entropy = ln(n) for all α
        let probs = [0.25, 0.25, 0.25, 0.25];
        for &alpha in &[0.5, 2.0, 3.0] {
            let h = renyi_entropy(&probs, alpha).expect("ok");
            assert_relative_eq!(h, (4.0f64).ln(), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_renyi_limit_shannon() {
        // α → 1 should give Shannon entropy
        let probs = [0.5, 0.3, 0.2];
        let h_shannon = entropy(&probs, std::f64::consts::E).expect("ok");
        let h_renyi = renyi_entropy(&probs, 1.0).expect("ok");
        assert_relative_eq!(h_shannon, h_renyi, epsilon = 1e-12);
    }

    #[test]
    fn test_tsallis_entropy() {
        // S_2([0.5,0.5]) = (1 - 2·0.25) / (2-1) = 0.5
        let h = tsallis_entropy(&[0.5, 0.5], 2.0).expect("ok");
        assert_relative_eq!(h, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_tsallis_limit_shannon() {
        let probs = [0.5, 0.3, 0.2];
        let h_shannon = entropy(&probs, std::f64::consts::E).expect("ok");
        let h_tsallis = tsallis_entropy(&probs, 1.0).expect("ok");
        assert_relative_eq!(h_shannon, h_tsallis, epsilon = 1e-12);
    }

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-14);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
        // Symmetry: σ(-x) = 1 - σ(x)
        assert_relative_eq!(sigmoid(-2.0), 1.0 - sigmoid(2.0), epsilon = 1e-14);
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert_relative_eq!(sigmoid_derivative(0.0), 0.25, epsilon = 1e-14);
        let s = sigmoid(1.5);
        assert_relative_eq!(sigmoid_derivative(1.5), s * (1.0 - s), epsilon = 1e-14);
    }

    #[test]
    fn test_log_sum_exp() {
        let v = log_sum_exp(&[1.0, 2.0]);
        let expected = (1.0f64.exp() + 2.0f64.exp()).ln();
        assert_relative_eq!(v, expected, epsilon = 1e-12);
        // Identity: log_sum_exp([x]) = x
        assert_relative_eq!(log_sum_exp(&[5.0]), 5.0, epsilon = 1e-14);
        // Empty
        assert!(log_sum_exp(&[]).is_infinite() && log_sum_exp(&[]).is_sign_negative());
    }

    #[test]
    fn test_softplus() {
        // softplus(0) = ln(2)
        assert_relative_eq!(softplus(0.0), LN_2, epsilon = 1e-12);
        // Large x: softplus(x) ≈ x
        assert_relative_eq!(softplus(50.0), 50.0, epsilon = 0.001);
        // Large negative: softplus(-50) ≈ 0
        assert!(softplus(-50.0) < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = s.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        // All positive
        assert!(s.iter().all(|&v| v > 0.0));
        // Larger input → larger output
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let s = softmax(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn test_log_softmax() {
        let xs = [1.0, 2.0, 3.0];
        let ls = log_softmax(&xs);
        let s = softmax(&xs);
        for (l, sv) in ls.iter().zip(s.iter()) {
            assert_relative_eq!(*l, sv.ln(), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_gumbel_softmax_valid() {
        let gs = gumbel_softmax(&[1.0, 2.0, 3.0], 1.0, 42).expect("ok");
        assert_eq!(gs.len(), 3);
        let sum: f64 = gs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        assert!(gs.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_gumbel_softmax_low_temp() {
        // Low temperature → near one-hot
        let gs = gumbel_softmax(&[0.0, 10.0, 0.0], 0.01, 7).expect("ok");
        assert!(gs[1] > 0.99);
    }

    #[test]
    fn test_gumbel_softmax_errors() {
        assert!(gumbel_softmax(&[], 1.0, 0).is_err());
        assert!(gumbel_softmax(&[1.0], 0.0, 0).is_err());
        assert!(gumbel_softmax(&[1.0], -1.0, 0).is_err());
    }
}
