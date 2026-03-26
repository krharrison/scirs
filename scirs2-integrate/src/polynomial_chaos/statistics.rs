//! Statistical analysis from PCE coefficients.
//!
//! Extracts mean, variance, Sobol sensitivity indices, and other statistical
//! quantities directly from the PCE coefficient vector without additional
//! sampling or integration.

use crate::error::{IntegrateError, IntegrateResult};

use super::expansion::PolynomialChaosExpansion;

/// Mean of the PCE: E\[Y\] = c_0 (the zeroth coefficient).
pub fn pce_mean(coefficients: &[f64]) -> f64 {
    if coefficients.is_empty() {
        return 0.0;
    }
    coefficients[0]
}

/// Variance of the PCE: Var\[Y\] = sum_{k>=1} c_k^2 * ||Psi_k||^2.
pub fn pce_variance(coefficients: &[f64], norms_squared: &[f64]) -> f64 {
    if coefficients.len() <= 1 {
        return 0.0;
    }
    coefficients
        .iter()
        .zip(norms_squared.iter())
        .skip(1)
        .map(|(&c, &n)| c * c * n)
        .sum()
}

/// Standard deviation of the PCE.
pub fn pce_std(coefficients: &[f64], norms_squared: &[f64]) -> f64 {
    pce_variance(coefficients, norms_squared).sqrt()
}

/// First-order Sobol sensitivity indices.
///
/// S_i = (sum of c_k^2 * ||Psi_k||^2 for k where only variable i is active) / Var\[Y\]
///
/// A multi-index alpha "depends only on variable i" when alpha\[j\] = 0 for all j != i
/// and alpha\[i\] > 0.
pub fn sobol_indices(
    coefficients: &[f64],
    multi_indices: &[Vec<usize>],
    norms_squared: &[f64],
) -> IntegrateResult<Vec<f64>> {
    if coefficients.len() != multi_indices.len() || coefficients.len() != norms_squared.len() {
        return Err(IntegrateError::DimensionMismatch(
            "coefficients, multi_indices, and norms_squared must have the same length".to_string(),
        ));
    }
    if multi_indices.is_empty() {
        return Err(IntegrateError::ValueError(
            "Empty multi-index set".to_string(),
        ));
    }

    let dim = multi_indices[0].len();
    let total_var = pce_variance(coefficients, norms_squared);

    if total_var.abs() < 1e-30 {
        // Zero variance: return equal indices
        return Ok(vec![1.0 / dim as f64; dim]);
    }

    let mut indices = vec![0.0_f64; dim];

    for (k, alpha) in multi_indices.iter().enumerate().skip(1) {
        let contribution = coefficients[k] * coefficients[k] * norms_squared[k];

        // Check if only one variable is active
        let active_vars: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 0)
            .map(|(i, _)| i)
            .collect();

        if active_vars.len() == 1 {
            indices[active_vars[0]] += contribution;
        }
    }

    // Normalize by total variance
    for s in &mut indices {
        *s /= total_var;
    }

    Ok(indices)
}

/// Total Sobol sensitivity indices.
///
/// ST_i = (sum of c_k^2 * ||Psi_k||^2 for k where alpha\[i\] > 0) / Var\[Y\]
///
/// Measures total effect of variable i including all interactions.
pub fn total_sobol_indices(
    coefficients: &[f64],
    multi_indices: &[Vec<usize>],
    norms_squared: &[f64],
) -> IntegrateResult<Vec<f64>> {
    if coefficients.len() != multi_indices.len() || coefficients.len() != norms_squared.len() {
        return Err(IntegrateError::DimensionMismatch(
            "coefficients, multi_indices, and norms_squared must have the same length".to_string(),
        ));
    }
    if multi_indices.is_empty() {
        return Err(IntegrateError::ValueError(
            "Empty multi-index set".to_string(),
        ));
    }

    let dim = multi_indices[0].len();
    let total_var = pce_variance(coefficients, norms_squared);

    if total_var.abs() < 1e-30 {
        return Ok(vec![1.0 / dim as f64; dim]);
    }

    let mut indices = vec![0.0_f64; dim];

    for (k, alpha) in multi_indices.iter().enumerate().skip(1) {
        let contribution = coefficients[k] * coefficients[k] * norms_squared[k];

        for (i, &a) in alpha.iter().enumerate() {
            if a > 0 {
                indices[i] += contribution;
            }
        }
    }

    // Normalize by total variance
    for s in &mut indices {
        *s /= total_var;
    }

    Ok(indices)
}

/// Skewness from PCE coefficients.
///
/// Computes E\[(Y - mu)^3\] / sigma^3 using PCE coefficient structure.
/// This requires evaluating triple products of basis functions.
///
/// For simplicity, this uses sampling-based estimation from the PCE surrogate.
pub fn pce_skewness(
    pce: &PolynomialChaosExpansion,
    n_samples: usize,
    seed: u64,
) -> IntegrateResult<f64> {
    let coeffs = pce
        .coefficients
        .as_ref()
        .ok_or_else(|| IntegrateError::ComputationError("PCE not fitted yet".to_string()))?;

    let mean = pce_mean(coeffs);
    let variance = pce_variance(coeffs, &pce.basis_norms_squared);
    let std_dev = variance.sqrt();

    if std_dev < 1e-15 {
        return Ok(0.0);
    }

    // Sample from the PCE and estimate skewness
    let samples = generate_pce_samples(pce, n_samples, seed)?;
    let m3: f64 = samples
        .iter()
        .map(|&s| ((s - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n_samples as f64;

    Ok(m3)
}

/// Confidence interval from PCE via sampling.
///
/// Generates `n_samples` realizations from the PCE surrogate and returns
/// the `(lower, upper)` bounds at the given confidence level.
pub fn pce_confidence_interval(
    pce: &PolynomialChaosExpansion,
    confidence: f64,
    n_samples: usize,
    seed: u64,
) -> IntegrateResult<(f64, f64)> {
    if !(0.0..1.0).contains(&confidence) {
        return Err(IntegrateError::ValueError(format!(
            "Confidence level must be in (0, 1), got {confidence}"
        )));
    }
    if n_samples < 10 {
        return Err(IntegrateError::ValueError(
            "Need at least 10 samples for confidence interval".to_string(),
        ));
    }

    let mut samples = generate_pce_samples(pce, n_samples, seed)?;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * n_samples as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha) * n_samples as f64).ceil() as usize;
    let upper_idx = upper_idx.min(n_samples - 1);

    Ok((samples[lower_idx], samples[upper_idx]))
}

/// Generate samples from the PCE surrogate by sampling the input random space.
fn generate_pce_samples(
    pce: &PolynomialChaosExpansion,
    n_samples: usize,
    seed: u64,
) -> IntegrateResult<Vec<f64>> {
    let dim = pce.dim();
    let mut rng_state = seed;
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut xi = Vec::with_capacity(dim);
        for basis in &pce.config.bases {
            let u = lcg_uniform(&mut rng_state);
            let value = match basis {
                super::types::PolynomialBasis::Hermite => {
                    let u2 = lcg_uniform(&mut rng_state);
                    let u1 = u.max(1e-15).min(1.0 - 1e-15);
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                }
                super::types::PolynomialBasis::Legendre => 2.0 * u - 1.0,
                super::types::PolynomialBasis::Laguerre => {
                    let uc = u.max(1e-15).min(1.0 - 1e-15);
                    -(1.0 - uc).ln()
                }
                super::types::PolynomialBasis::Jacobi { .. } => 2.0 * u - 1.0,
            };
            xi.push(value);
        }
        samples.push(pce.evaluate(&xi)?);
    }

    Ok(samples)
}

/// Linear congruential generator returning a value in \[0, 1).
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial_chaos::types::*;

    #[test]
    fn test_mean_is_c0() {
        let coeffs = vec![3.25, 1.0, 2.0, 0.5];
        assert!((pce_mean(&coeffs) - 3.25).abs() < 1e-14);
    }

    #[test]
    fn test_variance_computation() {
        // Legendre norms: 1/(2n+1) for degree n
        // Multi-indices for 1D, degree 2: [0], [1], [2]
        let coeffs = vec![1.0, 2.0, 3.0];
        let norms_sq = vec![1.0, 1.0 / 3.0, 1.0 / 5.0];
        // Var = 2^2 * 1/3 + 3^2 * 1/5 = 4/3 + 9/5 = 20/15 + 27/15 = 47/15
        let var = pce_variance(&coeffs, &norms_sq);
        assert!(
            (var - 47.0 / 15.0).abs() < 1e-14,
            "Got {var}, expected {}",
            47.0 / 15.0
        );
    }

    #[test]
    fn test_sobol_indices_additive() {
        // f(x1, x2) = x1 + 2*x2 with Legendre basis on [-1,1]^2
        // PCE: c_{(0,0)}=0, c_{(1,0)}=1 (x1 component), c_{(0,1)}=2 (x2 component)
        // Var_1 = 1^2 * ||P_1||^2 = 1/3, Var_2 = 2^2 * ||P_1||^2 = 4/3
        // Total Var = 5/3
        // S_1 = (1/3) / (5/3) = 1/5, S_2 = (4/3) / (5/3) = 4/5
        let coeffs = vec![0.0, 1.0, 2.0];
        let multi_indices = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
        let norms_sq = vec![1.0, 1.0 / 3.0, 1.0 / 3.0];

        let sobol = sobol_indices(&coeffs, &multi_indices, &norms_sq).expect("sobol failed");
        assert!(
            (sobol[0] - 1.0 / 5.0).abs() < 1e-14,
            "S_1: got {}, expected {}",
            sobol[0],
            1.0 / 5.0
        );
        assert!(
            (sobol[1] - 4.0 / 5.0).abs() < 1e-14,
            "S_2: got {}, expected {}",
            sobol[1],
            4.0 / 5.0
        );
    }

    #[test]
    fn test_total_sobol_additive() {
        // For additive functions, total Sobol = first-order Sobol (no interactions)
        let coeffs = vec![0.0, 1.0, 2.0];
        let multi_indices = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
        let norms_sq = vec![1.0, 1.0 / 3.0, 1.0 / 3.0];

        let sobol = sobol_indices(&coeffs, &multi_indices, &norms_sq).expect("sobol failed");
        let total =
            total_sobol_indices(&coeffs, &multi_indices, &norms_sq).expect("total sobol failed");

        for i in 0..2 {
            assert!(
                (sobol[i] - total[i]).abs() < 1e-14,
                "Dim {i}: S={}, ST={}",
                sobol[i],
                total[i]
            );
        }
    }

    #[test]
    fn test_sobol_with_interactions() {
        // f(x1,x2) = x1 + x1*x2
        // c_{(0,0)}=0, c_{(1,0)}=1, c_{(1,1)}=1 (interaction)
        // Legendre norms: ||P_n||^2 = 1/(2n+1)
        // Var_1 (only x1) = 1^2 * 1/3 = 1/3
        // Var_{12} (interaction) = 1^2 * (1/3)(1/3) = 1/9
        // Total Var = 1/3 + 1/9 = 4/9
        // S_1 = (1/3) / (4/9) = 3/4, S_2 = 0 (x2 alone never appears)
        // ST_1 = (1/3 + 1/9) / (4/9) = (4/9)/(4/9) = 1
        // ST_2 = (1/9) / (4/9) = 1/4
        let coeffs = vec![0.0, 1.0, 1.0];
        let multi_indices = vec![vec![0, 0], vec![1, 0], vec![1, 1]];
        let norms_sq = vec![1.0, 1.0 / 3.0, 1.0 / 9.0]; // product of 1-D norms

        let sobol = sobol_indices(&coeffs, &multi_indices, &norms_sq).expect("sobol failed");
        let total =
            total_sobol_indices(&coeffs, &multi_indices, &norms_sq).expect("total sobol failed");

        assert!(
            (sobol[0] - 0.75).abs() < 1e-14,
            "S_1: got {}, expected 0.75",
            sobol[0]
        );
        assert!(sobol[1].abs() < 1e-14, "S_2: got {}, expected 0", sobol[1]);
        assert!(
            (total[0] - 1.0).abs() < 1e-14,
            "ST_1: got {}, expected 1.0",
            total[0]
        );
        assert!(
            (total[1] - 0.25).abs() < 1e-14,
            "ST_2: got {}, expected 0.25",
            total[1]
        );
    }

    #[test]
    fn test_confidence_interval() {
        // Fit a simple PCE and check confidence interval
        let config = PCEConfig {
            bases: vec![PolynomialBasis::Legendre],
            max_degree: 2,
            truncation: TruncationScheme::TotalDegree,
            coefficient_method: CoefficientMethod::Projection {
                quadrature_order: 5,
            },
        };
        let mut pce = PolynomialChaosExpansion::new(config).expect("PCE creation failed");
        let _result = pce.fit(|xi| Ok(xi[0] * xi[0])).expect("fit failed");

        let (lower, upper) = pce_confidence_interval(&pce, 0.95, 10000, 123).expect("CI failed");
        // For xi^2 on [-1,1], range is [0, 1]
        assert!(lower >= -0.1, "Lower bound too low: {lower}");
        assert!(upper <= 1.1, "Upper bound too high: {upper}");
        assert!(lower < upper, "Lower >= upper: {lower} >= {upper}");
    }
}
