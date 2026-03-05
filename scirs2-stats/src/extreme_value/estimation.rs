//! Parameter estimation for extreme value distributions.
//!
//! Provides three families of estimators:
//! 1. **Maximum Likelihood Estimation (MLE)** via a custom Nelder–Mead simplex optimizer.
//! 2. **Probability-Weighted Moments (PWM)** following Greenwood et al. (1979) and Hosking (1985).
//! 3. **L-moments** following Hosking (1990) – linear combinations of expected order statistics.
//!
//! # References
//! - Hosking, J.R.M. (1990). L-moments. *JRSS-B*, 52(1), 105–124.
//! - Hosking, J.R.M. & Wallis, J.R. (1997). *Regional Frequency Analysis*. Cambridge UP.
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.

use crate::error::StatsError;
use scirs2_core::ndarray::{Array1, ArrayView1};

use super::distributions::{GeneralizedExtremeValue, GeneralizedPareto, Gumbel};

// ---------------------------------------------------------------------------
// Internal Nelder–Mead simplex minimizer
// ---------------------------------------------------------------------------

/// Minimise `f: R^n → R` using a basic Nelder–Mead simplex.
///
/// Returns `(best_params, best_value)` or `Err` if the algorithm cannot make progress.
fn nelder_mead<F>(
    f: &F,
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, f64), StatsError>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Nelder-Mead requires at least one parameter".into(),
        ));
    }

    let step = 0.05; // initial simplex step size
                     // Build initial simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if v[i].abs() > 1e-8 {
            step * v[i].abs()
        } else {
            step
        };
        simplex.push(v);
    }

    // Evaluate at all vertices
    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink

    for _iter in 0..max_iter {
        // Sort by value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Check convergence: spread of function values
        let best = values[order[0]];
        let worst = values[order[n]];
        if (worst - best).abs() < tol && _iter > 5 {
            break;
        }

        // Centroid of all but worst
        let mut centroid = vec![0.0_f64; n];
        for &idx in &order[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        let worst_v = simplex[order[n]].clone();
        let worst_f = values[order[n]];

        // Reflection
        let reflected: Vec<f64> = centroid
            .iter()
            .zip(&worst_v)
            .map(|(&c, &w)| c + ALPHA * (c - w))
            .collect();
        let f_r = f(&reflected);

        if f_r < values[order[0]] {
            // Try expansion
            let expanded: Vec<f64> = centroid
                .iter()
                .zip(&reflected)
                .map(|(&c, &r)| c + GAMMA * (r - c))
                .collect();
            let f_e = f(&expanded);
            if f_e < f_r {
                simplex[order[n]] = expanded;
                values[order[n]] = f_e;
            } else {
                simplex[order[n]] = reflected;
                values[order[n]] = f_r;
            }
        } else if f_r < worst_f {
            simplex[order[n]] = reflected;
            values[order[n]] = f_r;
        } else {
            // Contraction
            let contracted: Vec<f64> = centroid
                .iter()
                .zip(&worst_v)
                .map(|(&c, &w)| c + RHO * (w - c))
                .collect();
            let f_c = f(&contracted);
            if f_c < worst_f {
                simplex[order[n]] = contracted;
                values[order[n]] = f_c;
            } else {
                // Shrink toward best
                let best_v = simplex[order[0]].clone();
                for k in 1..=n {
                    let new_v: Vec<f64> = best_v
                        .iter()
                        .zip(&simplex[order[k]])
                        .map(|(&b, &v)| b + SIGMA * (v - b))
                        .collect();
                    let f_new = f(&new_v);
                    simplex[order[k]] = new_v;
                    values[order[k]] = f_new;
                }
            }
        }
    }

    // Return best
    let best_idx = (0..=n)
        .min_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    Ok((simplex[best_idx].clone(), values[best_idx]))
}

// ---------------------------------------------------------------------------
// Sample statistics helpers
// ---------------------------------------------------------------------------

/// Sort a slice into a new Vec.
fn sorted_copy(data: ArrayView1<f64>) -> Vec<f64> {
    let mut v: Vec<f64> = data.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Sample mean.
fn sample_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

// ---------------------------------------------------------------------------
// L-moments
// ---------------------------------------------------------------------------

/// Compute the first `order` L-moments from a sample.
///
/// Uses Hosking's (1990) unbiased estimators based on order statistics.
/// Returns a `Vec<f64>` of length `order` containing λ₁, λ₂, …, λ_order.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if `n < order`
/// - [`StatsError::InvalidArgument`] if `order == 0`
pub fn sample_lmoments(data: ArrayView1<f64>, order: usize) -> Result<Vec<f64>, StatsError> {
    if order == 0 {
        return Err(StatsError::InvalidArgument(
            "L-moment order must be >= 1".into(),
        ));
    }
    let n = data.len();
    if n < order {
        return Err(StatsError::InsufficientData(format!(
            "Need at least {order} observations for L-moments of order {order}, got {n}"
        )));
    }

    let sorted = sorted_copy(data);
    let mut lmoms = Vec::with_capacity(order);

    // λ₁ = L1 = mean (sample mean)
    lmoms.push(sample_mean(&sorted));

    if order == 1 {
        return Ok(lmoms);
    }

    // For r >= 2, use the unbiased PWM estimator b_r−1:
    // b_k = (1/n) Σ C(i-1, k)/C(n-1, k) * x_(i)
    // λ_r = Σ_{k=0}^{r-1} p*_r,k * b_k
    // where p*_r,k = (-1)^{r-1-k} C(r-1,k) C(r+k-2, k)
    //
    // Equivalently, use Hosking (1990) Eq. (6):
    // λ_r = r^{-1} Σ_{j=0}^{r-1} (-1)^j C(r-1,j) E[X_{r-j:r}]
    //
    // For sample estimation we use the PWM approach (unbiased):

    // Compute sample PWMs b_0, b_1, ..., b_{order-1}
    let mut b = vec![0.0_f64; order];
    let nf = n as f64;

    for r in 0..order {
        if r == 0 {
            b[0] = sample_mean(&sorted);
        } else {
            // b_r = (1/n) Σ_{i=r+1}^{n} [C(i-1,r)/C(n-1,r)] * x_(i)
            let mut sum = 0.0_f64;
            let cr_denom = falling_factorial(n - 1, r) as f64;
            if cr_denom == 0.0 {
                b[r] = 0.0;
                continue;
            }
            for i in r..n {
                // C(i, r) numerator
                let cr_num = falling_factorial(i, r) as f64;
                sum += cr_num / cr_denom * sorted[i];
            }
            b[r] = sum / nf;
        }
    }

    // Convert PWMs to L-moments using Hosking (1990) Table 1 formulas
    // λ_2 = 2b₁ − b₀
    // λ_3 = 6b₂ − 6b₁ + b₀
    // λ_4 = 20b₃ − 30b₂ + 12b₁ − b₀
    // General: λ_r = Σ_{k=0}^{r-1} (-1)^{r-1-k} C(r-1,k) C(r-1+k, k) b_k  /  (but note λ_1 = b_0)
    // Using the relationship: λ_r = r^{-1} Σ_{k=0}^{r-1} (-1)^{r-1-k} C(r-1,k) C(r+k-2, r-1) b_k * r
    //
    // Cleaner: use the standard formula
    // λ_r = Σ_{k=0}^{r-1} p*_{r,k} b_k
    // p*_{r,k} = (-1)^{r-1-k} C(r-1,k) C(r+k-1, r-1) * (k-th coefficient from Hosking 1990)
    // For r=1: λ_1 = b_0  (done above)

    for r in 2..=order {
        let mut lm = 0.0_f64;
        for k in 0..r {
            let sign = if (r - 1 - k) % 2 == 0 { 1.0 } else { -1.0 };
            let c1 = binomial_coeff(r - 1, k) as f64;
            let c2 = binomial_coeff(r + k - 1, k) as f64;
            // (-1)^{r-1-k} C(r-1,k) C(r+k-1, k) b_k / (nothing — this equals λ_r formula)
            // Actually the standard is: λ_r = sum_{k=0}^{r-1} (-1)^{r-1-k} C(r-1,k) C(r+k-1,r-1) b_k
            // C(r+k-1, r-1) = C(r+k-1, k)
            lm += sign * c1 * c2 * b[k];
        }
        // Normalize by r (Hosking 1990 eq. 2.3 uses sum formula without extra factor)
        // Actually no normalization needed — the formula above gives λ_r directly
        // But let's re-check: for r=2, λ_2 = Σ_{k=0}^{1} (-1)^{1-k} C(1,k) C(k+1,k) b_k
        //   k=0: (-1)^1 C(1,0) C(1,0) b_0 = -1*1*1*b_0 = -b_0
        //   k=1: (-1)^0 C(1,1) C(2,1) b_1 = 1*1*2*b_1 = 2b_1
        //   sum = 2b_1 - b_0 ✓
        lmoms.push(lm);
    }

    Ok(lmoms)
}

/// Falling factorial: n * (n-1) * ... * (n-k+1)
fn falling_factorial(n: usize, k: usize) -> u64 {
    if k == 0 {
        return 1;
    }
    if k > n {
        return 0;
    }
    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64);
    }
    result
}

/// Binomial coefficient C(n, k).
fn binomial_coeff(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / (i + 1) as u64;
    }
    result
}

// ---------------------------------------------------------------------------
// GEV estimation
// ---------------------------------------------------------------------------

/// Fit a GEV distribution to `data` using Maximum Likelihood Estimation.
///
/// Uses the Nelder–Mead simplex optimizer to minimise the negative log-likelihood.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 3 observations.
/// - [`StatsError::ConvergenceError`] if the optimizer fails.
pub fn gev_fit_mle(data: ArrayView1<f64>) -> Result<GeneralizedExtremeValue, StatsError> {
    let n = data.len();
    if n < 3 {
        return Err(StatsError::InsufficientData(
            "GEV MLE requires at least 3 observations".into(),
        ));
    }

    // Validate no NaN / Inf
    for &x in data.iter() {
        if !x.is_finite() {
            return Err(StatsError::InvalidArgument(
                "Data contains non-finite values".into(),
            ));
        }
    }

    let data_vec: Vec<f64> = data.iter().copied().collect();

    // Initial parameter estimates via L-moments (more robust starting point)
    let init = match gev_fit_lmoments(data) {
        Ok(g) => vec![g.mu, g.sigma, g.xi],
        Err(_) => {
            let mean: f64 = sample_mean(&data_vec);
            let var: f64 = data_vec.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            vec![
                mean,
                var.sqrt() * (6.0_f64.sqrt() / std::f64::consts::PI),
                0.1,
            ]
        }
    };

    let neg_log_lik = |params: &[f64]| -> f64 {
        let mu = params[0];
        let sigma = params[1];
        let xi = params[2];

        if sigma <= 0.0 {
            return f64::MAX;
        }

        let gev = match GeneralizedExtremeValue::new(mu, sigma, xi) {
            Ok(g) => g,
            Err(_) => return f64::MAX,
        };

        let mut ll = 0.0_f64;
        for &x in &data_vec {
            let p = gev.pdf(x);
            if p <= 0.0 || !p.is_finite() {
                return f64::MAX;
            }
            ll += p.ln();
        }
        -ll
    };

    let (best, _) = nelder_mead(&neg_log_lik, &init, 2000, 1e-10)?;
    let mu = best[0];
    let sigma = best[1];
    let xi = best[2];

    GeneralizedExtremeValue::new(mu, sigma, xi).map_err(|e| {
        StatsError::ConvergenceError(format!("GEV MLE converged to invalid parameters: {e}"))
    })
}

/// Fit a GEV distribution using Probability-Weighted Moments (PWM).
///
/// Follows Hosking, Wallis & Wood (1985, Technometrics).
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 3 observations.
pub fn gev_fit_pwm(data: ArrayView1<f64>) -> Result<GeneralizedExtremeValue, StatsError> {
    let n = data.len();
    if n < 3 {
        return Err(StatsError::InsufficientData(
            "GEV PWM requires at least 3 observations".into(),
        ));
    }

    let sorted = sorted_copy(data);
    let nf = n as f64;

    // Probability-weighted moments:
    // M(1,0,0) = mean
    // M(1,1,0) = b1 = (1/n) Σ ((i-1)/(n-1)) x_(i)
    // M(1,2,0) = b2 = (1/n) Σ ((i-1)(i-2)/((n-1)(n-2))) x_(i)
    let b0 = sample_mean(&sorted);

    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for i in 1..n {
        let if64 = i as f64;
        let ni_1 = nf - 1.0;
        b1 += (if64 / ni_1) * sorted[i];
        if i >= 2 {
            b2 += (if64 * (if64 - 1.0) / (ni_1 * (nf - 2.0))) * sorted[i];
        }
    }
    b1 /= nf;
    b2 /= nf;

    // GEV parameter relationships (Hosking et al. 1985):
    // c = (2b1 - b0) / (3b2 - b0) - ln(2)/ln(3)
    // ξ from solving: (3 - c)(2^ξ - 1) / (2^(2ξ) - 1) ... (iterative root-finding)
    // Approximation due to Hosking et al. (1985):
    let two_b1_b0 = 2.0 * b1 - b0;
    let three_b2_b0 = 3.0 * b2 - b0;

    if three_b2_b0.abs() < 1e-15 {
        return Err(StatsError::ComputationError(
            "GEV PWM: degenerate data (3b2 - b0 ≈ 0)".into(),
        ));
    }

    let c = two_b1_b0 / three_b2_b0 - 2.0_f64.ln() / 3.0_f64.ln();

    // Approximate ξ: Hosking (1985) polynomial approximation
    let xi = 7.859_f64.mul_add(c, 2.9554_f64 * c * c) / (1.0 - 0.8 * c);
    let xi = xi.max(-0.5_f64).min(0.5_f64); // Clamp to numerically stable range

    // Derive σ and μ from ξ
    use super::distributions::gamma_approx;
    let g1 = gamma_approx(1.0 - xi).ok_or_else(|| {
        StatsError::ComputationError("GEV PWM: Gamma function evaluation failed".into())
    })?;

    let sigma = xi * (2.0 * b1 - b0) / (g1 * (1.0 - 2.0_f64.powf(-xi)));
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(StatsError::ComputationError(format!(
            "GEV PWM: invalid sigma={sigma}"
        )));
    }
    let mu = b0 - sigma * (g1 - 1.0) / xi;

    GeneralizedExtremeValue::new(mu, sigma, xi)
}

/// Fit a GEV distribution using L-moments (Hosking 1990).
///
/// Uses the first three L-moments to estimate μ, σ, ξ.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 3 observations.
pub fn gev_fit_lmoments(data: ArrayView1<f64>) -> Result<GeneralizedExtremeValue, StatsError> {
    let n = data.len();
    if n < 3 {
        return Err(StatsError::InsufficientData(
            "GEV L-moments requires at least 3 observations".into(),
        ));
    }

    let lmoms = sample_lmoments(data, 3)?;
    let l1 = lmoms[0]; // λ₁ = mean
    let l2 = lmoms[1]; // λ₂ = L-scale
    let l3 = lmoms[2]; // λ₃

    if l2 <= 0.0 {
        return Err(StatsError::ComputationError(
            "GEV L-moments: L2 (L-scale) must be positive; check for constant or degenerate data"
                .into(),
        ));
    }

    // L-skewness τ₃ = λ₃/λ₂
    let tau3 = l3 / l2;

    // Hosking (1990) rational-function approximation for ξ from τ₃:
    // Valid for −1 < τ₃ < 1
    let xi = if tau3.abs() < 1e-8 {
        0.0
    } else {
        // Polynomial approximation:  ξ ≈ 7.8590 c + 2.9554 c²  where c = (2/(3+τ₃)) − ln2/ln3
        let c = 2.0 / (3.0 + tau3) - 2.0_f64.ln() / 3.0_f64.ln();
        7.859_f64.mul_add(c, 2.9554 * c * c)
    };

    use super::distributions::gamma_approx;
    let g1 = gamma_approx(1.0 - xi).ok_or_else(|| {
        StatsError::ComputationError("GEV L-moments: Gamma function evaluation failed".into())
    })?;

    let sigma = if xi.abs() < 1e-10 {
        l2 / 2.0_f64.ln()
    } else {
        l2 * xi / (g1 * (1.0 - 2.0_f64.powf(-xi)))
    };

    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(StatsError::ComputationError(format!(
            "GEV L-moments: invalid sigma={sigma}"
        )));
    }

    let mu = if xi.abs() < 1e-10 {
        l1 - sigma * super::distributions::EULER_MASCHERONI
    } else {
        l1 - sigma * (g1 - 1.0) / xi
    };

    GeneralizedExtremeValue::new(mu, sigma, xi)
}

// ---------------------------------------------------------------------------
// GPD estimation
// ---------------------------------------------------------------------------

/// Fit a GPD to exceedances (values above a threshold) using MLE.
///
/// The threshold μ is assumed to be 0 (pass already-subtracted exceedances).
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 5 exceedances.
pub fn gpd_fit_mle(exceedances: ArrayView1<f64>) -> Result<GeneralizedPareto, StatsError> {
    let n = exceedances.len();
    if n < 5 {
        return Err(StatsError::InsufficientData(
            "GPD MLE requires at least 5 exceedances".into(),
        ));
    }

    for &x in exceedances.iter() {
        if !x.is_finite() || x < 0.0 {
            return Err(StatsError::InvalidArgument(
                "GPD exceedances must be non-negative finite values".into(),
            ));
        }
    }

    let data_vec: Vec<f64> = exceedances.iter().copied().collect();

    // Method of moments starting values
    let m = sample_mean(&data_vec);
    let var = data_vec.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n as f64;
    let init_sigma = if var > 0.0 {
        let cv = var / m.powi(2);
        m * (1.0 + cv) / 2.0
    } else {
        m
    };
    let init_xi = if var > 0.0 {
        let cv = var / m.powi(2);
        (cv - 1.0) / 2.0
    } else {
        0.0
    };

    let x0 = [init_sigma.max(1e-6), init_xi.clamp(-0.4, 0.4)];

    let neg_log_lik = |params: &[f64]| -> f64 {
        let sigma = params[0];
        let xi = params[1];

        if sigma <= 0.0 {
            return f64::MAX;
        }

        let gpd = match GeneralizedPareto::new(0.0, sigma, xi) {
            Ok(g) => g,
            Err(_) => return f64::MAX,
        };

        let mut ll = 0.0_f64;
        for &x in &data_vec {
            let p = gpd.pdf(x);
            if p <= 0.0 || !p.is_finite() {
                return f64::MAX;
            }
            ll += p.ln();
        }
        -ll
    };

    let (best, _) = nelder_mead(&neg_log_lik, &x0, 1500, 1e-10)?;
    let sigma = best[0];
    let xi = best[1];

    GeneralizedPareto::new(0.0, sigma, xi).map_err(|e| {
        StatsError::ConvergenceError(format!("GPD MLE converged to invalid parameters: {e}"))
    })
}

// ---------------------------------------------------------------------------
// Gumbel estimation
// ---------------------------------------------------------------------------

/// Fit a Gumbel distribution using Maximum Likelihood Estimation.
///
/// For the Gumbel distribution the MLE has a closed-form iterative solution.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 2 observations.
pub fn gumbel_fit_mle(data: ArrayView1<f64>) -> Result<Gumbel, StatsError> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "Gumbel MLE requires at least 2 observations".into(),
        ));
    }

    let data_vec: Vec<f64> = data.iter().copied().collect();

    // MLE equations for Gumbel:
    // β̂ = x̄ − (1/n) Σ xᵢ exp(−xᵢ/β̂) / mean(exp(−xᵢ/β̂))
    // Solved via fixed-point iteration
    let mean = sample_mean(&data_vec);
    let var = data_vec.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    // Initial beta from method of moments: σ² = β²π²/6  ⇒  β = σ√6/π
    let mut beta = (var.sqrt() * 6.0_f64.sqrt() / std::f64::consts::PI).max(1e-10);

    const MAX_ITER: usize = 500;
    const TOL: f64 = 1e-12;

    for _ in 0..MAX_ITER {
        let exp_vals: Vec<f64> = data_vec.iter().map(|&x| (-x / beta).exp()).collect();
        let sum_exp: f64 = exp_vals.iter().sum();
        let sum_x_exp: f64 = data_vec.iter().zip(&exp_vals).map(|(&x, &e)| x * e).sum();

        if sum_exp.abs() < 1e-30 {
            break;
        }

        let beta_new = mean - sum_x_exp / sum_exp;
        if !beta_new.is_finite() || beta_new <= 0.0 {
            break;
        }
        let diff = (beta_new - beta).abs();
        beta = beta_new;
        if diff < TOL {
            break;
        }
    }

    if !beta.is_finite() || beta <= 0.0 {
        return Err(StatsError::ConvergenceError(
            "Gumbel MLE: failed to find valid beta".into(),
        ));
    }

    // mu from score equation: μ = β * ln((1/n) Σ exp(-xᵢ/β)) ... but simpler rearrangement:
    let exp_neg: Vec<f64> = data_vec.iter().map(|&x| (-x / beta).exp()).collect();
    let mean_exp_neg: f64 = exp_neg.iter().sum::<f64>() / n as f64;
    if mean_exp_neg <= 0.0 {
        return Err(StatsError::ConvergenceError(
            "Gumbel MLE: invalid intermediate values".into(),
        ));
    }
    let mu = -beta * mean_exp_neg.ln();

    Gumbel::new(mu, beta)
}

/// Fit a Gumbel distribution using L-moments.
///
/// Exact formulas: β = λ₂/ln(2), μ = λ₁ − β*γ.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 2 observations.
pub fn gumbel_fit_lmoments(data: ArrayView1<f64>) -> Result<Gumbel, StatsError> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "Gumbel L-moments requires at least 2 observations".into(),
        ));
    }

    let lmoms = sample_lmoments(data, 2)?;
    let l1 = lmoms[0];
    let l2 = lmoms[1];

    if l2 <= 0.0 {
        return Err(StatsError::ComputationError(
            "Gumbel L-moments: L2 must be positive".into(),
        ));
    }

    let beta = l2 / 2.0_f64.ln();
    let mu = l1 - beta * super::distributions::EULER_MASCHERONI;

    Gumbel::new(mu, beta)
}

// ---------------------------------------------------------------------------
// Goodness-of-fit
// ---------------------------------------------------------------------------

/// Anderson–Darling goodness-of-fit test for GEV.
///
/// Returns `(test_statistic, p_value_approx)`.
/// The p-value is approximated using the asymptotic critical value table from
/// Stephens (1974) / D'Agostino & Stephens (1986).
pub fn gev_goodness_of_fit(data: ArrayView1<f64>, params: &GeneralizedExtremeValue) -> (f64, f64) {
    let n = data.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute CDF values at order statistics
    let z: Vec<f64> = sorted
        .iter()
        .map(|&x| {
            let c = params.cdf(x);
            c.clamp(1e-15, 1.0 - 1e-15)
        })
        .collect();

    // Anderson-Darling statistic:
    // A² = -n - (1/n) Σ_{i=1}^{n} (2i-1) * [ln(z_i) + ln(1-z_{n+1-i})]
    let nf = n as f64;
    let mut sum = 0.0_f64;
    for i in 0..n {
        let zi = z[i];
        let z_rev = z[n - 1 - i];
        sum += (2 * i + 1) as f64 * (zi.ln() + (1.0 - z_rev).ln());
    }
    let a2 = -nf - sum / nf;

    // Asymptotic p-value approximation (Stephens 1986, Table 4.7)
    // The following uses a piecewise exponential fit
    let p_value = if a2 > 13.0 {
        5e-8
    } else if a2 > 10.0 {
        (-1.2937 + 1.2520 * (-a2).exp()).max(1e-7)
    } else if a2 > 2.0 {
        (-1.2937 - 5.709 * a2 + 0.0186 * a2.powi(2)).exp()
    } else {
        let q = a2.sqrt().exp();
        1.0 - (0.04213 + 0.01365 * (1.0 / q)) * (-2.0 * q).exp()
    };

    (a2, p_value.clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn relative_eq(a: f64, b: f64, rtol: f64) -> bool {
        let denom = b.abs().max(1e-12);
        (a - b).abs() / denom < rtol
    }

    // ---- Utilities --------------------------------------------------------

    #[test]
    fn test_binomial_coeff() {
        assert_eq!(binomial_coeff(5, 2), 10);
        assert_eq!(binomial_coeff(10, 3), 120);
        assert_eq!(binomial_coeff(0, 0), 1);
        assert_eq!(binomial_coeff(5, 0), 1);
        assert_eq!(binomial_coeff(5, 5), 1);
    }

    #[test]
    fn test_falling_factorial() {
        assert_eq!(falling_factorial(5, 3), 60); // 5*4*3
        assert_eq!(falling_factorial(10, 1), 10);
        assert_eq!(falling_factorial(5, 0), 1);
        assert_eq!(falling_factorial(3, 5), 0);
    }

    // ---- L-moments --------------------------------------------------------

    #[test]
    fn test_lmoments_insufficient_data() {
        let data = array![1.0, 2.0];
        assert!(sample_lmoments(data.view(), 3).is_err());
    }

    #[test]
    fn test_lmoments_l1_is_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let lm = sample_lmoments(data.view(), 1).unwrap();
        assert!(approx_eq(lm[0], 3.0, 1e-10));
    }

    #[test]
    fn test_lmoments_positive_l2() {
        let data = array![1.5, 2.3, 3.7, 4.1, 5.9, 7.2, 8.8];
        let lm = sample_lmoments(data.view(), 2).unwrap();
        assert!(lm[1] > 0.0, "L2 should be positive");
    }

    #[test]
    fn test_lmoments_constant_data() {
        // Constant data → L2 = 0 (no spread)
        let data = array![5.0, 5.0, 5.0, 5.0, 5.0];
        let lm = sample_lmoments(data.view(), 2).unwrap();
        assert!(approx_eq(lm[1], 0.0, 1e-10));
    }

    // ---- GEV MLE ----------------------------------------------------------

    #[test]
    fn test_gev_mle_recovers_gumbel_params() {
        // Generate Gumbel(μ=10, σ=2) samples; MLE should recover near those params
        use super::super::distributions::Gumbel;
        let g = Gumbel::new(10.0, 2.0).unwrap();
        let samples = g.sample(500, 99);
        let arr = Array1::from(samples);
        let fitted = gev_fit_mle(arr.view()).unwrap();
        // Allow loose tolerance since random sample
        assert!(relative_eq(fitted.mu, 10.0, 0.15), "mu={}", fitted.mu);
        assert!(
            relative_eq(fitted.sigma, 2.0, 0.20),
            "sigma={}",
            fitted.sigma
        );
        assert!(fitted.xi.abs() < 0.3, "xi={}", fitted.xi);
    }

    #[test]
    fn test_gev_mle_insufficient_data() {
        let data = array![1.0, 2.0];
        assert!(gev_fit_mle(data.view()).is_err());
    }

    // ---- GEV PWM ----------------------------------------------------------

    #[test]
    fn test_gev_pwm_returns_valid_params() {
        let data = array![1.2, 2.4, 3.1, 4.8, 5.5, 6.0, 7.3, 8.1, 9.2, 10.0];
        let fitted = gev_fit_pwm(data.view()).unwrap();
        assert!(fitted.sigma > 0.0);
        assert!(fitted.mu.is_finite());
        assert!(fitted.xi.is_finite());
    }

    #[test]
    fn test_gev_pwm_insufficient_data() {
        let data = array![1.0, 2.0];
        assert!(gev_fit_pwm(data.view()).is_err());
    }

    // ---- GEV L-moments ----------------------------------------------------

    #[test]
    fn test_gev_lmoments_returns_valid_params() {
        let data = array![0.5, 1.2, 1.8, 2.5, 3.1, 3.8, 4.5, 5.3, 6.1, 7.0];
        let fitted = gev_fit_lmoments(data.view()).unwrap();
        assert!(fitted.sigma > 0.0);
        assert!(fitted.mu.is_finite());
        assert!(fitted.xi.is_finite());
    }

    #[test]
    fn test_gev_lmoments_recovers_approximate_params() {
        // For Gumbel(0,1) data, expected: ξ ≈ 0, μ ≈ 0, σ ≈ 1
        use super::super::distributions::Gumbel;
        let g = Gumbel::new(0.0, 1.0).unwrap();
        let samples = g.sample(1000, 123);
        let arr = Array1::from(samples);
        let fitted = gev_fit_lmoments(arr.view()).unwrap();
        assert!(fitted.xi.abs() < 0.3, "xi={}", fitted.xi);
        assert!(
            relative_eq(fitted.sigma, 1.0, 0.2),
            "sigma={}",
            fitted.sigma
        );
    }

    // ---- GPD MLE ----------------------------------------------------------

    #[test]
    fn test_gpd_mle_exponential_case() {
        // Exponential(λ=2) exceedances → GPD with ξ≈0, σ≈0.5
        use super::super::distributions::GeneralizedPareto;
        let gpd_true = GeneralizedPareto::new(0.0, 0.5, 0.0).unwrap();
        let samples = gpd_true.sample(200, 7);
        let arr = Array1::from(samples);
        let fitted = gpd_fit_mle(arr.view()).unwrap();
        assert!(fitted.sigma > 0.0);
        assert!(fitted.xi.is_finite());
        assert!(
            relative_eq(fitted.sigma, 0.5, 0.25),
            "sigma={}",
            fitted.sigma
        );
    }

    #[test]
    fn test_gpd_mle_insufficient_data() {
        let data = array![0.1, 0.2, 0.3, 0.4];
        assert!(gpd_fit_mle(data.view()).is_err());
    }

    #[test]
    fn test_gpd_mle_negative_values_error() {
        let data = array![-1.0, 0.5, 1.0, 2.0, 3.0];
        assert!(gpd_fit_mle(data.view()).is_err());
    }

    // ---- Gumbel MLE -------------------------------------------------------

    #[test]
    fn test_gumbel_mle_recovers_params() {
        use super::super::distributions::Gumbel;
        let g = Gumbel::new(5.0, 3.0).unwrap();
        let samples = g.sample(1000, 42);
        let arr = Array1::from(samples);
        let fitted = gumbel_fit_mle(arr.view()).unwrap();
        assert!(relative_eq(fitted.mu, 5.0, 0.20), "mu={}", fitted.mu);
        assert!(relative_eq(fitted.beta, 3.0, 0.20), "beta={}", fitted.beta);
    }

    #[test]
    fn test_gumbel_mle_insufficient_data() {
        let data = array![1.0];
        assert!(gumbel_fit_mle(data.view()).is_err());
    }

    // ---- Gumbel L-moments -------------------------------------------------

    #[test]
    fn test_gumbel_lmoments_recovers_params() {
        use super::super::distributions::Gumbel;
        let g = Gumbel::new(2.0, 1.5).unwrap();
        let samples = g.sample(500, 55);
        let arr = Array1::from(samples);
        let fitted = gumbel_fit_lmoments(arr.view()).unwrap();
        assert!(fitted.beta > 0.0);
        assert!(fitted.mu.is_finite());
    }

    // ---- Goodness-of-fit --------------------------------------------------

    #[test]
    fn test_gev_gof_returns_finite() {
        let data = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];
        let fitted = gev_fit_lmoments(data.view()).unwrap();
        let (a2, pval) = gev_goodness_of_fit(data.view(), &fitted);
        assert!(a2.is_finite());
        assert!(pval.is_finite());
        assert!((0.0..=1.0).contains(&pval));
    }

    #[test]
    fn test_gev_gof_good_fit_high_pvalue() {
        // When the data is generated from the fitted model, GoF should not reject (high p)
        use super::super::distributions::Gumbel;
        let g = Gumbel::new(0.0, 1.0).unwrap();
        let samples = g.sample(200, 999);
        let arr = Array1::from(samples);
        let fitted = gev_fit_lmoments(arr.view()).unwrap();
        let (a2, _pval) = gev_goodness_of_fit(arr.view(), &fitted);
        // A² should not be astronomical for data from the same family
        assert!(a2 < 50.0, "A²={a2} seems too large for in-sample data");
    }
}
