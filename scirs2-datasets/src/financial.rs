//! Financial Dataset Generators.
//!
//! Provides stochastic price-path simulators and summary statistics commonly
//! used in quantitative finance, algorithmic trading, and risk management
//! research.
//!
//! # Available functions
//!
//! | Function | Description |
//! |---|---|
//! | [`gbm_prices`] | Geometric Brownian Motion price paths |
//! | [`correlated_gbm`] | Multi-asset correlated GBM |
//! | [`synthetic_order_book`] | Synthetic limit-order book snapshot |
//! | [`log_returns`] | Log-returns from a price series |
//! | [`rolling_volatility`] | Rolling standard deviation of returns |
//! | [`rolling_sharpe`] | Rolling Sharpe ratio |

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of a limit-order book.
///
/// `bids` are sorted in descending price order; `asks` in ascending order.
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Bid levels: `(price, quantity)` sorted descending.
    pub bids: Vec<(f64, f64)>,
    /// Ask levels: `(price, quantity)` sorted ascending.
    pub asks: Vec<(f64, f64)>,
    /// Mid-price: average of best bid and best ask.
    pub mid_price: f64,
    /// Bid-ask spread.
    pub spread: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn normal_dist(
    std: f64,
) -> Result<scirs2_core::random::rand_distributions::Normal<f64>> {
    scirs2_core::random::rand_distributions::Normal::new(0.0_f64, std).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })
}

fn uniform_dist(
    lo: f64,
    hi: f64,
) -> Result<scirs2_core::random::rand_distributions::Uniform<f64>> {
    scirs2_core::random::rand_distributions::Uniform::new(lo, hi).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform distribution creation failed: {e}"))
    })
}

/// Cholesky decomposition of a symmetric positive-definite matrix (lower triangular).
///
/// Uses the Banachiewicz algorithm in O(n³) time.
fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(DatasetsError::InvalidFormat(
            "cholesky_lower: matrix must be square".to_string(),
        ));
    }
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return Err(DatasetsError::ComputationError(
                        "cholesky_lower: matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, j]] = diag.sqrt();
            } else {
                let ljj = l[[j, j]];
                if ljj.abs() < 1e-15 {
                    return Err(DatasetsError::ComputationError(
                        "cholesky_lower: near-zero diagonal in L".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / ljj;
            }
        }
    }
    Ok(l)
}

// ─────────────────────────────────────────────────────────────────────────────
// GBM single-asset
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate stock price paths under Geometric Brownian Motion (GBM).
///
/// Each path follows:
/// ```text
/// S(t + dt) = S(t) · exp[(μ − σ²/2) dt + σ √dt · Z]
/// ```
/// where `Z ~ N(0,1)` is an independent increment.
///
/// # Arguments
///
/// * `s0`      – Initial price (must be > 0).
/// * `mu`      – Annualised drift (log-return mean per unit time).
/// * `sigma`   – Annualised volatility (must be > 0).
/// * `dt`      – Time step (must be > 0).
/// * `n_steps` – Number of time steps per path.
/// * `n_paths` – Number of independent paths.
/// * `seed`    – Random seed.
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_paths, n_steps + 1)`.  Column 0 contains `s0`.
///
/// # Errors
///
/// Returns an error when `s0 ≤ 0`, `sigma ≤ 0`, `dt ≤ 0`, `n_steps == 0`, or
/// `n_paths == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::financial::gbm_prices;
///
/// let paths = gbm_prices(100.0, 0.05, 0.2, 1.0/252.0, 252, 10, 42)
///     .expect("gbm failed");
/// assert_eq!(paths.nrows(), 10);
/// assert_eq!(paths.ncols(), 253);
/// ```
pub fn gbm_prices(
    s0: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Result<Array2<f64>> {
    if s0 <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "gbm_prices: s0 must be > 0".to_string(),
        ));
    }
    if sigma <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "gbm_prices: sigma must be > 0".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "gbm_prices: dt must be > 0".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(DatasetsError::InvalidFormat(
            "gbm_prices: n_steps must be > 0".to_string(),
        ));
    }
    if n_paths == 0 {
        return Err(DatasetsError::InvalidFormat(
            "gbm_prices: n_paths must be > 0".to_string(),
        ));
    }

    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();

    let mut rng = make_rng(seed);
    let z_dist = normal_dist(1.0)?;

    let mut out = Array2::zeros((n_paths, n_steps + 1));
    for p in 0..n_paths {
        out[[p, 0]] = s0;
        for step in 0..n_steps {
            let z = z_dist.sample(&mut rng);
            out[[p, step + 1]] = out[[p, step]] * (drift + diffusion * z).exp();
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Correlated GBM
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate multi-asset correlated Geometric Brownian Motion paths.
///
/// Correlated Brownian increments are generated via Cholesky decomposition of
/// the correlation matrix:
/// ```text
/// dW_i = Σ_j L_{ij} Z_j √dt,  Z_j ~ N(0,1) i.i.d.
/// ```
///
/// # Arguments
///
/// * `s0`          – Initial prices (length `n_assets`).
/// * `mu`          – Per-asset drifts.
/// * `correlation` – `n_assets × n_assets` correlation matrix (symmetric, PD).
/// * `sigma`       – Per-asset volatilities (all must be > 0).
/// * `dt`          – Time step.
/// * `n_steps`     – Number of steps.
/// * `seed`        – Random seed.
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_steps + 1, n_assets)`.  Row 0 contains `s0`.
///
/// # Errors
///
/// Returns an error when lengths mismatch, the correlation matrix is not PD,
/// or any parameter violates its domain constraint.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::financial::correlated_gbm;
/// use scirs2_core::ndarray::array;
///
/// let corr = array![[1.0, 0.7], [0.7, 1.0]];
/// let paths = correlated_gbm(
///     &[100.0, 50.0],
///     &[0.05, 0.08],
///     &corr,
///     &[0.2, 0.3],
///     1.0 / 252.0,
///     252,
///     1,
/// ).expect("correlated_gbm failed");
/// assert_eq!(paths.nrows(), 253);
/// assert_eq!(paths.ncols(), 2);
/// ```
pub fn correlated_gbm(
    s0: &[f64],
    mu: &[f64],
    correlation: &Array2<f64>,
    sigma: &[f64],
    dt: f64,
    n_steps: usize,
    seed: u64,
) -> Result<Array2<f64>> {
    let n_assets = s0.len();
    if mu.len() != n_assets || sigma.len() != n_assets {
        return Err(DatasetsError::InvalidFormat(
            "correlated_gbm: s0, mu, and sigma must have the same length".to_string(),
        ));
    }
    if correlation.nrows() != n_assets || correlation.ncols() != n_assets {
        return Err(DatasetsError::InvalidFormat(
            "correlated_gbm: correlation matrix dimensions must match n_assets".to_string(),
        ));
    }
    for (i, &s) in s0.iter().enumerate() {
        if s <= 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "correlated_gbm: s0[{i}] must be > 0"
            )));
        }
    }
    for (i, &v) in sigma.iter().enumerate() {
        if v <= 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "correlated_gbm: sigma[{i}] must be > 0"
            )));
        }
    }
    if dt <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "correlated_gbm: dt must be > 0".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(DatasetsError::InvalidFormat(
            "correlated_gbm: n_steps must be > 0".to_string(),
        ));
    }

    // Convert correlation to covariance: Σ_ij = ρ_ij σ_i σ_j.
    let mut cov = Array2::zeros((n_assets, n_assets));
    for i in 0..n_assets {
        for j in 0..n_assets {
            cov[[i, j]] = correlation[[i, j]] * sigma[i] * sigma[j];
        }
    }

    // Cholesky factor L such that Σ = L Lᵀ.
    let l = cholesky_lower(&cov)?;

    let mut rng = make_rng(seed);
    let z_dist = normal_dist(1.0)?;

    let mut out = Array2::zeros((n_steps + 1, n_assets));
    for a in 0..n_assets {
        out[[0, a]] = s0[a];
    }

    let sqrt_dt = dt.sqrt();
    for step in 0..n_steps {
        // Independent standard normals.
        let z_raw: Vec<f64> = (0..n_assets).map(|_| z_dist.sample(&mut rng)).collect();

        // Correlated increments: dW = L * z * sqrt(dt).
        for a in 0..n_assets {
            let mut dw = 0.0_f64;
            for k in 0..n_assets {
                dw += l[[a, k]] * z_raw[k];
            }
            dw *= sqrt_dt;
            let drift = (mu[a] - 0.5 * sigma[a] * sigma[a]) * dt;
            out[[step + 1, a]] = out[[step, a]] * (drift + dw).exp();
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Order book
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic limit-order book snapshot.
///
/// Bid and ask levels are placed at uniformly-spaced ticks from the mid-price,
/// with exponentially decreasing quantities (thin at the touch, thick further
/// away).
///
/// # Arguments
///
/// * `mid_price` – Mid-price (must be > 0).
/// * `n_levels`  – Number of bid/ask levels each side (must be ≥ 1).
/// * `tick_size` – Price increment per level (must be > 0).
/// * `seed`      – Random seed for quantity noise.
///
/// # Errors
///
/// Returns an error when `mid_price ≤ 0`, `n_levels == 0`, or `tick_size ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::financial::synthetic_order_book;
///
/// let ob = synthetic_order_book(100.0, 5, 0.01, 42).expect("ob failed");
/// assert_eq!(ob.bids.len(), 5);
/// assert_eq!(ob.asks.len(), 5);
/// assert!(ob.spread > 0.0);
/// ```
pub fn synthetic_order_book(
    mid_price: f64,
    n_levels: usize,
    tick_size: f64,
    seed: u64,
) -> Result<OrderBook> {
    if mid_price <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "synthetic_order_book: mid_price must be > 0".to_string(),
        ));
    }
    if n_levels == 0 {
        return Err(DatasetsError::InvalidFormat(
            "synthetic_order_book: n_levels must be >= 1".to_string(),
        ));
    }
    if tick_size <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "synthetic_order_book: tick_size must be > 0".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let noise = uniform_dist(0.8, 1.2)?;

    // Best bid is one tick below the mid-price; best ask is one tick above.
    let best_bid = mid_price - 0.5 * tick_size;
    let best_ask = mid_price + 0.5 * tick_size;

    let mut bids = Vec::with_capacity(n_levels);
    let mut asks = Vec::with_capacity(n_levels);

    for k in 0..n_levels {
        // Exponentially increasing quantity from the touch outwards.
        let base_qty = 10.0 * (1.5_f64).powi(k as i32);
        let qty = base_qty * noise.sample(&mut rng);

        bids.push((best_bid - k as f64 * tick_size, qty));
        asks.push((best_ask + k as f64 * tick_size, qty * noise.sample(&mut rng)));
    }

    // Bids: descending price order (highest price = best bid).
    bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Asks: ascending price order (lowest price = best ask).
    asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let best_bid_px = bids.first().map(|b| b.0).unwrap_or(0.0);
    let best_ask_px = asks.first().map(|a| a.0).unwrap_or(0.0);
    let computed_mid = (best_bid_px + best_ask_px) / 2.0;
    let spread = best_ask_px - best_bid_px;

    Ok(OrderBook {
        bids,
        asks,
        mid_price: computed_mid,
        spread,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-returns
// ─────────────────────────────────────────────────────────────────────────────

/// Compute log-returns from a price series.
///
/// ```text
/// r[t] = ln(S[t] / S[t-1])
/// ```
///
/// The output has length `prices.len() - 1`.
///
/// # Errors
///
/// Returns an error when the price series has fewer than 2 elements or
/// contains non-positive values.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::financial::log_returns;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![100.0, 110.0, 105.0, 115.0];
/// let r = log_returns(&prices).expect("log_returns failed");
/// assert_eq!(r.len(), 3);
/// ```
pub fn log_returns(prices: &Array1<f64>) -> Result<Array1<f64>> {
    if prices.len() < 2 {
        return Err(DatasetsError::InvalidFormat(
            "log_returns: prices must have at least 2 elements".to_string(),
        ));
    }

    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        let prev = prices[i - 1];
        let curr = prices[i];
        if prev <= 0.0 || curr <= 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "log_returns: non-positive price at index {}: prev={prev}, curr={curr}",
                i
            )));
        }
        returns.push((curr / prev).ln());
    }

    Ok(Array1::from_vec(returns))
}

// ─────────────────────────────────────────────────────────────────────────────
// Rolling volatility
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a rolling standard deviation (volatility) of a return series.
///
/// Uses an unbiased (Bessel-corrected) estimator.  The first `window - 1`
/// entries of the output are `NaN` since there are insufficient observations.
///
/// # Arguments
///
/// * `returns` – Return series.
/// * `window`  – Rolling window size (must be ≥ 2).
///
/// # Errors
///
/// Returns an error when `window < 2` or `returns` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::financial::{log_returns, rolling_volatility};
/// use scirs2_core::ndarray::array;
///
/// let prices = array![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];
/// let r = log_returns(&prices).expect("log_returns failed");
/// let vol = rolling_volatility(&r, 3).expect("rolling_vol failed");
/// assert_eq!(vol.len(), r.len());
/// ```
pub fn rolling_volatility(returns: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
    if window < 2 {
        return Err(DatasetsError::InvalidFormat(
            "rolling_volatility: window must be >= 2".to_string(),
        ));
    }
    if returns.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "rolling_volatility: returns must not be empty".to_string(),
        ));
    }

    let n = returns.len();
    let data: Vec<f64> = returns.to_vec();
    let mut out = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[i + 1 - window..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (slice.len() as f64 - 1.0);
        out[i] = var.sqrt();
    }

    Ok(Array1::from_vec(out))
}

// ─────────────────────────────────────────────────────────────────────────────
// Rolling Sharpe ratio
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a rolling Sharpe ratio for a return series.
///
/// ```text
/// Sharpe(t) = (mean(r[t-w+1..t]) − risk_free) / std(r[t-w+1..t])
/// ```
///
/// Uses the annualised standard deviation with Bessel correction.
/// The first `window - 1` entries are `NaN`.
///
/// # Arguments
///
/// * `returns`   – Return series (per-period, not annualised).
/// * `window`    – Rolling window size (must be ≥ 2).
/// * `risk_free` – Per-period risk-free rate.
///
/// # Errors
///
/// Returns an error when `window < 2` or `returns` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::financial::{log_returns, rolling_sharpe};
/// use scirs2_core::ndarray::array;
///
/// let prices = array![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];
/// let r = log_returns(&prices).expect("log_returns failed");
/// let sharpe = rolling_sharpe(&r, 3, 0.0).expect("rolling_sharpe failed");
/// assert_eq!(sharpe.len(), r.len());
/// ```
pub fn rolling_sharpe(returns: &Array1<f64>, window: usize, risk_free: f64) -> Result<Array1<f64>> {
    if window < 2 {
        return Err(DatasetsError::InvalidFormat(
            "rolling_sharpe: window must be >= 2".to_string(),
        ));
    }
    if returns.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "rolling_sharpe: returns must not be empty".to_string(),
        ));
    }

    let n = returns.len();
    let data: Vec<f64> = returns.to_vec();
    let mut out = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[i + 1 - window..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (slice.len() as f64 - 1.0);
        let std_dev = var.sqrt();
        if std_dev < 1e-15 {
            out[i] = 0.0;
        } else {
            out[i] = (mean - risk_free) / std_dev;
        }
    }

    Ok(Array1::from_vec(out))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── GBM ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_gbm_shape() {
        let paths = gbm_prices(100.0, 0.05, 0.2, 1.0 / 252.0, 252, 10, 42)
            .expect("gbm failed");
        assert_eq!(paths.nrows(), 10);
        assert_eq!(paths.ncols(), 253); // n_steps + 1
    }

    #[test]
    fn test_gbm_initial_price() {
        let paths = gbm_prices(50.0, 0.0, 0.1, 0.01, 100, 5, 1).expect("gbm failed");
        for p in 0..5 {
            assert!((paths[[p, 0]] - 50.0).abs() < 1e-12, "Initial price wrong");
        }
    }

    #[test]
    fn test_gbm_positive_prices() {
        let paths = gbm_prices(100.0, 0.05, 0.4, 1.0 / 252.0, 252, 50, 7)
            .expect("gbm failed");
        for p in 0..paths.nrows() {
            for t in 0..paths.ncols() {
                assert!(paths[[p, t]] > 0.0, "Non-positive price at ({p},{t})");
            }
        }
    }

    #[test]
    fn test_gbm_log_normal_distribution() {
        // Log-prices at terminal step should be approximately normally distributed.
        // Verify: mean of log(S_T / S_0) ≈ (mu - 0.5 σ²) T for many paths.
        let n_paths = 10_000;
        let mu = 0.05_f64;
        let sigma = 0.2_f64;
        let dt = 1.0 / 252.0_f64;
        let n_steps = 252;
        let paths = gbm_prices(100.0, mu, sigma, dt, n_steps, n_paths, 99)
            .expect("gbm failed");
        let t = n_steps as f64 * dt;
        let expected_mean = (mu - 0.5 * sigma * sigma) * t;
        let log_returns_final: Vec<f64> = (0..n_paths)
            .map(|p| (paths[[p, n_steps]] / paths[[p, 0]]).ln())
            .collect();
        let sample_mean =
            log_returns_final.iter().sum::<f64>() / log_returns_final.len() as f64;
        let tol = 0.03; // allow 3% tolerance for Monte Carlo noise
        assert!(
            (sample_mean - expected_mean).abs() < tol,
            "sample_mean={sample_mean:.4}, expected={expected_mean:.4}"
        );
    }

    #[test]
    fn test_gbm_reproducibility() {
        let a = gbm_prices(100.0, 0.0, 0.2, 0.01, 50, 3, 42).expect("gbm failed");
        let b = gbm_prices(100.0, 0.0, 0.2, 0.01, 50, 3, 42).expect("gbm failed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_gbm_error_negative_s0() {
        assert!(gbm_prices(-100.0, 0.05, 0.2, 0.01, 10, 1, 0).is_err());
    }

    #[test]
    fn test_gbm_error_zero_sigma() {
        assert!(gbm_prices(100.0, 0.05, 0.0, 0.01, 10, 1, 0).is_err());
    }

    // ── Correlated GBM ───────────────────────────────────────────────────────

    #[test]
    fn test_correlated_gbm_shape() {
        let corr = array![[1.0, 0.6], [0.6, 1.0]];
        let paths = correlated_gbm(
            &[100.0, 50.0],
            &[0.05, 0.08],
            &corr,
            &[0.2, 0.3],
            1.0 / 252.0,
            252,
            1,
        )
        .expect("correlated_gbm failed");
        assert_eq!(paths.nrows(), 253);
        assert_eq!(paths.ncols(), 2);
    }

    #[test]
    fn test_correlated_gbm_initial_prices() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let s0 = &[80.0, 120.0];
        let paths = correlated_gbm(s0, &[0.0, 0.0], &corr, &[0.1, 0.15], 0.01, 100, 3)
            .expect("correlated_gbm failed");
        assert!((paths[[0, 0]] - s0[0]).abs() < 1e-12);
        assert!((paths[[0, 1]] - s0[1]).abs() < 1e-12);
    }

    #[test]
    fn test_correlated_gbm_positive_prices() {
        let corr = array![[1.0, 0.8], [0.8, 1.0]];
        let paths =
            correlated_gbm(&[50.0, 75.0], &[0.1, 0.05], &corr, &[0.25, 0.2], 0.01, 200, 5)
                .expect("correlated_gbm failed");
        for i in 0..paths.nrows() {
            for j in 0..paths.ncols() {
                assert!(paths[[i, j]] > 0.0, "Non-positive price at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_correlated_gbm_correlation_preserved() {
        // With high correlation ρ=0.9, the two assets should move together more
        // often than not.  Verify by checking that the cross-correlation of
        // log-returns is closer to 0.9 than to 0.
        let n_steps = 2000;
        let corr = array![[1.0, 0.9], [0.9, 1.0]];
        let paths = correlated_gbm(
            &[100.0, 100.0],
            &[0.0, 0.0],
            &corr,
            &[0.2, 0.2],
            1.0 / 252.0,
            n_steps,
            42,
        )
        .expect("correlated_gbm failed");

        let n = paths.nrows() - 1;
        let r0: Vec<f64> = (0..n).map(|t| (paths[[t + 1, 0]] / paths[[t, 0]]).ln()).collect();
        let r1: Vec<f64> = (0..n).map(|t| (paths[[t + 1, 1]] / paths[[t, 1]]).ln()).collect();

        let mean0 = r0.iter().sum::<f64>() / n as f64;
        let mean1 = r1.iter().sum::<f64>() / n as f64;

        let cov: f64 =
            r0.iter().zip(r1.iter()).map(|(a, b)| (a - mean0) * (b - mean1)).sum::<f64>()
                / n as f64;
        let var0 = r0.iter().map(|a| (a - mean0).powi(2)).sum::<f64>() / n as f64;
        let var1 = r1.iter().map(|b| (b - mean1).powi(2)).sum::<f64>() / n as f64;

        let measured_corr = cov / (var0.sqrt() * var1.sqrt());
        assert!(
            (measured_corr - 0.9).abs() < 0.05,
            "Expected correlation ≈ 0.9, got {measured_corr:.4}"
        );
    }

    #[test]
    fn test_correlated_gbm_error_mismatched_inputs() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        assert!(correlated_gbm(
            &[100.0],
            &[0.05, 0.08],
            &corr,
            &[0.2, 0.3],
            0.01,
            10,
            0
        )
        .is_err());
    }

    // ── Order book ───────────────────────────────────────────────────────────

    #[test]
    fn test_order_book_levels() {
        let ob = synthetic_order_book(100.0, 5, 0.01, 42).expect("ob failed");
        assert_eq!(ob.bids.len(), 5);
        assert_eq!(ob.asks.len(), 5);
    }

    #[test]
    fn test_order_book_spread_positive() {
        let ob = synthetic_order_book(200.0, 10, 0.5, 0).expect("ob failed");
        assert!(ob.spread > 0.0, "Spread must be positive");
    }

    #[test]
    fn test_order_book_bid_ask_sorted() {
        let ob = synthetic_order_book(100.0, 5, 0.01, 7).expect("ob failed");
        // Bids descending.
        for w in ob.bids.windows(2) {
            assert!(w[0].0 >= w[1].0, "Bids not descending");
        }
        // Asks ascending.
        for w in ob.asks.windows(2) {
            assert!(w[0].0 <= w[1].0, "Asks not ascending");
        }
    }

    #[test]
    fn test_order_book_best_bid_below_ask() {
        let ob = synthetic_order_book(100.0, 3, 0.01, 1).expect("ob failed");
        let best_bid = ob.bids[0].0;
        let best_ask = ob.asks[0].0;
        assert!(best_bid < best_ask, "Best bid must be below best ask");
    }

    #[test]
    fn test_order_book_error_zero_levels() {
        assert!(synthetic_order_book(100.0, 0, 0.01, 0).is_err());
    }

    // ── Log-returns ──────────────────────────────────────────────────────────

    #[test]
    fn test_log_returns_shape() {
        let prices = array![100.0, 110.0, 105.0, 115.0];
        let r = log_returns(&prices).expect("log_returns failed");
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_log_returns_exponential_growth() {
        // If prices = exp(a, 2a, 3a, …) then log-returns = a for all steps.
        let a = 0.1_f64;
        let n = 10_usize;
        let prices = Array1::from_vec((0..n).map(|i| (a * i as f64).exp()).collect());
        let r = log_returns(&prices).expect("log_returns failed");
        for i in 0..r.len() {
            assert!((r[i] - a).abs() < 1e-10, "log_return[{i}] = {} expected {a}", r[i]);
        }
    }

    #[test]
    fn test_log_returns_error_too_short() {
        let prices = array![100.0];
        assert!(log_returns(&prices).is_err());
    }

    #[test]
    fn test_log_returns_error_non_positive() {
        let prices = array![100.0, 0.0, 50.0];
        assert!(log_returns(&prices).is_err());
    }

    // ── Rolling volatility ───────────────────────────────────────────────────

    #[test]
    fn test_rolling_volatility_shape() {
        let prices = array![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];
        let r = log_returns(&prices).expect("lr");
        let vol = rolling_volatility(&r, 3).expect("rvol");
        assert_eq!(vol.len(), r.len());
    }

    #[test]
    fn test_rolling_volatility_initial_nan() {
        let prices = array![100.0, 102.0, 101.0, 103.0, 105.0];
        let r = log_returns(&prices).expect("lr");
        let vol = rolling_volatility(&r, 3).expect("rvol");
        // First two entries should be NaN.
        assert!(vol[0].is_nan());
        assert!(vol[1].is_nan());
        assert!(!vol[2].is_nan());
    }

    #[test]
    fn test_rolling_volatility_constant_returns_zero() {
        // Constant returns → zero volatility.
        let r = Array1::from_vec(vec![0.01; 10]);
        let vol = rolling_volatility(&r, 4).expect("rvol");
        for i in 3..10 {
            assert!(vol[i].abs() < 1e-10, "vol[{i}]={}", vol[i]);
        }
    }

    #[test]
    fn test_rolling_volatility_error_window_too_small() {
        let r = Array1::from_vec(vec![0.01; 10]);
        assert!(rolling_volatility(&r, 1).is_err());
    }

    // ── Rolling Sharpe ───────────────────────────────────────────────────────

    #[test]
    fn test_rolling_sharpe_shape() {
        let prices = array![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];
        let r = log_returns(&prices).expect("lr");
        let sh = rolling_sharpe(&r, 3, 0.0).expect("sharpe");
        assert_eq!(sh.len(), r.len());
    }

    #[test]
    fn test_rolling_sharpe_initial_nan() {
        let r = Array1::from_vec(vec![0.01, -0.01, 0.02, 0.01, -0.005]);
        let sh = rolling_sharpe(&r, 3, 0.0).expect("sharpe");
        assert!(sh[0].is_nan());
        assert!(sh[1].is_nan());
        assert!(!sh[2].is_nan());
    }

    #[test]
    fn test_rolling_sharpe_positive_for_positive_returns() {
        // Uniformly positive returns with zero risk-free rate → positive Sharpe.
        let r = Array1::from_vec(vec![0.01, 0.02, 0.015, 0.012, 0.018]);
        let sh = rolling_sharpe(&r, 3, 0.0).expect("sharpe");
        for i in 2..sh.len() {
            assert!(sh[i] > 0.0, "Expected positive Sharpe at {i}, got {}", sh[i]);
        }
    }
}
