//! Fractional Brownian Motion (fBm) path generation.
//!
//! Fractional Brownian Motion (fBm) is a generalisation of standard Brownian
//! motion parametrised by the Hurst exponent H ∈ (0, 1):
//!
//! * H = 0.5 — standard Brownian motion (independent increments)
//! * H > 0.5 — long-range dependence (persistent)
//! * H < 0.5 — anti-persistent (negatively correlated increments)
//!
//! ## Covariance function
//!
//! ```text
//! R(s, t) = ½ (|s|^{2H} + |t|^{2H} − |s − t|^{2H})
//! ```
//!
//! ## Algorithms
//!
//! | Method | Time | Memory | Notes |
//! |--------|------|--------|-------|
//! | Hosking | O(n²) | O(n²) | exact, best for n ≤ 1 000 |
//! | Davies-Harte | O(n log n) | O(n) | exact for H ≠ 0.5, clamped for edge cases |
//!
//! ## References
//!
//! * Hosking, J.R.M. (1984). *Modeling persistence in hydrological time series*
//! * Davies, R.B., Harte, D.S. (1987). *Tests for Hurst effect*

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft, ifft};

// ---------------------------------------------------------------------------
// Internal LCG / splitmix64 PRNG (no rand crate dependency)
// ---------------------------------------------------------------------------

/// 64-bit splitmix64 PRNG combined with a linear congruential generator.
///
/// Provides a uniform [0, 1) float and standard-normal samples via Box-Muller.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Warm-up: one splitmix64 mix to avoid seed = 0 degeneracies
        let state = seed
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add(1_442_695_040_888_963_407_u64);
        Self { state }
    }

    /// Advance and return a value in [0, 1)
    fn next_f64(&mut self) -> f64 {
        // splitmix64 step
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15_u64);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9_u64);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb_u64);
        z = z ^ (z >> 31);
        // map to [0, 1)
        (z >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Box-Muller transform — returns a standard normal sample.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-300 {
                let mag = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                return mag * theta.cos(); // (sin-half discarded — cheap)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// Algorithm for generating fBm paths.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbmMethod {
    /// Exact Cholesky decomposition of the full covariance matrix.
    ///
    /// O(n²) time and memory.  Best for n ≤ 1 000.
    Hosking,
    /// Circulant embedding of the autocovariance via FFT.
    ///
    /// O(n log n) time, O(n) memory.  Best for n > 1 000.
    /// For H near 0 or 1 the circulant may have small negative eigenvalues;
    /// these are clamped to 0 (approximation).
    DaviesHarte,
}

/// Configuration for [`FractionalBrownianMotion`].
#[derive(Debug, Clone)]
pub struct FbmConfig {
    /// Hurst exponent H ∈ (0, 1).  H = 0.5 is standard BM.
    pub hurst: f64,
    /// Number of time steps.  The generated path has `n_steps + 1` points.
    pub n_steps: usize,
    /// Length of each time step.
    pub dt: f64,
    /// Seed for the internal PRNG.
    pub seed: u64,
    /// Algorithm to use for path generation.
    pub method: FbmMethod,
}

impl Default for FbmConfig {
    fn default() -> Self {
        Self {
            hurst: 0.7,
            n_steps: 256,
            dt: 1.0 / 256.0,
            seed: 42,
            method: FbmMethod::DaviesHarte,
        }
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Generator for fractional Brownian Motion sample paths.
///
/// # Example
///
/// ```rust
/// use scirs2_integrate::sde::fractional_brownian::{FractionalBrownianMotion, FbmConfig};
///
/// let cfg = FbmConfig { hurst: 0.7, n_steps: 64, dt: 1.0 / 64.0, seed: 0,
///                       ..Default::default() };
/// let fbm = FractionalBrownianMotion::new(cfg);
/// let path = fbm.sample_path().unwrap();
/// assert_eq!(path.len(), 65); // n_steps + 1
/// assert!((path[0]).abs() < 1e-15); // starts at zero
/// ```
pub struct FractionalBrownianMotion {
    config: FbmConfig,
}

impl FractionalBrownianMotion {
    /// Create a new generator from the given configuration.
    pub fn new(config: FbmConfig) -> Self {
        Self { config }
    }

    /// Validate the configuration.
    fn validate(&self) -> IntegrateResult<()> {
        if self.config.hurst <= 0.0 || self.config.hurst >= 1.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "Hurst exponent must be in (0, 1), got {}",
                self.config.hurst
            )));
        }
        if self.config.n_steps == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_steps must be at least 1".to_string(),
            ));
        }
        if self.config.dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "dt must be positive, got {}",
                self.config.dt
            )));
        }
        Ok(())
    }

    /// Generate the fractional Gaussian noise increments (length `n_steps`).
    ///
    /// The increments ξ_i = B^H_{(i+1)dt} − B^H_{i·dt} satisfy
    /// E[ξ_i²] = dt^{2H}.
    pub fn increments(&self) -> IntegrateResult<Array1<f64>> {
        self.validate()?;
        match self.config.method {
            FbmMethod::Hosking => self.increments_hosking(),
            FbmMethod::DaviesHarte => self.increments_davies_harte(),
        }
    }

    /// Generate a sample path B^H_0, B^H_{dt}, …, B^H_{n·dt} (length `n_steps + 1`).
    ///
    /// The path always starts at 0.
    pub fn sample_path(&self) -> IntegrateResult<Array1<f64>> {
        let xi = self.increments()?;
        let n = xi.len();
        let mut path = Array1::zeros(n + 1);
        let mut cum = 0.0_f64;
        for i in 0..n {
            cum += xi[i];
            path[i + 1] = cum;
        }
        Ok(path)
    }

    /// Theoretical variance of a single increment: dt^{2H}.
    pub fn increment_variance(&self) -> f64 {
        self.config.dt.powf(2.0 * self.config.hurst)
    }

    // -----------------------------------------------------------------------
    // Hosking (exact Cholesky, O(n²))
    // -----------------------------------------------------------------------

    fn increments_hosking(&self) -> IntegrateResult<Array1<f64>> {
        let n = self.config.n_steps;
        let h = self.config.hurst;
        let dt = self.config.dt;

        // Covariance of fractional Gaussian noise (fGn) increments at lag ℓ = |i-j|:
        // Cov[ξ_i, ξ_j] = γ(ℓ) = ½·dt^{2H}·(|ℓ+1|^{2H} − 2|ℓ|^{2H} + |ℓ-1|^{2H})
        let h2 = 2.0 * h;
        let gamma_lag = |lag: usize| -> f64 {
            let l = lag as f64;
            0.5 * dt.powf(h2) * ((l + 1.0).powf(h2) - 2.0 * l.powf(h2) + (l - 1.0).abs().powf(h2))
        };

        // Build covariance matrix C[i,j] = γ(|i-j|)
        let cov_elem = |i: usize, j: usize| -> f64 {
            let lag = if i >= j { i - j } else { j - i };
            gamma_lag(lag)
        };

        // Build lower Cholesky L (n×n) in row-major order stored flat.
        let mut l = vec![0.0_f64; n * n];

        for i in 0..n {
            for j in 0..=i {
                let c_ij = cov_elem(i + 1, j + 1); // 1-indexed times
                let mut sum = c_ij;
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum <= 0.0 {
                        // Numerical rounding: clamp to tiny positive
                        sum = 1e-15;
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    let l_jj = l[j * n + j];
                    if l_jj.abs() < 1e-300 {
                        return Err(IntegrateError::ComputationError(
                            "Cholesky: near-zero diagonal element".to_string(),
                        ));
                    }
                    l[i * n + j] = sum / l_jj;
                }
            }
        }

        // Generate z ~ N(0, I_n)
        let mut rng = Lcg::new(self.config.seed);
        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();

        // xi = L @ z
        let xi_vec: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = 0.0_f64;
                for k in 0..=i {
                    s += l[i * n + k] * z[k];
                }
                s
            })
            .collect();

        Ok(Array1::from_vec(xi_vec))
    }

    // -----------------------------------------------------------------------
    // Davies-Harte (circulant embedding, O(n log n))
    // -----------------------------------------------------------------------

    fn increments_davies_harte(&self) -> IntegrateResult<Array1<f64>> {
        let n = self.config.n_steps;

        // Davies-Harte requires at least n=2 for the circulant of size m=2n≥4.
        // Fall back to Hosking for tiny n.
        if n < 2 {
            return self.increments_hosking();
        }

        let h = self.config.hurst;
        let dt = self.config.dt;

        // Autocovariance of fractional Gaussian noise at lag k:
        // γ(k) = ½·dt^{2H}·(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})
        let gamma = |k: usize| -> f64 {
            let kf = k as f64;
            let h2 = 2.0 * h;
            0.5 * dt.powf(h2)
                * ((kf + 1.0).powf(h2) - 2.0 * kf.powf(h2) + (kf - 1.0).abs().powf(h2))
        };

        // Build the first row of the 2n-circulant (length m = 2n):
        // c = [γ(0), γ(1), …, γ(n-1), γ(n-1), γ(n-2), …, γ(1)]
        // (symmetric / Toeplitz-circulant embedding)
        let m = 2 * n;
        let mut circulant_row: Vec<Complex64> = Vec::with_capacity(m);
        for k in 0..n {
            circulant_row.push(Complex64::new(gamma(k), 0.0));
        }
        for k in (1..n).rev() {
            circulant_row.push(Complex64::new(gamma(k), 0.0));
        }

        // λ_k = Re(FFT(c))   (eigenvalues of the circulant, should all be ≥ 0)
        // scirs2-fft's `fft` returns the forward DFT (unnormalised sum).
        let fft_c = fft(&circulant_row, None).map_err(|e| {
            IntegrateError::ComputationError(format!("Davies-Harte FFT failed: {e}"))
        })?;
        // Clamp small negatives arising from floating-point rounding.
        let lambda: Vec<f64> = fft_c.iter().map(|c| c.re.max(0.0)).collect();

        // ------------------------------------------------------------------
        // Build the Hermitian-symmetric W vector such that IFFT(W) is real.
        //
        // Sampling rule (Wood-Chan / Davies-Harte):
        //   W_0          = sqrt(λ_0) * N(0,1)
        //   W_{m/2}      = sqrt(λ_{m/2}) * N(0,1)
        //   W_k          = sqrt(λ_k / 2) * (N1 + i·N2)  for 1 ≤ k ≤ m/2 − 1
        //   W_{m-k}      = conj(W_k)                     for 1 ≤ k ≤ m/2 − 1
        //
        // Then ξ = Re(IFFT_unnorm(W)) / sqrt(m)  (first n elements)
        //
        // scirs2-fft's `ifft` applies 1/m normalisation, so:
        //   ξ = Re(ifft(W)) * sqrt(m)
        // ------------------------------------------------------------------
        let mut rng = Lcg::new(self.config.seed);
        let mut w = vec![Complex64::new(0.0, 0.0); m];
        let half = m / 2;

        // k = 0
        w[0] = Complex64::new(lambda[0].sqrt() * rng.next_normal(), 0.0);
        // k = m/2
        w[half] = Complex64::new(lambda[half].sqrt() * rng.next_normal(), 0.0);
        // k = 1 .. half-1
        for k in 1..half {
            let sigma = (lambda[k] / 2.0).sqrt();
            let re = sigma * rng.next_normal();
            let im = sigma * rng.next_normal();
            w[k] = Complex64::new(re, im);
            w[m - k] = Complex64::new(re, -im); // conjugate
        }

        // ξ = Re(ifft(W)) * sqrt(m)
        let z_complex = ifft(&w, None).map_err(|e| {
            IntegrateError::ComputationError(format!("Davies-Harte IFFT failed: {e}"))
        })?;

        let sqrt_m = (m as f64).sqrt();
        let xi_vec: Vec<f64> = z_complex[..n]
            .iter()
            .map(|c| c.re * sqrt_m)
            .collect();

        Ok(Array1::from_vec(xi_vec))
    }

    // -----------------------------------------------------------------------
    // R/S Hurst estimation
    // -----------------------------------------------------------------------

    /// Estimate the Hurst exponent from a realised fBm path using the R/S statistic.
    ///
    /// The function first-differences the path to obtain the fractional Gaussian
    /// noise increments, then applies the rescaled-range (R/S) method over
    /// logarithmically-spaced block sizes to estimate H via a log-log regression.
    ///
    /// # Notes
    ///
    /// R/S is a biased estimator for small samples.  Pass paths of length ≥ 512
    /// for reliable results.
    pub fn estimate_hurst(path: &Array1<f64>) -> f64 {
        let n_path = path.len();
        if n_path < 21 {
            return 0.5;
        }
        // First-difference to get fGn increments (R/S is defined on these)
        let increments: Vec<f64> = (1..n_path).map(|i| path[i] - path[i - 1]).collect();
        let n = increments.len();
        if n < 20 {
            return 0.5;
        }

        // Block sizes: powers of 2 from 8 up to n/2.
        let mut block_sizes: Vec<usize> = Vec::new();
        let mut bs = 8_usize;
        while bs <= n / 2 {
            block_sizes.push(bs);
            bs *= 2;
        }
        if block_sizes.is_empty() {
            block_sizes.push(n / 2);
        }

        let mut log_rs_vals: Vec<f64> = Vec::with_capacity(block_sizes.len());
        let mut log_n_vals: Vec<f64> = Vec::with_capacity(block_sizes.len());

        for &blk in &block_sizes {
            let n_blocks = n / blk;
            if n_blocks == 0 {
                continue;
            }
            let mut rs_sum = 0.0_f64;
            let mut count = 0_usize;
            for b in 0..n_blocks {
                let start = b * blk;
                let end = start + blk;
                let slice = &increments[start..end];
                let mean = slice.iter().sum::<f64>() / (blk as f64);
                // Cumulative deviation (rescaled range statistic)
                let mut cum = 0.0_f64;
                let mut min_cum = 0.0_f64;
                let mut max_cum = 0.0_f64;
                let mut sum_sq = 0.0_f64;
                for &v in slice.iter() {
                    let dev = v - mean;
                    cum += dev;
                    sum_sq += dev * dev;
                    if cum < min_cum {
                        min_cum = cum;
                    }
                    if cum > max_cum {
                        max_cum = cum;
                    }
                }
                let range = max_cum - min_cum;
                let std_dev = (sum_sq / blk as f64).sqrt();
                if std_dev > 1e-300 {
                    rs_sum += range / std_dev;
                    count += 1;
                }
            }
            if count > 0 {
                let rs_avg = rs_sum / count as f64;
                if rs_avg > 0.0 {
                    log_rs_vals.push(rs_avg.ln());
                    log_n_vals.push((blk as f64).ln());
                }
            }
        }

        if log_rs_vals.len() < 2 {
            return 0.5;
        }

        // OLS slope ≈ H
        let k = log_rs_vals.len() as f64;
        let sum_x: f64 = log_n_vals.iter().sum();
        let sum_y: f64 = log_rs_vals.iter().sum();
        let sum_xx: f64 = log_n_vals.iter().map(|&x| x * x).sum();
        let sum_xy: f64 = log_n_vals.iter().zip(&log_rs_vals).map(|(&x, &y)| x * y).sum();
        let denom = k * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-300 {
            return 0.5;
        }
        let slope = (k * sum_xy - sum_x * sum_y) / denom;
        slope.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> FbmConfig {
        FbmConfig::default()
    }

    fn hosking_cfg(h: f64, n: usize) -> FbmConfig {
        FbmConfig {
            hurst: h,
            n_steps: n,
            dt: 1.0 / n as f64,
            seed: 1234,
            method: FbmMethod::Hosking,
        }
    }

    fn dh_cfg(h: f64, n: usize) -> FbmConfig {
        FbmConfig {
            hurst: h,
            n_steps: n,
            dt: 1.0 / n as f64,
            seed: 5678,
            method: FbmMethod::DaviesHarte,
        }
    }

    #[test]
    fn test_default_config() {
        let fbm = FractionalBrownianMotion::new(default_cfg());
        let path = fbm.sample_path().expect("sample_path should succeed");
        assert_eq!(path.len(), 257); // n_steps + 1
    }

    #[test]
    fn test_hosking_path_starts_at_zero() {
        let fbm = FractionalBrownianMotion::new(hosking_cfg(0.7, 64));
        let path = fbm.sample_path().expect("sample_path should succeed");
        assert!(
            path[0].abs() < 1e-15,
            "Hosking path[0] = {}, expected 0",
            path[0]
        );
    }

    #[test]
    fn test_davies_harte_path_starts_at_zero() {
        let fbm = FractionalBrownianMotion::new(dh_cfg(0.7, 128));
        let path = fbm.sample_path().expect("sample_path should succeed");
        assert!(
            path[0].abs() < 1e-15,
            "Davies-Harte path[0] = {}, expected 0",
            path[0]
        );
    }

    #[test]
    fn test_path_length() {
        for &n in &[32_usize, 64, 128, 256] {
            let fbm = FractionalBrownianMotion::new(dh_cfg(0.6, n));
            let path = fbm.sample_path().expect("sample_path should succeed");
            assert_eq!(path.len(), n + 1, "Expected n+1 = {} path points", n + 1);
        }
    }

    #[test]
    fn test_increment_variance() {
        let cfg = FbmConfig {
            hurst: 0.7,
            n_steps: 512,
            dt: 1.0 / 512.0,
            seed: 99,
            method: FbmMethod::DaviesHarte,
        };
        let expected_var = cfg.dt.powf(2.0 * cfg.hurst);
        let fbm = FractionalBrownianMotion::new(cfg);
        let var = fbm.increment_variance();
        let rel = (var - expected_var).abs() / expected_var;
        assert!(rel < 1e-12, "increment_variance relative error = {}", rel);
    }

    #[test]
    fn test_hosking_vs_davies_harte_statistical_properties() {
        // Both methods should give increments with the same approximate mean and variance.
        let n = 256;
        let dt = 1.0 / n as f64;
        let h = 0.7;

        let h_cfg = FbmConfig {
            hurst: h, n_steps: n, dt, seed: 42, method: FbmMethod::Hosking,
        };
        let dh_cfg_ = FbmConfig {
            hurst: h, n_steps: n, dt, seed: 42, method: FbmMethod::DaviesHarte,
        };

        let xi_h = FractionalBrownianMotion::new(h_cfg).increments().expect("hosking increments");
        let xi_dh = FractionalBrownianMotion::new(dh_cfg_).increments().expect("dh increments");

        // Mean ~ 0 for both
        let mean_h: f64 = xi_h.sum() / n as f64;
        let mean_dh: f64 = xi_dh.sum() / n as f64;
        let expected_var = dt.powf(2.0 * h);

        // Variance ~ dt^{2H}
        let var_h: f64 = xi_h.iter().map(|&v| v * v).sum::<f64>() / n as f64;
        let var_dh: f64 = xi_dh.iter().map(|&v| v * v).sum::<f64>() / n as f64;

        assert!(
            mean_h.abs() < 5.0 * expected_var.sqrt(),
            "Hosking mean {} too large",
            mean_h
        );
        assert!(
            mean_dh.abs() < 5.0 * expected_var.sqrt(),
            "DH mean {} too large",
            mean_dh
        );
        // Variance should be within a factor of 3 of expected
        assert!(var_h < 3.0 * expected_var, "Hosking variance {} >> {}", var_h, expected_var);
        assert!(var_dh < 3.0 * expected_var, "DH variance {} >> {}", var_dh, expected_var);
    }

    #[test]
    fn test_h_half_linear_variance() {
        // H = 0.5 → standard BM → path variance grows linearly: Var[B_t] = t.
        let n = 512;
        let dt = 1.0 / n as f64;
        let cfg = FbmConfig {
            hurst: 0.5, n_steps: n, dt, seed: 7, method: FbmMethod::DaviesHarte,
        };
        let fbm = FractionalBrownianMotion::new(cfg);

        // Generate many paths and check average squared value at t = T/2.
        let n_paths = 200;
        let mid = n / 2;
        let mut sum_sq = 0.0_f64;
        for seed in 0..n_paths_u64(n_paths) {
            let cfg2 = FbmConfig {
                hurst: 0.5, n_steps: n, dt, seed, method: FbmMethod::DaviesHarte,
            };
            let path = FractionalBrownianMotion::new(cfg2)
                .sample_path()
                .expect("sample_path");
            sum_sq += path[mid] * path[mid];
        }
        let sample_var = sum_sq / n_paths as f64;
        let theoretical = (mid as f64) * dt; // = 0.5
        let rel = (sample_var - theoretical).abs() / theoretical;
        assert!(
            rel < 0.3,
            "H=0.5 variance at t=0.5: sample={:.4}, theory={:.4}, rel={:.4}",
            sample_var, theoretical, rel
        );
        // Verify generator still gives path
        let _ = fbm.sample_path().expect("sample_path");
    }

    #[test]
    fn test_estimate_hurst_within_tolerance() {
        // Generate many paths with H=0.7, estimate H from each, average should be ≈ 0.7.
        let h_true = 0.7_f64;
        let n = 512;
        let dt = 1.0 / n as f64;
        let mut h_estimates = Vec::new();
        for seed in 0..20_u64 {
            let cfg = FbmConfig {
                hurst: h_true, n_steps: n, dt, seed, method: FbmMethod::DaviesHarte,
            };
            let path = FractionalBrownianMotion::new(cfg).sample_path().expect("path");
            h_estimates.push(FractionalBrownianMotion::estimate_hurst(&path));
        }
        let mean_h: f64 = h_estimates.iter().sum::<f64>() / h_estimates.len() as f64;
        assert!(
            (mean_h - h_true).abs() < 0.2,
            "Hurst estimate {:.3} not within 0.2 of true {:.1}",
            mean_h, h_true
        );
    }

    #[test]
    fn test_long_range_dependence_h_gt_half() {
        // H > 0.5 → lag-1 autocorrelation of increments > 0.
        let n = 512;
        let dt = 1.0 / n as f64;
        let cfg = FbmConfig {
            hurst: 0.8, n_steps: n, dt, seed: 13, method: FbmMethod::DaviesHarte,
        };
        let xi = FractionalBrownianMotion::new(cfg).increments().expect("increments");
        let lag1_acf = compute_lag1_acf(&xi);
        assert!(
            lag1_acf > 0.0,
            "H=0.8 lag-1 acf should be positive, got {}",
            lag1_acf
        );
    }

    #[test]
    fn test_anti_persistent_h_lt_half() {
        // H < 0.5 → lag-1 autocorrelation of increments < 0.
        let n = 512;
        let dt = 1.0 / n as f64;
        let cfg = FbmConfig {
            hurst: 0.2, n_steps: n, dt, seed: 17, method: FbmMethod::DaviesHarte,
        };
        let xi = FractionalBrownianMotion::new(cfg).increments().expect("increments");
        let lag1_acf = compute_lag1_acf(&xi);
        assert!(
            lag1_acf < 0.0,
            "H=0.2 lag-1 acf should be negative, got {}",
            lag1_acf
        );
    }

    #[test]
    fn test_invalid_h_zero() {
        let cfg = FbmConfig { hurst: 0.0, ..Default::default() };
        let fbm = FractionalBrownianMotion::new(cfg);
        assert!(fbm.sample_path().is_err(), "H=0 should be invalid");
    }

    #[test]
    fn test_invalid_h_one() {
        let cfg = FbmConfig { hurst: 1.0, ..Default::default() };
        let fbm = FractionalBrownianMotion::new(cfg);
        assert!(fbm.sample_path().is_err(), "H=1 should be invalid");
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn compute_lag1_acf(xi: &Array1<f64>) -> f64 {
        let n = xi.len();
        if n < 2 {
            return 0.0;
        }
        let mean = xi.sum() / n as f64;
        let var: f64 = xi.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-300 {
            return 0.0;
        }
        let cov: f64 = xi
            .iter()
            .zip(xi.iter().skip(1))
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / (n - 1) as f64;
        cov / var
    }

    fn n_paths_u64(n: usize) -> u64 {
        n as u64
    }
}
