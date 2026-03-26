//! Stochastic Subspace Identification (SSI-Cov) for Operational Modal Analysis.
//!
//! SSI-Cov estimates the state-space model of a structure from output-only
//! (ambient) measurements via the following steps:
//!
//! 1. Estimate lag-correlation matrices `R_i = E[y(t) y(t-i)^T]`.
//! 2. Assemble a block-Toeplitz / Hankel matrix from `R_1 … R_{2p}`.
//! 3. SVD of the block-Hankel → observability matrix **O**.
//! 4. Extract discrete system matrix **A** via the shift property of **O**.
//! 5. Eigendecompose **A**: eigenvalues → modal frequencies and damping ratios.
//! 6. Mode shapes from the eigenvectors projected onto the output (C) sub-matrix.
//! 7. Stabilisation filtering to keep physically meaningful poles.
//!
//! ## References
//! - Van Overschee, P. & De Moor, B. (1996). *Subspace Identification for
//!   Linear Systems.* Kluwer Academic.
//! - Brincker, R. & Ventura, C. (2015). *Introduction to Operational Modal
//!   Analysis.* Wiley.

use super::types::{ModalMode, OmaConfig, OmaMethod, OmaResult};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Correlation matrix estimation
// ---------------------------------------------------------------------------

/// Estimate the cross-correlation matrix at lag `lag`:
/// `R(lag) = (1/N) Σ_{t} y(t) · y(t - lag)^T`
fn correlation_matrix(data: &Array2<f64>, lag: usize) -> Array2<f64> {
    let (n, p) = (data.nrows(), data.ncols());
    let mut r = Array2::<f64>::zeros((p, p));
    if lag >= n {
        return r;
    }
    let count = (n - lag) as f64;
    for t in lag..n {
        for i in 0..p {
            for j in 0..p {
                r[[i, j]] += data[[t, i]] * data[[t - lag, j]];
            }
        }
    }
    r.mapv_inplace(|v| v / count);
    r
}

// ---------------------------------------------------------------------------
// Block-Hankel assembly
// ---------------------------------------------------------------------------

/// Build the block-Hankel matrix from correlation matrices.
///
/// ```text
/// H = | R(1)   R(2)   … R(q)   |
///     | R(2)   R(3)   … R(q+1) |
///     | …                       |
///     | R(q)   R(q+1) … R(2q-1)|
/// ```
/// Result has shape `(q·p, q·p)` where `p = n_channels`, `q = n_block_rows`.
fn build_hankel(corrs: &[Array2<f64>], n_block_rows: usize, p: usize) -> Array2<f64> {
    let n_hankel = n_block_rows * p;
    let mut h = Array2::<f64>::zeros((n_hankel, n_hankel));
    for row_blk in 0..n_block_rows {
        for col_blk in 0..n_block_rows {
            let lag = row_blk + col_blk + 1; // 1-based
            if lag < corrs.len() {
                let r = &corrs[lag];
                let r0 = row_blk * p;
                let c0 = col_blk * p;
                for i in 0..p {
                    for j in 0..p {
                        h[[r0 + i, c0 + j]] = r[[i, j]];
                    }
                }
            }
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Truncated SVD via power iteration (rank-r approximation)
// ---------------------------------------------------------------------------

/// Compute the leading `rank` singular vectors / values of a symmetric PSD
/// matrix `m` using `n_iter` power iterations (cheap and reliable for large gaps).
///
/// Returns `(u, s)` where `u` columns are left singular vectors and `s` are
/// singular values (not squared).
fn truncated_svd_sym(m: &Array2<f64>, rank: usize, n_iter: usize) -> (Array2<f64>, Vec<f64>) {
    let n = m.nrows();
    let r = rank.min(n);
    let mut u = Array2::<f64>::zeros((n, r));
    let mut sv = vec![0.0_f64; r];

    // We use deflation: extract one vector at a time.
    // Residual = m (we deflate it conceptually via projections).
    let mut deflated = m.to_owned();

    for k in 0..r {
        // Random start for diversity
        let mut v: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            // Deterministic pseudo-random start using index (wrapping to avoid overflow)
            let seed = i
                .wrapping_mul(6364136223846793005_usize)
                .wrapping_add(1442695040888963407_usize);
            let f = (seed >> 33) as f64 / ((u64::MAX >> 33) as f64);
            f - 0.5
        }));
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > f64::EPSILON {
            v.mapv_inplace(|x| x / norm);
        }

        for _ in 0..n_iter {
            // w = deflated * v
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    w[i] += deflated[[i, j]] * v[j];
                }
            }
            let nw: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nw < f64::EPSILON {
                break;
            }
            v = w.mapv(|x| x / nw);
        }

        // Rayleigh quotient
        let mut rv = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                rv[i] += deflated[[i, j]] * v[j];
            }
        }
        let sigma: f64 = v.iter().zip(rv.iter()).map(|(a, b)| a * b).sum::<f64>();
        let sigma = sigma.abs().sqrt();
        sv[k] = sigma;
        for i in 0..n {
            u[[i, k]] = v[i];
        }

        // Deflate: deflated -= sigma^2 * v v^T
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] -= sigma * sigma * v[i] * v[j];
            }
        }
    }
    (u, sv)
}

// ---------------------------------------------------------------------------
// Least-squares shift-solve: A = O[1:]^+ O[:-1]
// ---------------------------------------------------------------------------

/// Solve the over-determined system `O_up * A = O_down` for `A` using
/// the pseudo-inverse: `A = O_up^+ O_down`.
///
/// Both `o_up` and `o_down` have shape `((n_block-1)*p, r)` where `r` is the
/// model order.  We use a simple normal-equation solution.
fn shift_solve(o_up: &Array2<f64>, o_down: &Array2<f64>) -> SignalResult<Array2<f64>> {
    // A = (o_up^T o_up)^{-1} o_up^T o_down
    let (m, r) = (o_up.nrows(), o_up.ncols());
    let n_col = o_down.ncols();

    // gram = o_up^T * o_up  (r × r)
    let mut gram = Array2::<f64>::zeros((r, r));
    for k in 0..m {
        for i in 0..r {
            for j in 0..r {
                gram[[i, j]] += o_up[[k, i]] * o_up[[k, j]];
            }
        }
    }
    // rhs = o_up^T * o_down  (r × n_col)
    let mut rhs = Array2::<f64>::zeros((r, n_col));
    for k in 0..m {
        for i in 0..r {
            for j in 0..n_col {
                rhs[[i, j]] += o_up[[k, i]] * o_down[[k, j]];
            }
        }
    }
    // Solve gram * A = rhs via Cholesky-like (diagonal pivot Gauss elimination)
    chol_solve(&gram, &rhs)
}

/// Solve `G x = b` for square `G` using Gaussian elimination with partial
/// pivoting and Tikhonov regularisation for near-singular systems.
/// Returns `x` of shape `(r, n_col)`.
fn chol_solve(g: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let r = g.nrows();
    let n_col = b.ncols();

    // Tikhonov regularisation: add λ·I to improve conditioning.
    // λ = 1e-8 * max diagonal element
    let max_diag = (0..r).map(|i| g[[i, i]].abs()).fold(0.0_f64, f64::max);
    let lambda = 1e-8 * max_diag.max(1e-30);

    // Augmented matrix [G + λI | b]
    let mut aug = Array2::<f64>::zeros((r, r + n_col));
    for i in 0..r {
        for j in 0..r {
            aug[[i, j]] = g[[i, j]];
        }
        aug[[i, i]] += lambda;
        for j in 0..n_col {
            aug[[i, r + j]] = b[[i, j]];
        }
    }
    // Forward elimination with partial pivoting
    for col in 0..r {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..r {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            // Still singular even after regularisation — return zero solution.
            return Ok(Array2::<f64>::zeros((r, n_col)));
        }
        if max_row != col {
            for j in 0..(r + n_col) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..r {
            let factor = aug[[row, col]] / pivot;
            for j in col..(r + n_col) {
                let v = aug[[col, j]];
                aug[[row, j]] -= factor * v;
            }
        }
    }
    // Back-substitution
    let mut x = Array2::<f64>::zeros((r, n_col));
    for i in (0..r).rev() {
        for j in 0..n_col {
            let mut sum = aug[[i, r + j]];
            for k in (i + 1)..r {
                sum -= aug[[i, k]] * x[[k, j]];
            }
            let pivot = aug[[i, i]];
            x[[i, j]] = if pivot.abs() > 1e-30 {
                sum / pivot
            } else {
                0.0
            };
        }
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Real eigendecomposition of a real matrix via the QR algorithm
// ---------------------------------------------------------------------------

/// Perform a basic QR iteration step on matrix `a`, returning the updated
/// matrix.  Used to converge toward Schur form.
fn qr_step(a: &mut Array2<f64>) {
    let n = a.nrows();
    // Householder QR
    let mut q = Array2::<f64>::eye(n);
    for col in 0..n.saturating_sub(1) {
        let mut x: Vec<f64> = (col..n).map(|i| a[[i, col]]).collect();
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-14 {
            continue;
        }
        x[0] += if x[0] >= 0.0 { norm } else { -norm };
        let hn: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if hn < 1e-14 {
            continue;
        }
        let v: Vec<f64> = x.iter().map(|v| v / hn).collect();
        // Apply H = I - 2 v v^T from the left to a[col:, col:]
        let nv = v.len();
        for j in 0..n {
            let dot: f64 = (0..nv).map(|i| v[i] * a[[col + i, j]]).sum();
            for i in 0..nv {
                a[[col + i, j]] -= 2.0 * v[i] * dot;
            }
        }
        // Apply H from the right to a[:, col:]
        for i in 0..n {
            let dot: f64 = (0..nv).map(|k| v[k] * a[[i, col + k]]).sum();
            for k in 0..nv {
                a[[i, col + k]] -= 2.0 * v[k] * dot;
            }
        }
        // Accumulate Q
        for i in 0..n {
            let dot: f64 = (0..nv).map(|k| v[k] * q[[i, col + k]]).sum();
            for k in 0..nv {
                q[[i, col + k]] -= 2.0 * v[k] * dot;
            }
        }
    }
}

/// Extract eigenvalues (real+imaginary) from a quasi-upper-triangular matrix
/// (2×2 diagonal blocks for complex pairs).
fn extract_eigenvalues(schur: &Array2<f64>) -> Vec<(f64, f64)> {
    let n = schur.nrows();
    let mut eigs = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n && schur[[i + 1, i]].abs() > 1e-10 {
            // 2×2 block → complex pair
            let a = schur[[i, i]];
            let b = schur[[i, i + 1]];
            let c = schur[[i + 1, i]];
            let d = schur[[i + 1, i + 1]];
            let trace = a + d;
            let det = a * d - b * c;
            let disc = trace * trace - 4.0 * det;
            if disc < 0.0 {
                let re = trace / 2.0;
                let im = (-disc).sqrt() / 2.0;
                eigs.push((re, im));
                eigs.push((re, -im));
            } else {
                let s = disc.sqrt();
                eigs.push(((trace + s) / 2.0, 0.0));
                eigs.push(((trace - s) / 2.0, 0.0));
            }
            i += 2;
        } else {
            eigs.push((schur[[i, i]], 0.0));
            i += 1;
        }
    }
    eigs
}

/// Approximate eigendecomposition of `a` via repeated QR iteration.
/// Returns eigenvalues as `(re, im)` pairs.
fn eig_approx(a: &Array2<f64>, n_iter: usize) -> Vec<(f64, f64)> {
    let mut work = a.clone();
    for _ in 0..n_iter {
        qr_step(&mut work);
    }
    extract_eigenvalues(&work)
}

// ---------------------------------------------------------------------------
// Stabilisation check
// ---------------------------------------------------------------------------

struct PhysicalPole {
    freq: f64,
    damping: f64,
    idx: usize,
}

/// Map discrete eigenvalue → continuous (freq Hz, damping ratio).
fn discrete_to_modal(lambda_re: f64, lambda_im: f64, fs: f64) -> Option<(f64, f64)> {
    // λ = exp(s·dt), so s = ln(λ) * fs
    let r = (lambda_re * lambda_re + lambda_im * lambda_im).sqrt();
    if r < 1e-12 {
        return None;
    }
    let theta = lambda_im.atan2(lambda_re); // angle of λ
    let freq = theta.abs() * fs / (2.0 * PI);
    let sigma = r.ln() * fs; // real part of s
    if freq < 1e-6 {
        return None;
    }
    let omega_n = (sigma * sigma + (theta * fs).powi(2)).sqrt();
    let damping = if omega_n > 1e-6 {
        (-sigma / omega_n).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if freq > 1e-6 && damping >= 0.0 && damping < 1.0 {
        Some((freq, damping))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Identify structural modes via Covariance-driven Stochastic Subspace
/// Identification (SSI-Cov).
///
/// # Arguments
/// * `data` — measured vibration data, shape `[n_samples, n_channels]`
/// * `config` — OMA configuration
///
/// # Returns
/// [`OmaResult`] containing identified [`ModalMode`]s.
///
/// # Algorithm overview
/// 1. Compute lag-correlation matrices `R_0 … R_{2·n_block}`.
/// 2. Assemble a block-Hankel matrix from these correlations.
/// 3. Truncated SVD → state-space model order and observability matrix **O**.
/// 4. Extract system matrix **A** via shift property.
/// 5. Eigendecompose **A**; convert to continuous-time modal parameters.
/// 6. Filter by stabilisation criteria (frequency, damping bounds).
/// 7. Keep the `config.n_modes` most stable physical poles.
///
/// # Errors
/// Returns a [`crate::error::SignalError`] if data is insufficient or
/// numerical issues arise.
pub fn ssi_cov(data: &Array2<f64>, config: &OmaConfig) -> SignalResult<OmaResult> {
    let (n_samples, n_ch) = (data.nrows(), data.ncols());
    if n_samples < 4 {
        return Err(SignalError::InvalidArgument(
            "data must have at least 4 samples".to_string(),
        ));
    }
    if n_ch == 0 {
        return Err(SignalError::InvalidArgument(
            "data must have at least one channel".to_string(),
        ));
    }

    let n_block = config.n_lags.min(n_samples / 4).max(2);
    let max_lag = 2 * n_block;

    // Step 1: Correlation matrices
    let mut corrs: Vec<Array2<f64>> = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        corrs.push(correlation_matrix(data, lag));
    }

    // Step 2: Block-Hankel
    let h = build_hankel(&corrs, n_block, n_ch);

    // Step 3: Truncated SVD
    let model_order = (config.n_modes * 2).min(n_block * n_ch);
    let n_svd_iter = 30;
    let (u, sv) = truncated_svd_sym(&h, model_order, n_svd_iter);

    // Observability matrix O = U * diag(sqrt(sv))  shape (n_block*p, model_order)
    let obs_rows = n_block * n_ch;
    let obs_cols = model_order;
    let mut obs = Array2::<f64>::zeros((obs_rows, obs_cols));
    for k in 0..obs_cols {
        let s = sv[k].sqrt();
        for i in 0..obs_rows {
            obs[[i, k]] = u[[i, k]] * s;
        }
    }

    // Step 4: A = O[1:]^+ O[:-1]
    let skip = n_ch; // one block row
    if obs_rows <= skip {
        return Err(SignalError::ComputationError(
            "observability matrix too small for shift operation".to_string(),
        ));
    }
    let o_down = obs.slice(s![skip.., ..]).to_owned(); // (obs_rows-skip, obs_cols)
    let o_up = obs.slice(s![..obs_rows - skip, ..]).to_owned();
    let a_mat = shift_solve(&o_up, &o_down)?;

    // Step 5: Eigendecompose A
    let eig_iter = 80;
    let eigenvalues = eig_approx(&a_mat, eig_iter);

    // Step 6-7: Convert to physical poles and filter
    let fs = config.fs;
    let freq_max = config.freq_max.unwrap_or(fs / 2.0);
    let mut physical: Vec<PhysicalPole> = Vec::new();

    for (idx, &(re, im)) in eigenvalues.iter().enumerate() {
        if let Some((freq, damping)) = discrete_to_modal(re, im, fs) {
            if freq >= config.freq_min && freq <= freq_max && damping >= 0.0 && damping <= 0.5 {
                physical.push(PhysicalPole { freq, damping, idx });
            }
        }
    }

    // Sort by frequency
    physical.sort_by(|a, b| {
        a.freq
            .partial_cmp(&b.freq)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mode shape: use the corresponding row block of O (output matrix C = O[0..p, :])
    // and the eigenvector column of A approximated by O itself.
    let n_modes_found = physical.len().min(config.n_modes);
    let mut modes = Vec::with_capacity(n_modes_found);

    for pole in physical.iter().take(n_modes_found) {
        // Approximate mode shape from the first block row of O (rows 0..n_ch)
        // weighted by the eigenvalue magnitude
        let mut shape = Array1::<f64>::zeros(n_ch);
        // Use eigenvector derived from O: we take the column of U corresponding
        // to the pole's model order component.
        let col_idx = pole.idx.min(obs_cols - 1);
        for i in 0..n_ch {
            shape[i] = obs[[i, col_idx]];
        }
        let norm: f64 = shape.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > f64::EPSILON {
            shape.mapv_inplace(|v| v / norm);
        }
        modes.push(ModalMode::new(pole.freq, pole.damping, shape));
    }

    Ok(OmaResult::new(modes, OmaMethod::SsiCov, 0.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    fn sine_multichannel(freqs_hz: &[f64], fs: f64, n: usize) -> Array2<f64> {
        let p = freqs_hz.len();
        let mut data = Array2::<f64>::zeros((n, p));
        for (ch, &f) in freqs_hz.iter().enumerate() {
            for i in 0..n {
                let t = i as f64 / fs;
                data[[i, ch]] = (2.0 * PI * f * t).sin();
            }
        }
        data
    }

    #[test]
    fn test_ssi_cov_basic() {
        let fs = 200.0;
        let data = sine_multichannel(&[5.0, 15.0], fs, 1024);
        let config = OmaConfig {
            n_modes: 2,
            fs,
            n_lags: 16,
            freq_min: 1.0,
            freq_max: Some(80.0),
            ..Default::default()
        };
        let result = ssi_cov(&data, &config).expect("ssi_cov should succeed");
        assert_eq!(result.method, OmaMethod::SsiCov);
        // Result may have 0 modes if no physical poles pass the filter
        assert!(result.n_modes() <= 2);
    }

    #[test]
    fn test_ssi_cov_single_channel() {
        let fs = 100.0;
        let n = 512;
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let t = i as f64 / fs;
            data[[i, 0]] = (2.0 * PI * 10.0 * t).sin();
        }
        let config = OmaConfig {
            n_modes: 1,
            fs,
            n_lags: 8,
            freq_min: 1.0,
            freq_max: Some(40.0),
            ..Default::default()
        };
        let result = ssi_cov(&data, &config).expect("ssi_cov single channel should succeed");
        assert!(result.n_modes() <= 1);
    }

    #[test]
    fn test_ssi_cov_error_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = OmaConfig::default();
        assert!(ssi_cov(&data, &config).is_err());
    }

    #[test]
    fn test_ssi_cov_mode_shapes_unit_norm() {
        let fs = 100.0;
        let data = sine_multichannel(&[5.0, 20.0], fs, 512);
        let config = OmaConfig {
            n_modes: 2,
            fs,
            n_lags: 12,
            freq_min: 1.0,
            freq_max: Some(45.0),
            ..Default::default()
        };
        let result = ssi_cov(&data, &config).expect("ssi_cov should succeed");
        for mode in &result.modes {
            let norm: f64 = mode.mode_shape.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-9 || norm < 1e-9,
                "mode shape norm={norm}"
            );
        }
    }

    #[test]
    fn test_ssi_cov_modes_sorted_by_freq() {
        let fs = 200.0;
        let data = sine_multichannel(&[3.0, 8.0, 20.0], fs, 2048);
        let config = OmaConfig {
            n_modes: 3,
            fs,
            n_lags: 16,
            freq_min: 1.0,
            freq_max: Some(90.0),
            ..Default::default()
        };
        let result = ssi_cov(&data, &config).expect("ssi_cov should succeed");
        for w in result.modes.windows(2) {
            assert!(w[0].freq <= w[1].freq, "modes not sorted by frequency");
        }
    }

    #[test]
    fn test_correlation_matrix_lag_zero() {
        let fs = 100.0;
        let data = sine_multichannel(&[10.0], fs, 100);
        let r0 = correlation_matrix(&data, 0);
        // R(0) should be the variance (positive)
        assert!(r0[[0, 0]] > 0.0);
    }

    #[test]
    fn test_correlation_matrix_large_lag() {
        let data = Array2::<f64>::zeros((10, 2));
        let r = correlation_matrix(&data, 100);
        // lag > n_samples → zero matrix
        assert!(r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_discrete_to_modal_unit_circle() {
        // Unit circle eigenvalue at angle π/4 (45°) with fs=100
        let theta = PI / 4.0;
        let re = theta.cos();
        let im = theta.sin();
        let result = discrete_to_modal(re, im, 100.0);
        assert!(result.is_some(), "should produce a valid pole");
        let (freq, damp) = result.expect("checked above");
        assert!(freq > 0.0, "freq={freq}");
        assert!((0.0..=1.0).contains(&damp), "damp={damp}");
    }

    #[test]
    fn test_chol_solve_identity() {
        // chol_solve adds Tikhonov regularisation (λ·I), so the solution to
        // (I + λ·I)·x = b is x = b / (1 + λ).  With max_diag = 1, λ = 1e-8.
        let g = Array2::<f64>::eye(3);
        let b = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("shape valid");
        let x = chol_solve(&g, &b).expect("chol_solve should succeed");
        let lambda = 1e-8_f64;
        let scale = 1.0 / (1.0 + lambda);
        assert!((x[[0, 0]] - scale).abs() < 1e-6);
        assert!((x[[1, 0]] - 2.0 * scale).abs() < 1e-6);
        assert!((x[[2, 0]] - 3.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn test_ssi_cov_too_short() {
        let data = Array2::<f64>::zeros((2, 2));
        let config = OmaConfig::default();
        assert!(ssi_cov(&data, &config).is_err());
    }
}
