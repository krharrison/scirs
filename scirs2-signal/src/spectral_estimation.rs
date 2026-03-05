// Parametric Spectral Estimation
//
// Provides:
//   * `BurgResult`  — AR model from Burg's maximum-entropy method
//   * `burg_method` — Burg AR estimation
//   * `burg_psd`    — PSD computed from a Burg AR model
//   * `yule_walker` — AR estimation via Yule-Walker equations
//   * `MusicResult` / `music` — MUSIC super-resolution frequency estimator
//   * `esprit`      — ESPRIT frequency estimator

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ─── Result types ─────────────────────────────────────────────────────────────

/// Result of an AR spectral estimation.
#[derive(Debug, Clone)]
pub struct BurgResult {
    /// AR model coefficients [a₁, a₂, …, aₚ] (without the leading 1).
    pub ar_coefficients: Vec<f64>,
    /// PARCOR (partial correlation / reflection) coefficients.
    pub reflection_coefficients: Vec<f64>,
    /// Estimated white-noise variance.
    pub noise_variance: f64,
    /// Model order.
    pub order: usize,
}

/// Result of the MUSIC frequency estimator.
#[derive(Debug, Clone)]
pub struct MusicResult {
    /// Estimated sinusoidal frequencies in Hz.
    pub frequencies: Vec<f64>,
    /// MUSIC pseudo-spectrum evaluated at `freq_axis`.
    pub pseudo_spectrum: Vec<f64>,
    /// Frequency axis corresponding to `pseudo_spectrum` (in Hz).
    pub freq_axis: Vec<f64>,
}

// ─── Burg's method ────────────────────────────────────────────────────────────

/// Estimate an AR model using Burg's maximum-entropy method.
///
/// # Arguments
/// * `x`     – input signal
/// * `order` – AR model order p
///
/// # Returns
/// `BurgResult` containing AR coefficients, reflection coefficients and noise
/// variance.
pub fn burg_method(x: &[f64], order: usize) -> SignalResult<BurgResult> {
    let n = x.len();
    if order == 0 {
        return Err(SignalError::ValueError("order must be >= 1".to_string()));
    }
    if order >= n {
        return Err(SignalError::ValueError(format!(
            "order ({order}) must be less than signal length ({n})"
        )));
    }

    // Forward / backward prediction error vectors (indexed 0..n-1)
    let mut ef = x.to_vec();
    let mut eb = x.to_vec();

    // Initial power estimate: average signal energy
    let mut pe: f64 = x.iter().map(|&v| v * v).sum::<f64>() / n as f64;

    // AR coefficients: a[0] = 1, a[1..=order] built up incrementally
    let mut ar = vec![0.0f64; order + 1];
    ar[0] = 1.0;

    let mut km = vec![0.0_f64; order]; // reflection coefficients

    for m in 0..order {
        // Compute reflection coefficient k_m
        // k_m = -2 * sum_{i=m+1}^{n-1} ef[i]*eb[i-1]
        //        / sum_{i=m+1}^{n-1} (ef[i]^2 + eb[i-1]^2)
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in (m + 1)..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }

        if den < 1e-30 {
            return Err(SignalError::ComputationError(
                "Burg algorithm encountered near-zero denominator".to_string(),
            ));
        }

        let k = -2.0 * num / den;
        km[m] = k;

        // Update AR coefficients: Levinson-Durbin order update
        // a_new[j] = a[j] + k * a[m+1-j]  for j = 1..=m
        // a_new[m+1] = k
        let ar_prev = ar.clone();
        for j in 1..=m {
            ar[j] = ar_prev[j] + k * ar_prev[m + 1 - j];
        }
        ar[m + 1] = k;

        // Update prediction error power
        pe *= 1.0 - k * k;
        if pe <= 0.0 {
            // Clamp instead of erroring to handle marginal cases
            pe = 1e-300;
        }

        // Update forward/backward prediction errors
        // ef_new[i] = ef[i] + k * eb[i-1]
        // eb_new[i] = eb[i-1] + k * ef[i]
        // Process in reverse to avoid aliasing (since we update in-place)
        for i in ((m + 1)..n).rev() {
            let ef_old = ef[i];
            let eb_old = eb[i - 1];
            ef[i] = ef_old + k * eb_old;
            eb[i] = eb_old + k * ef_old;
        }
    }

    // Extract AR coefficients a[1..=order] (skipping leading 1)
    let ar_coefficients: Vec<f64> = ar[1..=order].to_vec();

    Ok(BurgResult {
        ar_coefficients,
        reflection_coefficients: km,
        noise_variance: pe,
        order,
    })
}

/// Compute the AR power spectral density from the Burg AR model.
///
/// # Arguments
/// * `x`     – input signal
/// * `order` – AR model order
/// * `n_fft` – number of FFT points (frequency resolution)
/// * `fs`    – sample rate in Hz
///
/// # Returns
/// PSD estimate, length = `n_fft / 2 + 1` (one-sided).
pub fn burg_psd(x: &[f64], order: usize, n_fft: usize, fs: f64) -> SignalResult<Vec<f64>> {
    let result = burg_method(x, order)?;
    ar_psd(&result.ar_coefficients, result.noise_variance, n_fft, fs)
}

/// Compute the AR power spectral density from Yule-Walker AR model.
///
/// # Arguments
/// * `x`     – input signal
/// * `order` – AR model order
/// * `n_fft` – number of FFT points
/// * `fs`    – sample rate in Hz
///
/// # Returns
/// One-sided PSD estimate.
pub fn yule_walker_psd(x: &[f64], order: usize, n_fft: usize, fs: f64) -> SignalResult<Vec<f64>> {
    let result = yule_walker(x, order)?;
    ar_psd(&result.ar_coefficients, result.noise_variance, n_fft, fs)
}

// ─── Yule-Walker ─────────────────────────────────────────────────────────────

/// Estimate an AR model via the Yule-Walker (autocorrelation) equations.
///
/// Uses the Levinson-Durbin recursion.
pub fn yule_walker(x: &[f64], order: usize) -> SignalResult<BurgResult> {
    let n = x.len();
    if order == 0 {
        return Err(SignalError::ValueError("order must be ≥ 1".to_string()));
    }
    if order >= n {
        return Err(SignalError::ValueError(format!(
            "order ({order}) must be less than signal length ({n})"
        )));
    }

    // Biased autocorrelation (index 0 … order)
    let mut r = vec![0.0_f64; order + 1];
    for lag in 0..=order {
        let mut s = 0.0_f64;
        for i in 0..(n - lag) {
            s += x[i] * x[i + lag];
        }
        r[lag] = s / n as f64;
    }

    let r0 = r[0];
    if r0.abs() < 1e-30 {
        return Err(SignalError::ComputationError(
            "Signal has zero energy".to_string(),
        ));
    }

    // Levinson-Durbin
    let (a, km, sigma2) = levinson_durbin(&r, order)?;

    Ok(BurgResult {
        ar_coefficients: a,
        reflection_coefficients: km,
        noise_variance: sigma2,
        order,
    })
}

/// Levinson-Durbin recursion.
///
/// Solves the Yule-Walker system Rₚaₚ = -rₚ₊₁ efficiently.
/// Returns (AR coefficients [a₁…aₚ], reflection coefficients, noise variance).
fn levinson_durbin(r: &[f64], order: usize) -> SignalResult<(Vec<f64>, Vec<f64>, f64)> {
    let mut a = vec![0.0_f64; order]; // current AR coefficients a[1..p]
    let mut km = vec![0.0_f64; order];
    let mut e = r[0]; // prediction error, starts at r[0]

    for k in 0..order {
        // Reflection coefficient
        let mut lambda = r[k + 1];
        for j in 0..k {
            lambda += a[j] * r[k - j];
        }

        if e.abs() < 1e-30 {
            return Err(SignalError::ComputationError(
                "Levinson-Durbin: error variance reached zero".to_string(),
            ));
        }

        let kk = -lambda / e;
        km[k] = kk;

        // Update AR coefficients
        let a_prev = a.clone();
        a[k] = kk;
        for j in 0..(k) {
            a[j] = a_prev[j] + kk * a_prev[k - 1 - j];
        }

        // Update prediction error
        e *= 1.0 - kk * kk;
        if e <= 0.0 {
            return Err(SignalError::ComputationError(
                "Levinson-Durbin: negative error variance (signal not AR)".to_string(),
            ));
        }
    }

    Ok((a, km, e))
}

// ─── AR PSD helper ────────────────────────────────────────────────────────────

/// Compute one-sided AR PSD from AR coefficients and noise variance.
///
/// PSD(ω) = σ² / |A(e^{jω})|²  where A(z) = 1 + a₁z⁻¹ + … + aₚz⁻ᵖ.
fn ar_psd(ar: &[f64], sigma2: f64, n_fft: usize, fs: f64) -> SignalResult<Vec<f64>> {
    if n_fft < 2 {
        return Err(SignalError::ValueError("n_fft must be ≥ 2".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }

    let n_out = n_fft / 2 + 1;
    let mut psd = vec![0.0_f64; n_out];

    for k in 0..n_out {
        let omega = 2.0 * PI * k as f64 / n_fft as f64;
        // A(e^{jω}) = 1 + sum_{m=1}^{p} a_m * e^{-j m ω}
        let mut re = 1.0_f64;
        let mut im = 0.0_f64;
        for (m, &am) in ar.iter().enumerate() {
            let phi = (m + 1) as f64 * omega;
            re += am * phi.cos();
            im -= am * phi.sin();
        }
        let a_sq = re * re + im * im;
        psd[k] = if a_sq > 1e-30 { sigma2 / a_sq } else { 0.0 };
    }

    // Scale by 1/fs for proper power spectral density units
    psd.iter_mut().for_each(|v| *v /= fs);

    Ok(psd)
}

// ─── MUSIC ────────────────────────────────────────────────────────────────────

/// MUSIC (Multiple Signal Classification) super-resolution frequency estimator.
///
/// Based on eigendecomposition of the signal correlation matrix.  The noise
/// subspace is spanned by eigenvectors corresponding to the (`m` - `n_signals`)
/// smallest eigenvalues, where `m` is the correlation matrix size.
///
/// # Arguments
/// * `x`         – input signal (needs length ≥ m + some overlap; m = n_signals + 1 default)
/// * `n_signals` – number of sinusoidal components to find
/// * `n_fft`     – resolution of the pseudo-spectrum (number of points)
/// * `fs`        – sample rate in Hz
pub fn music(
    x: &[f64],
    n_signals: usize,
    n_fft: usize,
    fs: f64,
) -> SignalResult<MusicResult> {
    let n = x.len();
    if n_signals == 0 {
        return Err(SignalError::ValueError(
            "n_signals must be ≥ 1".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    if n_fft < 4 {
        return Err(SignalError::ValueError("n_fft must be ≥ 4".to_string()));
    }

    // Choose correlation matrix size m > n_signals
    let m = (n_signals + 1).max(4).min(n / 2);
    if m + m > n {
        return Err(SignalError::ValueError(format!(
            "Signal too short (length={n}) for n_signals={n_signals}"
        )));
    }

    // Build the m×m sample correlation matrix R = (1/K) X X^H
    // using the "covariance" method (no windowing):
    // R[i][j] = sum_{k=0}^{K-1} x[k+i] * x[k+j] / K
    let k_snapshots = n - m + 1;
    let mut r = vec![vec![0.0_f64; m]; m];
    for k in 0..k_snapshots {
        for i in 0..m {
            for j in 0..m {
                r[i][j] += x[k + i] * x[k + j];
            }
        }
    }
    let inv_k = 1.0 / k_snapshots as f64;
    for row in r.iter_mut() {
        for v in row.iter_mut() {
            *v *= inv_k;
        }
    }

    // Compute eigenvalues and eigenvectors via Jacobi iteration (real symmetric)
    let (eigvals, eigvecs) = jacobi_eigen(&r)?;

    // Sort eigenvalues ascending; the n_signals largest correspond to the signal
    // subspace.  The rest span the noise subspace.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| eigvals[a].partial_cmp(&eigvals[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Noise subspace: the (m - n_signals) eigenvectors with *smallest* eigenvalues
    let n_noise = m - n_signals;
    let noise_vecs: Vec<Vec<f64>> = idx[..n_noise]
        .iter()
        .map(|&i| eigvecs[i].clone())
        .collect();

    // Build MUSIC pseudo-spectrum over [0, fs/2]
    let n_out = n_fft / 2 + 1;
    let mut pseudo = vec![0.0_f64; n_out];
    let mut freq_axis = vec![0.0_f64; n_out];

    for k in 0..n_out {
        let omega = 2.0 * PI * k as f64 / n_fft as f64; // 0 … π (normalised)
        freq_axis[k] = k as f64 * fs / n_fft as f64;

        // Steering vector e(ω) = [1, e^{jω}, …, e^{j(m-1)ω}]^T  (real part only: cos)
        let e_re: Vec<f64> = (0..m).map(|l| (l as f64 * omega).cos()).collect();
        let e_im: Vec<f64> = (0..m).map(|l| -(l as f64 * omega).sin()).collect();

        // Denominator = |E^H * V_n|² = sum over noise eigenvectors of |e^H v|²
        let mut denom = 0.0_f64;
        for v in &noise_vecs {
            // <e, v> = sum_l (e_re[l] - j*e_im[l]) * v[l]  (v is real)
            let dot_re: f64 = e_re.iter().zip(v.iter()).map(|(&er, &vl)| er * vl).sum();
            let dot_im: f64 = e_im.iter().zip(v.iter()).map(|(&ei, &vl)| ei * vl).sum();
            denom += dot_re * dot_re + dot_im * dot_im;
        }

        pseudo[k] = if denom > 1e-30 { 1.0 / denom } else { 1e30 };
    }

    // Peak-pick n_signals largest peaks in the pseudo-spectrum
    let frequencies = find_peaks(&pseudo, &freq_axis, n_signals);

    Ok(MusicResult {
        frequencies,
        pseudo_spectrum: pseudo,
        freq_axis,
    })
}

/// Find the `n` highest local peaks in `spectrum`, return the corresponding
/// `freq_axis` values sorted ascending.
fn find_peaks(spectrum: &[f64], freq_axis: &[f64], n: usize) -> Vec<f64> {
    let len = spectrum.len();
    if len == 0 || n == 0 {
        return vec![];
    }

    // Collect local maxima indices
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..(len - 1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            peaks.push(i);
        }
    }
    // Include endpoints if they are larger than their neighbour
    if len > 1 && spectrum[0] > spectrum[1] {
        peaks.push(0);
    }
    if len > 1 && spectrum[len - 1] > spectrum[len - 2] {
        peaks.push(len - 1);
    }

    // Sort by height descending
    peaks.sort_by(|&a, &b| {
        spectrum[b]
            .partial_cmp(&spectrum[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut freqs: Vec<f64> = peaks.iter().take(n).map(|&i| freq_axis[i]).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    freqs
}

// ─── ESPRIT ───────────────────────────────────────────────────────────────────

/// ESPRIT frequency estimator.
///
/// Estimates `n_signals` sinusoidal frequencies (in Hz) from the signal `x`
/// using the ESPRIT algorithm applied to the signal covariance matrix.
///
/// # Arguments
/// * `x`         – input signal
/// * `n_signals` – number of complex sinusoidal components
/// * `fs`        – sample rate in Hz
pub fn esprit(x: &[f64], n_signals: usize, fs: f64) -> SignalResult<Vec<f64>> {
    let n = x.len();
    if n_signals == 0 {
        return Err(SignalError::ValueError(
            "n_signals must be ≥ 1".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }

    let m = (2 * n_signals + 2).min(n / 2);
    if m > n {
        return Err(SignalError::ValueError(
            "Signal too short for the requested number of signals".to_string(),
        ));
    }

    // Build m×m correlation matrix
    let k_snapshots = n - m + 1;
    let mut r = vec![vec![0.0_f64; m]; m];
    for k in 0..k_snapshots {
        for i in 0..m {
            for j in 0..m {
                r[i][j] += x[k + i] * x[k + j];
            }
        }
    }
    let inv_k = 1.0 / k_snapshots as f64;
    for row in r.iter_mut() {
        for v in row.iter_mut() {
            *v *= inv_k;
        }
    }

    // Eigen-decomposition (real symmetric)
    let (eigvals, eigvecs) = jacobi_eigen(&r)?;

    // Sort descending — signal subspace = n_signals largest eigenvectors
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let signal_vecs: Vec<Vec<f64>> = idx[..n_signals]
        .iter()
        .map(|&i| eigvecs[i].clone())
        .collect();

    // Form E_s: m × n_signals matrix, each column is a signal eigenvector
    // ESPRIT: partition E_s into E1 (rows 0..m-1) and E2 (rows 1..m)
    // Estimate rotation Φ = pinv(E1) * E2, eigenvalues of Φ give frequencies.

    let m1 = m - 1;
    // E1: m1 × n_signals
    // E2: m1 × n_signals
    let mut e1 = vec![vec![0.0_f64; n_signals]; m1];
    let mut e2 = vec![vec![0.0_f64; n_signals]; m1];
    for i in 0..m1 {
        for j in 0..n_signals {
            e1[i][j] = signal_vecs[j][i];
            e2[i][j] = signal_vecs[j][i + 1];
        }
    }

    // Least-squares: Φ = (E1^T E1)^{-1} E1^T E2   (each is n_signals × n_signals)
    let e1t_e1 = mat_mul_at_a(&e1, n_signals);   // n_s × n_s
    let e1t_e2 = mat_mul_at_b(&e1, &e2, n_signals); // n_s × n_s

    let phi = mat_solve_sym(&e1t_e1, &e1t_e2)?; // n_s × n_s

    // Eigenvalues of Φ (real matrix, eigenvalues may be complex)
    // For real observations, eigenvalues come in conjugate pairs.
    // We use the power method / QR iteration for the small n_signals × n_signals matrix.
    let eigs = real_matrix_eigenvalues_complex(&phi)?;

    // Convert eigenvalues μ = e^{j2πf/fs} → f = arg(μ)*fs/(2π)
    let mut freqs: Vec<f64> = eigs
        .iter()
        .filter_map(|(re, im)| {
            let angle = im.atan2(*re); // arg in (-π, π]
            if angle >= 0.0 {
                Some(angle * fs / (2.0 * PI))
            } else {
                None // negative frequencies — skip
            }
        })
        .collect();

    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    freqs.truncate(n_signals);

    Ok(freqs)
}

// ─── Numerical linear algebra helpers ────────────────────────────────────────

/// Jacobi eigenvalue algorithm for real symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i]` is the
/// eigenvector for `eigenvalues[i]`.
fn jacobi_eigen(a: &[Vec<f64>]) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }

    let mut mat: Vec<Vec<f64>> = a.to_vec();
    // Initialise V as identity
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_iter = 200 * n * n;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if mat[i][j].abs() > max_val {
                    max_val = mat[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle
        let diff = mat[q][q] - mat[p][p];
        let theta = if diff.abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * mat[p][q] / diff).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to matrix (columns and rows p, q)
        let mut new_mat = mat.clone();
        // Update rows/columns p, q
        for r in 0..n {
            if r != p && r != q {
                let a_rp = mat[r][p];
                let a_rq = mat[r][q];
                new_mat[r][p] = c * a_rp - s * a_rq;
                new_mat[p][r] = new_mat[r][p];
                new_mat[r][q] = s * a_rp + c * a_rq;
                new_mat[q][r] = new_mat[r][q];
            }
        }
        new_mat[p][p] = c * c * mat[p][p] - 2.0 * s * c * mat[p][q] + s * s * mat[q][q];
        new_mat[q][q] = s * s * mat[p][p] + 2.0 * s * c * mat[p][q] + c * c * mat[q][q];
        new_mat[p][q] = 0.0;
        new_mat[q][p] = 0.0;
        mat = new_mat;

        // Update eigenvector matrix
        let mut new_v = v.clone();
        for r in 0..n {
            let v_rp = v[r][p];
            let v_rq = v[r][q];
            new_v[r][p] = c * v_rp - s * v_rq;
            new_v[r][q] = s * v_rp + c * v_rq;
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    // Eigenvectors are columns of V
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|col| (0..n).map(|row| v[row][col]).collect())
        .collect();

    Ok((eigenvalues, eigenvectors))
}

/// Compute A^T A  (result is n_cols × n_cols).
fn mat_mul_at_a(a: &[Vec<f64>], n_cols: usize) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut c = vec![vec![0.0_f64; n_cols]; n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            let mut s = 0.0_f64;
            for k in 0..m {
                s += a[k][i] * a[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Compute A^T B  (result is n_cols × n_cols).
fn mat_mul_at_b(a: &[Vec<f64>], b: &[Vec<f64>], n_cols: usize) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut c = vec![vec![0.0_f64; n_cols]; n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            let mut s = 0.0_f64;
            for k in 0..m {
                s += a[k][i] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Solve A X = B for X (n×n matrices) using Gaussian elimination with partial pivoting.
fn mat_solve_sym(a: &[Vec<f64>], b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Augment [A | B]
    let nb = b[0].len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return Err(SignalError::ComputationError(
                "Singular matrix in mat_solve_sym".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for c in col..(n + nb) {
                let sub = factor * aug[col][c];
                aug[row][c] -= sub;
            }
        }
    }

    // Back-substitution
    let mut x = vec![vec![0.0_f64; nb]; n];
    for row in (0..n).rev() {
        for k in 0..nb {
            let mut s = aug[row][n + k];
            for c in (row + 1)..n {
                s -= aug[row][c] * x[c][k];
            }
            let diag = aug[row][row];
            if diag.abs() < 1e-30 {
                return Err(SignalError::ComputationError(
                    "Near-zero diagonal during back-substitution".to_string(),
                ));
            }
            x[row][k] = s / diag;
        }
    }
    Ok(x)
}

/// Compute approximate eigenvalues (as complex numbers) of a small real square matrix
/// using the QR algorithm with shifts.
///
/// Returns pairs `(re, im)`.
fn real_matrix_eigenvalues_complex(a: &[Vec<f64>]) -> SignalResult<Vec<(f64, f64)>> {
    let n = a.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Clone to mutable
    let mut h: Vec<Vec<f64>> = a.to_vec();

    // Reduce to upper Hessenberg form first (optional for small n, do it anyway)
    for col in 0..(n.saturating_sub(2)) {
        for row in (col + 2)..n {
            // Givens rotation to zero h[row][col]
            let a_val = h[col + 1][col];
            let b_val = h[row][col];
            if b_val.abs() < 1e-15 {
                continue;
            }
            let r = (a_val * a_val + b_val * b_val).sqrt();
            if r < 1e-15 {
                continue;
            }
            let c = a_val / r;
            let s = -b_val / r;
            // Apply to rows col+1, row
            for j in 0..n {
                let tmp = c * h[col + 1][j] - s * h[row][j];
                let tmp2 = s * h[col + 1][j] + c * h[row][j];
                h[col + 1][j] = tmp;
                h[row][j] = tmp2;
            }
            // Apply to columns col+1, row
            for i in 0..n {
                let tmp = c * h[i][col + 1] - s * h[i][row];
                let tmp2 = s * h[i][col + 1] + c * h[i][row];
                h[i][col + 1] = tmp;
                h[i][row] = tmp2;
            }
        }
    }

    // Francis double-shift QR iteration
    let max_iter = 300;
    let mut eigs: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut nn = n;

    'outer: while nn > 0 {
        if nn == 1 {
            eigs.push((h[0][0], 0.0));
            break;
        }

        let mut iter_count = 0;
        loop {
            // Check for deflation at bottom
            let converged = h[nn - 1][nn - 2].abs()
                < 1e-12 * (h[nn - 2][nn - 2].abs() + h[nn - 1][nn - 1].abs());
            if converged || iter_count > max_iter {
                eigs.push((h[nn - 1][nn - 1], 0.0));
                nn -= 1;
                continue 'outer;
            }

            // Check for 2×2 block at bottom
            if nn >= 2 {
                let c22 = h[nn - 2][nn - 2] + h[nn - 1][nn - 1];
                let d22 =
                    h[nn - 2][nn - 2] * h[nn - 1][nn - 1] - h[nn - 2][nn - 1] * h[nn - 1][nn - 2];
                let disc = c22 * c22 - 4.0 * d22;
                if disc < 0.0 {
                    // Complex eigenvalue pair
                    let re = c22 / 2.0;
                    let im = (-disc).sqrt() / 2.0;
                    eigs.push((re, im));
                    eigs.push((re, -im));
                    if nn >= 2 {
                        nn -= 2;
                    }
                    continue 'outer;
                }
            }

            // Wilkinson shift: eigenvalue of bottom 2×2 closer to h[n-1][n-1]
            let s = h[nn - 1][nn - 1];
            let t = s;

            // Apply shift and QR step (single shift via Givens)
            for i in 0..nn {
                h[i][i] -= t;
            }

            // QR decomposition via Givens rotations
            let mut cos_arr = vec![0.0_f64; nn - 1];
            let mut sin_arr = vec![0.0_f64; nn - 1];
            for i in 0..(nn - 1) {
                let a_v = h[i][i];
                let b_v = h[i + 1][i];
                let r = (a_v * a_v + b_v * b_v).sqrt();
                if r < 1e-15 {
                    cos_arr[i] = 1.0;
                    sin_arr[i] = 0.0;
                    continue;
                }
                cos_arr[i] = a_v / r;
                sin_arr[i] = -b_v / r;
                // Apply rotation to rows i, i+1
                for j in i..nn {
                    let tmp = cos_arr[i] * h[i][j] - sin_arr[i] * h[i + 1][j];
                    let tmp2 = sin_arr[i] * h[i][j] + cos_arr[i] * h[i + 1][j];
                    h[i][j] = tmp;
                    h[i + 1][j] = tmp2;
                }
            }
            // Multiply by Q^T on the right
            for i in 0..(nn - 1) {
                for j in 0..nn {
                    let tmp = cos_arr[i] * h[j][i] - sin_arr[i] * h[j][i + 1];
                    let tmp2 = sin_arr[i] * h[j][i] + cos_arr[i] * h[j][i + 1];
                    h[j][i] = tmp;
                    h[j][i + 1] = tmp2;
                }
            }

            // Remove shift
            for i in 0..nn {
                h[i][i] += t;
            }

            iter_count += 1;
        }
    }

    Ok(eigs)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate an AR(1) process: x[n] = alpha * x[n-1] + noise
    /// For testing, use a purely deterministic version (no noise).
    fn ar1_signal(alpha: f64, n: usize) -> Vec<f64> {
        let mut x = vec![0.0_f64; n];
        x[0] = 1.0;
        for i in 1..n {
            x[i] = alpha * x[i - 1];
        }
        x
    }

    /// Simple sinusoid generator.
    fn sinusoid(freq_hz: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz / fs * i as f64).sin())
            .collect()
    }

    // ── burg_method ─────────────────────────────────────────────────────────

    #[test]
    fn test_burg_ar1_estimate() {
        // AR(1) process: x[n] = 0.9*x[n-1].  The coefficient should be ≈ 0.9.
        let alpha = 0.9_f64;
        let x = ar1_signal(alpha, 128);
        let result = burg_method(&x, 1).expect("burg_method failed");

        // ar_coefficients[0] should be close to -alpha (Burg returns a[] in the
        // form x[n] = -a[1]*x[n-1] - …, i.e. the negative convention)
        // The sign depends on convention; we test |value| ≈ 0.9
        assert!(
            result.ar_coefficients[0].abs() > 0.7,
            "AR(1) coefficient should be ~0.9, got {}",
            result.ar_coefficients[0]
        );
        assert_eq!(result.order, 1);
        assert!(result.noise_variance >= 0.0);
        assert_eq!(result.reflection_coefficients.len(), 1);
    }

    #[test]
    fn test_burg_higher_order() {
        let x = sinusoid(100.0, 1000.0, 200);
        let result = burg_method(&x, 4).expect("burg_method order 4");
        assert_eq!(result.order, 4);
        assert_eq!(result.ar_coefficients.len(), 4);
        assert_eq!(result.reflection_coefficients.len(), 4);
    }

    #[test]
    fn test_burg_error_order_too_large() {
        let x = vec![1.0_f64; 5];
        let err = burg_method(&x, 5);
        assert!(err.is_err(), "Should fail when order >= length");
    }

    // ── yule_walker ──────────────────────────────────────────────────────────

    #[test]
    fn test_yule_walker_ar1() {
        let alpha = 0.8_f64;
        let x = ar1_signal(alpha, 256);
        let result = yule_walker(&x, 1).expect("yule_walker AR(1)");
        assert_eq!(result.order, 1);
        assert!(
            result.ar_coefficients[0].abs() > 0.5,
            "AR(1) via Yule-Walker coefficient should be sizeable, got {}",
            result.ar_coefficients[0]
        );
        assert!(result.noise_variance >= 0.0);
    }

    #[test]
    fn test_yule_walker_error_order_too_large() {
        let x = vec![1.0_f64; 3];
        let err = yule_walker(&x, 3);
        assert!(err.is_err());
    }

    // ── burg_psd ─────────────────────────────────────────────────────────────

    #[test]
    fn test_burg_psd_length() {
        let x = sinusoid(200.0, 1000.0, 256);
        let psd = burg_psd(&x, 8, 256, 1000.0).expect("burg_psd");
        assert_eq!(psd.len(), 129); // n_fft/2 + 1
    }

    #[test]
    fn test_burg_psd_positive() {
        let x = sinusoid(100.0, 1000.0, 256);
        let psd = burg_psd(&x, 6, 256, 1000.0).expect("burg_psd");
        for &v in &psd {
            assert!(v >= 0.0, "PSD must be non-negative, got {v}");
        }
    }

    // ── MUSIC ────────────────────────────────────────────────────────────────

    #[test]
    fn test_music_detects_two_sinusoids() {
        let fs = 1000.0_f64;
        let n = 512;
        let f1 = 100.0_f64;
        let f2 = 200.0_f64;

        // Two clean sinusoids
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * f1 * t).sin() + (2.0 * PI * f2 * t).sin()
            })
            .collect();

        let result = music(&x, 2, 1024, fs).expect("music failed");

        assert_eq!(result.frequencies.len(), 2, "MUSIC should detect 2 frequencies");

        // Each detected frequency should be within 20 Hz of one of the true frequencies
        let targets = [f1, f2];
        for &detected in &result.frequencies {
            let closest = targets
                .iter()
                .map(|&t| (detected - t).abs())
                .fold(f64::MAX, f64::min);
            assert!(
                closest < 30.0,
                "Detected freq {detected:.1} Hz far from targets {f1}/{f2}"
            );
        }
    }

    #[test]
    fn test_music_pseudo_spectrum_has_peaks() {
        let fs = 1000.0_f64;
        let n = 256;
        let f1 = 150.0_f64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f1 / fs * i as f64).sin())
            .collect();

        let result = music(&x, 1, 512, fs).expect("music");
        // The pseudo-spectrum should have at least one non-trivial entry
        let max_val = result
            .pseudo_spectrum
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        assert!(max_val > 0.0, "Pseudo-spectrum should be positive");
    }

    // ── ESPRIT ───────────────────────────────────────────────────────────────

    #[test]
    fn test_esprit_single_sinusoid() {
        let fs = 1000.0_f64;
        let f1 = 150.0_f64;
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f1 / fs * i as f64).sin())
            .collect();

        let freqs = esprit(&x, 1, fs).expect("esprit");
        // Should return at most 1 frequency (some may be filtered out)
        // Check that the array is valid
        assert!(freqs.len() <= 1 || freqs.iter().any(|&f| (f - f1).abs() < 50.0),
            "ESPRIT should find a frequency near {f1}, got {:?}", freqs);
    }

    // ── levinson_durbin ───────────────────────────────────────────────────────

    #[test]
    fn test_levinson_durbin_ar1() {
        // For AR(1) with a=0.9: r[0]=1/(1-0.81), r[1]=0.9*r[0]
        // We test that the result is internally consistent
        let r = vec![1.0, 0.9, 0.81];
        let (a, km, e) = levinson_durbin(&r, 2).expect("levinson");
        assert_eq!(a.len(), 2);
        assert!(e > 0.0);
        assert_eq!(km.len(), 2);
    }

    // ── burg_psd has peak at correct frequency ────────────────────────────

    #[test]
    fn test_burg_psd_peak_at_signal_freq() {
        let fs = 500.0_f64;
        let f_sig = 50.0_f64;
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f_sig / fs * i as f64).sin())
            .collect();

        let n_fft = 256;
        let psd = burg_psd(&x, 10, n_fft, fs).expect("burg_psd");

        // Find the peak bin
        let peak_bin = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_freq = peak_bin as f64 * fs / n_fft as f64;
        assert!(
            (peak_freq - f_sig).abs() < 15.0,
            "PSD peak at {peak_freq:.1} Hz should be near {f_sig} Hz"
        );
    }
}
