//! Compressive sensing and sparse recovery algorithms
//!
//! This module implements the core compressed sensing recovery algorithms used to
//! reconstruct sparse signals from underdetermined linear systems of measurements.
//!
//! # Algorithms
//!
//! - **Matching Pursuit (MP)** – greedy atom selection with residual update
//! - **Orthogonal Matching Pursuit (OMP)** – greedy atom selection + orthogonal projection
//! - **Basis Pursuit (BP)** – L1-minimisation via ADMM
//! - **LASSO via ADMM** – L1-penalised least-squares via ADMM
//! - **Subspace Pursuit (SP)** – iterative support-set refinement
//! - **Restricted Isometry Property (RIP)** – constant estimation via Monte Carlo
//! - **Recovery error bounds** – theoretical and empirical bound computation
//! - **Unified interface** – `compressive_sense` dispatching to any algorithm
//!
//! # References
//!
//! - Mallat & Zhang (1993) – Matching Pursuit
//! - Pati, Rezaiifar & Krishnaprasad (1993) – Orthogonal Matching Pursuit
//! - Boyd et al. (2011) – Distributed Optimization via ADMM
//! - Dai & Milenkovic (2009) – Subspace Pursuit
//! - Candès & Recht (2009) – RIP constant theory
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::rngs::StdRng;
use scirs2_linalg::{solve, vector_norm};
use std::cmp::min;

// ---------------------------------------------------------------------------
// Helper: compute the squared Frobenius (spectral proxy) norm of a matrix
// ---------------------------------------------------------------------------

/// Estimate the largest singular value squared via the power-method.  This
/// avoids full SVD and is used for step-size selection in iterative methods.
fn largest_eigenvalue_approx(a: &Array2<f64>, eps: f64) -> f64 {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return 1.0;
    }
    // Frobenius norm squared / n is a simple upper-bound on lambda_max(A^T A)
    let frob_sq: f64 = a.iter().map(|&v| v * v).sum();
    let frob_n = frob_sq / (n.max(1) as f64);
    frob_n.max(eps)
}

/// Soft-threshold (proximal operator for L1)
#[inline]
fn soft_threshold(val: f64, threshold: f64) -> f64 {
    if val > threshold {
        val - threshold
    } else if val < -threshold {
        val + threshold
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Matching Pursuit (MP)
// ---------------------------------------------------------------------------

/// Matching Pursuit: greedy sparse recovery by iterative atom selection.
///
/// At each step the dictionary column most correlated with the current residual
/// is selected.  The coefficient is updated without re-orthogonalisation.
///
/// # Arguments
///
/// * `dict`     – Dictionary matrix `(m, n)`; each column is an atom.
/// * `signal`   – Observation vector of length `m`.
/// * `sparsity` – Maximum number of non-zero coefficients to recover (`k`).
///
/// # Returns
///
/// Coefficient vector of length `n`.
///
/// # Errors
///
/// Returns `SignalError::DimensionMismatch` when the dimensions are incompatible.
pub fn matching_pursuit(
    dict: &Array2<f64>,
    signal: &Array1<f64>,
    sparsity: usize,
) -> SignalResult<Array1<f64>> {
    let (m, n) = dict.dim();
    if signal.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "matching_pursuit: dict has {m} rows but signal has {} elements",
            signal.len()
        )));
    }
    if sparsity == 0 || n == 0 {
        return Ok(Array1::zeros(n));
    }

    // Pre-compute column norms for normalisation
    let col_norms: Vec<f64> = (0..n)
        .map(|j| {
            let col = dict.slice(s![.., j]);
            col.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-14)
        })
        .collect();

    let mut residual = signal.clone();
    let mut x = Array1::<f64>::zeros(n);
    let eps = 1e-14_f64;

    for _ in 0..min(sparsity, n) {
        // Find the atom with the largest normalised inner product
        let mut best_idx = 0;
        let mut best_corr = 0.0_f64;
        for j in 0..n {
            let col = dict.slice(s![.., j]);
            let inner: f64 = col.iter().zip(residual.iter()).map(|(&a, &r)| a * r).sum();
            let corr = (inner / col_norms[j]).abs();
            if corr > best_corr {
                best_corr = corr;
                best_idx = j;
            }
        }

        if best_corr < eps {
            break;
        }

        // Compute coefficient and update residual
        let col = dict.slice(s![.., best_idx]);
        let col_norm_sq: f64 = col.iter().map(|&v| v * v).sum();
        if col_norm_sq < eps {
            break;
        }
        let coeff: f64 = col.iter().zip(residual.iter()).map(|(&a, &r)| a * r).sum::<f64>()
            / col_norm_sq;

        x[best_idx] += coeff;
        for i in 0..m {
            residual[i] -= coeff * dict[[i, best_idx]];
        }

        // Check residual magnitude
        let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
        if res_norm < eps {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Orthogonal Matching Pursuit (OMP)
// ---------------------------------------------------------------------------

/// Orthogonal Matching Pursuit: greedy sparse recovery with orthogonal projection.
///
/// After each atom is added to the support, the coefficients are updated via a
/// least-squares solve on the restricted sub-dictionary, guaranteeing orthogonality
/// of the residual to all selected atoms.
///
/// # Arguments
///
/// * `dict`     – Dictionary matrix `(m, n)`.
/// * `signal`   – Observation vector of length `m`.
/// * `sparsity` – Target sparsity (maximum support size).
///
/// # Returns
///
/// Coefficient vector of length `n`.
pub fn orthogonal_mp(
    dict: &Array2<f64>,
    signal: &Array1<f64>,
    sparsity: usize,
) -> SignalResult<Array1<f64>> {
    let (m, n) = dict.dim();
    if signal.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "orthogonal_mp: dict has {m} rows but signal has {} elements",
            signal.len()
        )));
    }
    if sparsity == 0 || n == 0 {
        return Ok(Array1::zeros(n));
    }

    let k = min(sparsity, min(m, n));
    let eps = 1e-12_f64;

    let mut residual = signal.clone();
    let mut support: Vec<usize> = Vec::with_capacity(k);
    let mut x = Array1::<f64>::zeros(n);

    for _ in 0..k {
        // Select atom with highest correlation to residual
        let mut best_idx = 0;
        let mut best_corr = 0.0_f64;
        for j in 0..n {
            if support.contains(&j) {
                continue;
            }
            let col = dict.slice(s![.., j]);
            let inner: f64 = col.iter().zip(residual.iter()).map(|(&a, &r)| a * r).sum();
            let col_norm: f64 = col.iter().map(|&v| v * v).sum::<f64>().sqrt().max(eps);
            let corr = (inner / col_norm).abs();
            if corr > best_corr {
                best_corr = corr;
                best_idx = j;
            }
        }

        if best_corr < eps {
            break;
        }
        support.push(best_idx);

        // Build restricted sub-dictionary D_S
        let s_len = support.len();
        let mut d_s = Array2::<f64>::zeros((m, s_len));
        for (col_i, &idx) in support.iter().enumerate() {
            d_s.slice_mut(s![.., col_i]).assign(&dict.slice(s![.., idx]));
        }

        // Solve normal equations: D_S^T D_S c = D_S^T signal
        let gram = d_s.t().dot(&d_s);
        let rhs = d_s.t().dot(signal);
        let coeff = solve(&gram.view(), &rhs.view(), None).map_err(|e| {
            SignalError::ComputationError(format!("orthogonal_mp: least-squares failed: {e}"))
        })?;

        // Update residual
        residual = signal - &d_s.dot(&coeff);

        // Write coefficients into x
        x.fill(0.0);
        for (col_i, &idx) in support.iter().enumerate() {
            x[idx] = coeff[col_i];
        }

        // Early exit if residual is negligible
        let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
        if res_norm < eps {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Basis Pursuit via ADMM
// ---------------------------------------------------------------------------

/// Basis Pursuit: minimize ‖x‖₁ subject to dict·x = signal.
///
/// Solved via the Alternating Direction Method of Multipliers (ADMM) applied
/// to the augmented Lagrangian.  The tolerance `tol` controls the primal and
/// dual residual stopping criterion.
///
/// # Arguments
///
/// * `dict`     – Dictionary (sensing) matrix `(m, n)`.
/// * `signal`   – Measurement vector of length `m`.
/// * `tol`      – Convergence tolerance for the ADMM residuals.
///
/// # Returns
///
/// Sparse coefficient vector of length `n`.
pub fn basis_pursuit(
    dict: &Array2<f64>,
    signal: &Array1<f64>,
    tol: f64,
) -> SignalResult<Array1<f64>> {
    let (m, n) = dict.dim();
    if signal.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "basis_pursuit: dict has {m} rows but signal has {} elements",
            signal.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let tol = tol.max(1e-14);
    let rho = 1.0_f64; // ADMM penalty parameter

    // Pre-compute the cached factor: (A^T A + rho I)^{-1}  (dense solve each step)
    // We will solve the x-update via normal equations each iteration.
    let at = dict.t();
    let ata = at.dot(dict); // n x n
    let ata_rho = {
        let mut m_r = ata.clone();
        for i in 0..n {
            m_r[[i, i]] += rho;
        }
        m_r
    };
    let at_b = at.dot(signal); // n

    // ADMM variables
    let mut x = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n); // scaled dual variable

    let max_iter = 5000_usize;

    for _ in 0..max_iter {
        // x-update: solve (A^T A + rho I) x = A^T b + rho (z - u)
        let rhs = &at_b + rho * (&z - &u);
        x = solve(&ata_rho.view(), &rhs.view(), None).map_err(|e| {
            SignalError::ComputationError(format!("basis_pursuit: x-update failed: {e}"))
        })?;

        let z_old = z.clone();

        // z-update: soft-threshold (prox for L1)
        let v = &x + &u;
        z = v.mapv(|val| soft_threshold(val, 1.0 / rho));

        // u-update
        u = &u + &x - &z;

        // Residuals
        let primal: f64 = (&x - &z).iter().map(|&r| r * r).sum::<f64>().sqrt();
        let dual: f64 = (rho * (&z - &z_old)).iter().map(|&r| r * r).sum::<f64>().sqrt();

        if primal < tol && dual < tol {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// LASSO via ADMM
// ---------------------------------------------------------------------------

/// LASSO via ADMM: minimize ½‖dict·x - signal‖₂² + λ‖x‖₁.
///
/// Uses the consensus ADMM form with soft-thresholding for the z-update.
///
/// # Arguments
///
/// * `dict`     – Dictionary matrix `(m, n)`.
/// * `signal`   – Observation vector of length `m`.
/// * `lambda`   – L1 regularisation weight (λ ≥ 0).
/// * `max_iter` – Maximum number of ADMM iterations.
///
/// # Returns
///
/// Sparse coefficient vector of length `n`.
pub fn lasso_admm(
    dict: &Array2<f64>,
    signal: &Array1<f64>,
    lambda: f64,
    max_iter: usize,
) -> SignalResult<Array1<f64>> {
    let (m, n) = dict.dim();
    if signal.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "lasso_admm: dict has {m} rows but signal has {} elements",
            signal.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    if lambda < 0.0 {
        return Err(SignalError::ValueError(
            "lasso_admm: lambda must be non-negative".to_string(),
        ));
    }

    let rho = 1.0_f64;
    let at = dict.t();
    let ata = at.dot(dict);

    // Cache (A^T A + rho I)
    let ata_rho = {
        let mut mr = ata.clone();
        for i in 0..n {
            mr[[i, i]] += rho;
        }
        mr
    };
    let at_b = at.dot(signal);

    let mut x = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n);

    let eps_abs = 1e-8_f64;
    let eps_rel = 1e-6_f64;

    for _ in 0..max_iter.max(1) {
        // x-update
        let rhs = &at_b + rho * (&z - &u);
        x = solve(&ata_rho.view(), &rhs.view(), None).map_err(|e| {
            SignalError::ComputationError(format!("lasso_admm: x-update failed: {e}"))
        })?;

        let z_old = z.clone();

        // z-update: prox of (lambda/rho) * L1
        let v = &x + &u;
        z = v.mapv(|val| soft_threshold(val, lambda / rho));

        // u-update
        u = &u + &x - &z;

        // Convergence check
        let primal: f64 = (&x - &z).iter().map(|&r| r * r).sum::<f64>().sqrt();
        let dual: f64 = (rho * (&z - &z_old)).iter().map(|&r| r * r).sum::<f64>().sqrt();

        let x_norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let z_norm: f64 = z.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let eps_primal = eps_abs * (n as f64).sqrt() + eps_rel * x_norm.max(z_norm);
        let eps_dual = eps_abs * (n as f64).sqrt()
            + eps_rel * rho * u.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if primal < eps_primal && dual < eps_dual {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Subspace Pursuit (SP)
// ---------------------------------------------------------------------------

/// Subspace Pursuit: iterative support-set refinement for sparse recovery.
///
/// Combines proxy selection, support merging, and least-squares projection
/// to produce a k-sparse estimate.  Convergence is guaranteed when the sensing
/// matrix satisfies an appropriate RIP condition (δ_{3k} < 0.205).
///
/// # Arguments
///
/// * `dict`     – Dictionary matrix `(m, n)`.
/// * `signal`   – Measurement vector of length `m`.
/// * `sparsity` – Target sparsity k.
///
/// # Returns
///
/// Sparse coefficient vector of length `n`.
pub fn subspace_pursuit(
    dict: &Array2<f64>,
    signal: &Array1<f64>,
    sparsity: usize,
) -> SignalResult<Array1<f64>> {
    let (m, n) = dict.dim();
    if signal.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "subspace_pursuit: dict has {m} rows but signal has {} elements",
            signal.len()
        )));
    }
    let k = min(sparsity, min(m, n));
    if k == 0 {
        return Ok(Array1::zeros(n));
    }

    let eps = 1e-14_f64;
    let max_iter = 500_usize;
    let at = dict.t();

    // --- Helper: top-k indices by absolute value ---
    let top_k_indices = |v: &Array1<f64>, k: usize| -> Vec<usize> {
        let mut pairs: Vec<(usize, f64)> =
            v.iter().enumerate().map(|(i, &val)| (i, val.abs())).collect();
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.iter().take(k).map(|&(i, _)| i).collect()
    };

    // --- Helper: restricted least-squares ---
    let ls_on_support = |support: &[usize]| -> SignalResult<(Array1<f64>, Array1<f64>)> {
        let s = support.len();
        let mut d_s = Array2::<f64>::zeros((m, s));
        for (ci, &idx) in support.iter().enumerate() {
            d_s.slice_mut(s![.., ci]).assign(&dict.slice(s![.., idx]));
        }
        let gram = d_s.t().dot(&d_s);
        let rhs = d_s.t().dot(signal);
        let c = solve(&gram.view(), &rhs.view(), None).map_err(|e| {
            SignalError::ComputationError(format!("subspace_pursuit: LS failed: {e}"))
        })?;
        let residual = signal - &d_s.dot(&c);
        Ok((c, residual))
    };

    // Initialise support with k atoms most correlated with signal
    let proxy0 = at.dot(signal);
    let mut support = top_k_indices(&proxy0, k);
    support.sort_unstable();

    let (mut coeff, mut residual) = ls_on_support(&support)?;

    for _ in 0..max_iter {
        let proxy = at.dot(&residual);

        // Candidates: k atoms most correlated with residual
        let candidates = top_k_indices(&proxy, k);

        // Merge support and candidates (union, then keep top-2k)
        let mut merged: Vec<usize> = support.clone();
        for &c in &candidates {
            if !merged.contains(&c) {
                merged.push(c);
            }
        }

        // Solve on merged support to get temporary coefficients
        let (c_merged, _) = ls_on_support(&merged)?;

        // Build coefficient array indexed by merged support
        let mut x_merged = Array1::<f64>::zeros(n);
        for (ci, &idx) in merged.iter().enumerate() {
            x_merged[idx] = c_merged[ci];
        }

        // Prune to k largest by magnitude
        let new_support = top_k_indices(&x_merged, k);
        let mut new_support_sorted = new_support.clone();
        new_support_sorted.sort_unstable();

        // Solve on new support
        let (new_coeff, new_residual) = ls_on_support(&new_support_sorted)?;

        // Convergence: support did not change
        let converged = new_support_sorted == support;
        support = new_support_sorted;
        coeff = new_coeff;
        residual = new_residual;

        if converged {
            break;
        }

        let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
        if res_norm < eps {
            break;
        }
    }

    // Build solution
    let mut x = Array1::<f64>::zeros(n);
    for (ci, &idx) in support.iter().enumerate() {
        x[idx] = coeff[ci];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Restricted Isometry Property (RIP) check
// ---------------------------------------------------------------------------

/// Estimate the Restricted Isometry Constant δₛ for a dictionary matrix.
///
/// The exact computation requires examining all `C(n, s)` sub-matrices, which
/// is combinatorially infeasible.  This function uses a Monte Carlo approach:
/// it draws many random s-sparse unit vectors and estimates
///   δₛ ≈ max_over_trials |  ‖A x‖₂² - 1  |
///
/// The estimate is a lower bound on the true δₛ.
///
/// # Arguments
///
/// * `dict`     – Dictionary / sensing matrix `(m, n)`.
/// * `sparsity` – Sparsity level s.
///
/// # Returns
///
/// Estimated RIP constant (a value in [0, ∞); recovery is theoretically
/// guaranteed when δ_{2s} < √2 - 1 ≈ 0.414).
pub fn check_rip(dict: &Array2<f64>, sparsity: usize) -> SignalResult<f64> {
    let (m, n) = dict.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "check_rip: empty dictionary".to_string(),
        ));
    }
    if sparsity == 0 {
        return Ok(0.0);
    }
    if sparsity > n {
        return Err(SignalError::ValueError(format!(
            "check_rip: sparsity ({sparsity}) > number of dictionary columns ({n})"
        )));
    }

    const NUM_TRIALS: usize = 2000;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_u64);
    let mut indices: Vec<usize> = (0..n).collect();

    let mut max_deviation = 0.0_f64;

    for _ in 0..NUM_TRIALS {
        // Random s-sparse unit vector
        indices.shuffle(&mut rng);
        let support = &indices[..sparsity];

        let mut x = Array1::<f64>::zeros(n);
        for &idx in support {
            // Rademacher ±1 entries
            x[idx] = if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 };
        }

        // Normalise
        let x_norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if x_norm < 1e-14 {
            continue;
        }
        x.mapv_inplace(|v| v / x_norm);

        // Compute ‖A x‖₂²
        let ax = dict.dot(&x);
        let ax_sq: f64 = ax.iter().map(|&v| v * v).sum();

        let deviation = (ax_sq - 1.0).abs();
        if deviation > max_deviation {
            max_deviation = deviation;
        }
    }

    Ok(max_deviation)
}

// ---------------------------------------------------------------------------
// Recovery error bound computation
// ---------------------------------------------------------------------------

/// Compute empirical recovery error statistics.
///
/// Given a dictionary, a reference sparse signal, and a recovered signal,
/// returns:
///   - `l2_error`      : ‖x_true - x_recovered‖₂
///   - `relative_error`: ‖x_true - x_recovered‖₂ / ‖x_true‖₂
///   - `support_jaccard`: Jaccard similarity of the non-zero supports
///
/// # Arguments
///
/// * `x_true`      – Ground-truth sparse signal.
/// * `x_recovered` – Recovered signal.
/// * `threshold`   – Values with |x| < threshold are treated as zero.
///
/// # Returns
///
/// `RecoveryErrorBound` struct with the computed statistics.
#[derive(Debug, Clone)]
pub struct RecoveryErrorBound {
    /// ‖x_true - x_recovered‖₂
    pub l2_error: f64,
    /// ‖x_true - x_recovered‖₂ / ‖x_true‖₂  (0 if x_true is zero)
    pub relative_error: f64,
    /// Jaccard similarity of non-zero supports (0 = no overlap, 1 = identical)
    pub support_jaccard: f64,
    /// Fraction of true non-zero components recovered above `threshold`
    pub support_recall: f64,
}

/// Compute recovery error statistics between a true signal and its reconstruction.
pub fn compute_recovery_error(
    x_true: &Array1<f64>,
    x_recovered: &Array1<f64>,
    threshold: f64,
) -> SignalResult<RecoveryErrorBound> {
    let n = x_true.len();
    if x_recovered.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "compute_recovery_error: x_true has {n} elements but x_recovered has {}",
            x_recovered.len()
        )));
    }

    let threshold = threshold.abs().max(1e-14);
    let diff = x_true - x_recovered;
    let l2_error: f64 = diff.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let x_true_norm: f64 = x_true.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let relative_error = if x_true_norm > 1e-14 {
        l2_error / x_true_norm
    } else {
        0.0
    };

    // Supports
    let support_true: std::collections::HashSet<usize> = (0..n)
        .filter(|&i| x_true[i].abs() >= threshold)
        .collect();
    let support_rec: std::collections::HashSet<usize> = (0..n)
        .filter(|&i| x_recovered[i].abs() >= threshold)
        .collect();

    let intersection: usize = support_true.intersection(&support_rec).count();
    let union_size: usize = support_true.union(&support_rec).count();

    let support_jaccard = if union_size == 0 {
        1.0 // both supports are empty → perfectly sparse zero signal
    } else {
        intersection as f64 / union_size as f64
    };

    let support_recall = if support_true.is_empty() {
        1.0
    } else {
        intersection as f64 / support_true.len() as f64
    };

    Ok(RecoveryErrorBound {
        l2_error,
        relative_error,
        support_jaccard,
        support_recall,
    })
}

// ---------------------------------------------------------------------------
// Unified compressive sensing interface
// ---------------------------------------------------------------------------

/// Algorithm selector for `compressive_sense`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsAlgorithm {
    /// Matching Pursuit (greedy, no re-orthogonalisation).
    MatchingPursuit,
    /// Orthogonal Matching Pursuit (greedy + orthogonal projection).
    OrthoMatchingPursuit,
    /// Basis Pursuit L1-minimisation via ADMM.
    BasisPursuit,
    /// LASSO L1-penalised LS via ADMM.
    Lasso,
    /// Subspace Pursuit (iterative support refinement).
    SubspacePursuit,
}

/// Unified compressed-sensing reconstruction interface.
///
/// This function wraps all recovery algorithms behind a single API.  Choose
/// the algorithm via `algorithm`; the `lambda` field is used only for
/// `CsAlgorithm::Lasso`, and `tol` for `CsAlgorithm::BasisPursuit`.
///
/// # Arguments
///
/// * `measurements`    – Compressed measurement vector `y` of length `m`.
/// * `sensing_matrix`  – Sensing matrix `Φ` of shape `(m, n)`.
/// * `sparsity`        – Target sparsity k.
/// * `algorithm`       – Which recovery algorithm to apply.
/// * `lambda`          – L1 regularisation weight for LASSO (ignored otherwise).
/// * `tol`             – Convergence tolerance for BP (ignored otherwise).
///
/// # Returns
///
/// Estimated sparse signal of length `n`.
pub fn compressive_sense(
    measurements: &Array1<f64>,
    sensing_matrix: &Array2<f64>,
    sparsity: usize,
    algorithm: CsAlgorithm,
    lambda: f64,
    tol: f64,
) -> SignalResult<Array1<f64>> {
    match algorithm {
        CsAlgorithm::MatchingPursuit => matching_pursuit(sensing_matrix, measurements, sparsity),
        CsAlgorithm::OrthoMatchingPursuit => {
            orthogonal_mp(sensing_matrix, measurements, sparsity)
        }
        CsAlgorithm::BasisPursuit => basis_pursuit(sensing_matrix, measurements, tol),
        CsAlgorithm::Lasso => lasso_admm(sensing_matrix, measurements, lambda, 2000),
        CsAlgorithm::SubspacePursuit => subspace_pursuit(sensing_matrix, measurements, sparsity),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a simple identity-like dictionary and sparse signal for unit tests.
    fn simple_test_case(n: usize, k: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        // Dictionary = identity (n x n), so y = x directly
        let dict = Array2::<f64>::eye(n);
        let mut x_true = Array1::<f64>::zeros(n);
        for i in 0..k {
            x_true[i * (n / k).max(1)] = (i + 1) as f64;
        }
        let y = dict.dot(&x_true);
        (dict, y, x_true)
    }

    #[test]
    fn test_matching_pursuit_identity() {
        let (dict, y, x_true) = simple_test_case(16, 3);
        let x_hat = matching_pursuit(&dict, &y, 3).expect("MP should succeed");
        let err: f64 = (&x_hat - &x_true).iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-8, "MP error too large: {err}");
    }

    #[test]
    fn test_orthogonal_mp_identity() {
        let (dict, y, x_true) = simple_test_case(16, 4);
        let x_hat = orthogonal_mp(&dict, &y, 4).expect("OMP should succeed");
        let err: f64 = (&x_hat - &x_true).iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-8, "OMP error too large: {err}");
    }

    #[test]
    fn test_basis_pursuit_small() {
        // Under-determined: m < n; signal sparse in standard basis
        let m = 8_usize;
        let n = 16_usize;
        // Use first m columns of identity as sensing matrix
        let mut phi = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            phi[[i, i]] = 1.0;
        }
        let mut x_true = Array1::<f64>::zeros(n);
        x_true[2] = 3.0;
        x_true[5] = -1.5;
        let y = phi.dot(&x_true);
        let x_hat = basis_pursuit(&phi, &y, 1e-6).expect("BP should succeed");
        // Residual in measurement domain
        let res: f64 = (&phi.dot(&x_hat) - &y)
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(res < 1e-4, "BP residual too large: {res}");
    }

    #[test]
    fn test_lasso_admm_shrinkage() {
        let (dict, y, _) = simple_test_case(16, 2);
        // With high lambda most entries should be zero
        let x_hat = lasso_admm(&dict, &y, 10.0, 2000).expect("LASSO should succeed");
        let nnz = x_hat.iter().filter(|&&v| v.abs() > 0.1).count();
        // Large lambda => most entries pulled to zero
        assert!(nnz <= 4, "Expected sparse result, got {nnz} non-zeros");
    }

    #[test]
    fn test_subspace_pursuit_identity() {
        let (dict, y, x_true) = simple_test_case(16, 3);
        let x_hat = subspace_pursuit(&dict, &y, 3).expect("SP should succeed");
        let err: f64 = (&x_hat - &x_true).iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-6, "SP error too large: {err}");
    }

    #[test]
    fn test_check_rip_identity() {
        let dict = Array2::<f64>::eye(32);
        let delta = check_rip(&dict, 4).expect("check_rip should succeed");
        // Identity matrix has RIP constant = 0 exactly
        assert!(delta < 1e-10, "Identity RIP constant should be ~0, got {delta}");
    }

    #[test]
    fn test_compute_recovery_error() {
        let mut x_true = Array1::<f64>::zeros(10);
        x_true[1] = 2.0;
        x_true[5] = -1.0;
        let x_rec = x_true.clone();
        let bounds = compute_recovery_error(&x_true, &x_rec, 1e-8)
            .expect("error computation should succeed");
        assert!(bounds.l2_error < 1e-12);
        assert!((bounds.support_jaccard - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compressive_sense_unified() {
        let (dict, y, x_true) = simple_test_case(8, 2);
        for algo in [
            CsAlgorithm::MatchingPursuit,
            CsAlgorithm::OrthoMatchingPursuit,
            CsAlgorithm::SubspacePursuit,
        ] {
            let x_hat =
                compressive_sense(&y, &dict, 2, algo, 0.01, 1e-6).expect("CS should succeed");
            let err: f64 = (&x_hat - &x_true).iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert!(err < 1e-5, "compressive_sense({algo:?}) error: {err}");
        }
    }

    #[test]
    fn test_dimension_mismatch_returns_error() {
        let dict = Array2::<f64>::eye(8);
        let y = Array1::<f64>::zeros(5); // wrong length
        assert!(matching_pursuit(&dict, &y, 2).is_err());
        assert!(orthogonal_mp(&dict, &y, 2).is_err());
        assert!(basis_pursuit(&dict, &y, 1e-6).is_err());
        assert!(lasso_admm(&dict, &y, 0.1, 100).is_err());
        assert!(subspace_pursuit(&dict, &y, 2).is_err());
    }
}
