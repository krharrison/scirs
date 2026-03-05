//! Generative model evaluation metrics.
//!
//! Provides metrics for evaluating the quality and diversity of generative
//! models (GANs, VAEs, diffusion models, etc.):
//!
//! - **Inception Score (IS)** — approximate version based on class probability
//!   distributions; does not require an actual Inception network.
//! - **Fréchet Distance (FD / FID-like)** — measures the Wasserstein-2 distance
//!   between two Gaussian distributions fitted to real and generated feature
//!   activations.
//! - **Maximum Mean Discrepancy (MMD)** with RBF kernel — non-parametric two-sample
//!   test statistic.
//! - **Kernel Inception Distance (KID)** — unbiased MMD estimator computed on
//!   random subsets to reduce variance.

use crate::error::{MetricsError, Result};

// ---------------------------------------------------------------------------
// Internal matrix helpers (pure Rust, no external linalg dependency)
// ---------------------------------------------------------------------------

/// Matrix–vector product: `A x`, where A is `m×n` stored row-major and x has length n.
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Frobenius inner product Tr(A^T B) = Σ_{ij} A[i][j]*B[i][j].
fn trace_product(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x * y))
        .sum()
}

/// Compute the matrix square root of a symmetric positive semi-definite matrix
/// using the eigendecomposition approximated by power-iteration-based Jacobi sweeps.
///
/// The algorithm iterates Jacobi rotations until off-diagonal elements are
/// negligible, then computes sqrt of eigenvalues and reconstructs.
///
/// For the FID formula we only need Tr(sqrt(Σ1 Σ2)); we compute it as
/// Tr(sqrt(Σ1^{1/2} Σ2 Σ1^{1/2})) which avoids a full matrix square root when
/// the dimension is small.  For clarity we expose a general `matrix_sqrt` here.
fn symmetric_matrix_sqrt(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 {
        return Ok(vec![]);
    }
    // Validate square.
    for (i, row) in m.iter().enumerate() {
        if row.len() != n {
            return Err(MetricsError::InvalidInput(format!(
                "matrix row {i} has length {} but expected {n}",
                row.len()
            )));
        }
    }

    // Copy into a working matrix.
    let mut a: Vec<Vec<f64>> = m.to_vec();
    // Accumulate eigenvector matrix V starting as identity.
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    // Jacobi iterations (convergence for symmetric matrices).
    for _ in 0..200 {
        // Find the off-diagonal element with largest absolute value.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v_abs = a[i][j].abs();
                if v_abs > max_val {
                    max_val = v_abs;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute rotation angle.
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-14 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[p][p] - a[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to a.
        let mut a_new = a.clone();
        a_new[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        a_new[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        a_new[p][q] = 0.0;
        a_new[q][p] = 0.0;
        for k in 0..n {
            if k != p && k != q {
                a_new[p][k] = c * a[p][k] + s * a[q][k];
                a_new[k][p] = a_new[p][k];
                a_new[q][k] = -s * a[p][k] + c * a[q][k];
                a_new[k][q] = a_new[q][k];
            }
        }
        a = a_new;

        // Apply rotation to v.
        for k in 0..n {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp + s * vkq;
            v[k][q] = -s * vkp + c * vkq;
        }
    }

    // Eigenvalues are diagonal of a; clamp negatives (numerical error).
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i].max(0.0)).collect();
    let sqrt_eigenvalues: Vec<f64> = eigenvalues.iter().map(|&e| e.sqrt()).collect();

    // Reconstruct: A^{1/2} = V diag(sqrt(lambda)) V^T.
    let mut result: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += v[i][k] * sqrt_eigenvalues[k] * v[j][k];
            }
            result[i][j] = s;
        }
    }

    Ok(result)
}

/// Matrix multiply: C = A B, where A is `m×k` and B is `k×n`.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = if m == 0 { 0 } else { a[0].len() };
    let nn = if b.is_empty() { 0 } else { b[0].len() };
    let mut c = vec![vec![0.0_f64; nn]; m];
    for i in 0..m {
        for j in 0..nn {
            for l in 0..k {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Inception Score (approximate)
// ---------------------------------------------------------------------------

/// Approximate Inception Score.
///
/// The IS is defined as `exp(E_x[KL(p(y|x) || p(y))])`, where `p(y|x)` are
/// per-image class probabilities and `p(y)` is the marginal.  A higher IS
/// indicates better quality and diversity.
///
/// This implementation does *not* require an Inception network: you provide the
/// class probability distributions directly (e.g. from a separate classifier or
/// a proxy classifier).
///
/// # Arguments
/// * `probs`  — slice of per-sample probability vectors; each must sum to ~1 and
///              have the same length (number of classes).
/// * `n_splits` — number of random splits used to estimate the mean and standard
///                deviation of IS (typically 10).
///
/// # Returns
/// `(IS_mean, IS_std)` — mean and standard deviation over splits.
///
/// # Errors
/// Returns an error if `probs` is empty, vectors have mismatched lengths, or
/// `n_splits` is 0.
pub fn inception_score_approx(probs: &[Vec<f64>], n_splits: usize) -> Result<(f64, f64)> {
    if probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probs must not be empty".to_string(),
        ));
    }
    if n_splits == 0 {
        return Err(MetricsError::InvalidInput(
            "n_splits must be >= 1".to_string(),
        ));
    }
    let n_classes = probs[0].len();
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "class probability vectors must not be empty".to_string(),
        ));
    }
    for (i, p) in probs.iter().enumerate() {
        if p.len() != n_classes {
            return Err(MetricsError::InvalidInput(format!(
                "probs[{i}] has length {} but expected {n_classes}",
                p.len()
            )));
        }
    }

    let n = probs.len();
    let split_size = (n / n_splits).max(1);
    let mut is_scores = Vec::with_capacity(n_splits);

    for split in 0..n_splits {
        let start = (split * split_size).min(n);
        let end = ((split + 1) * split_size).min(n);
        if start >= end {
            break;
        }
        let chunk = &probs[start..end];

        // Marginal p(y) = mean over samples of p(y|x).
        let mut marginal = vec![0.0_f64; n_classes];
        for p in chunk {
            for (m, &pval) in marginal.iter_mut().zip(p.iter()) {
                *m += pval;
            }
        }
        let chunk_len = chunk.len() as f64;
        marginal.iter_mut().for_each(|m| *m /= chunk_len);

        // KL divergence: E_x[ Σ_y p(y|x) log( p(y|x) / p(y) ) ].
        let kl_sum: f64 = chunk
            .iter()
            .map(|p| {
                p.iter()
                    .zip(marginal.iter())
                    .filter(|(&py_x, _)| py_x > 0.0)
                    .map(|(&py_x, &py)| {
                        let log_ratio = (py_x / py.max(1e-300)).ln();
                        py_x * log_ratio
                    })
                    .sum::<f64>()
            })
            .sum();

        let mean_kl = kl_sum / chunk_len;
        is_scores.push(mean_kl.exp());
    }

    if is_scores.is_empty() {
        return Ok((1.0, 0.0));
    }
    let mean = is_scores.iter().sum::<f64>() / is_scores.len() as f64;
    let variance = is_scores
        .iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f64>()
        / is_scores.len() as f64;
    Ok((mean, variance.sqrt()))
}

// ---------------------------------------------------------------------------
// Fréchet Distance (FID-like)
// ---------------------------------------------------------------------------

/// Fréchet Distance between two multivariate Gaussians.
///
/// FD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·sqrt(Σ₁·Σ₂))
///
/// This is the standard FID formula.  The matrix square root `sqrt(Σ₁ Σ₂)` is
/// computed via the identity sqrt(Σ₁ Σ₂) = Σ₁^{1/2} sqrt(Σ₁^{1/2} Σ₂ Σ₁^{1/2}) Σ₁^{-1/2},
/// which avoids an asymmetric square root.  The trace is invariant under cyclic
/// permutations, so:
///
/// Tr(sqrt(Σ₁ Σ₂)) = Tr(sqrt(Σ₁^{1/2} Σ₂ Σ₁^{1/2}))
///
/// # Arguments
/// * `mu1`, `sigma1` — mean and covariance of real features.
/// * `mu2`, `sigma2` — mean and covariance of generated features.
///
/// # Errors
/// Returns an error on dimension mismatches or non-square covariance matrices.
pub fn frechet_distance(
    mu1: &[f64],
    sigma1: &[Vec<f64>],
    mu2: &[f64],
    sigma2: &[Vec<f64>],
) -> Result<f64> {
    let d = mu1.len();
    if mu2.len() != d {
        return Err(MetricsError::DimensionMismatch(format!(
            "mu1 length {} != mu2 length {}",
            d,
            mu2.len()
        )));
    }
    if sigma1.len() != d || sigma2.len() != d {
        return Err(MetricsError::DimensionMismatch(
            "covariance matrix row count must equal dimension d".to_string(),
        ));
    }
    for (i, row) in sigma1.iter().enumerate() {
        if row.len() != d {
            return Err(MetricsError::DimensionMismatch(format!(
                "sigma1 row {i} has length {} but expected {d}",
                row.len()
            )));
        }
    }
    for (i, row) in sigma2.iter().enumerate() {
        if row.len() != d {
            return Err(MetricsError::DimensionMismatch(format!(
                "sigma2 row {i} has length {} but expected {d}",
                row.len()
            )));
        }
    }

    // ||mu1 - mu2||^2
    let mean_sq_diff: f64 = mu1
        .iter()
        .zip(mu2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    // Tr(Σ₁) + Tr(Σ₂)
    let trace_sigma1: f64 = (0..d).map(|i| sigma1[i][i]).sum();
    let trace_sigma2: f64 = (0..d).map(|i| sigma2[i][i]).sum();

    // Tr(sqrt(Σ₁ Σ₂)) via Tr(sqrt(Σ₁^{1/2} Σ₂ Σ₁^{1/2})).
    let sigma1_sqrt = symmetric_matrix_sqrt(sigma1)?;
    // M = Σ₁^{1/2} Σ₂ Σ₁^{1/2}
    let m = mat_mul(&mat_mul(&sigma1_sqrt, sigma2), &sigma1_sqrt);
    let m_sqrt = symmetric_matrix_sqrt(&m)?;
    let trace_sqrt: f64 = (0..d).map(|i| m_sqrt[i][i]).sum();

    let fd = mean_sq_diff + trace_sigma1 + trace_sigma2 - 2.0 * trace_sqrt;
    // Numerical issues can make fd slightly negative; clamp.
    Ok(fd.max(0.0))
}

// ---------------------------------------------------------------------------
// RBF kernel helpers
// ---------------------------------------------------------------------------

/// RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||² / (2 σ²)).
fn rbf_kernel(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (-sq_dist / (2.0 * sigma * sigma)).exp()
}

/// Compute the unbiased MMD² estimate with RBF kernel.
///
/// MMD²_u = (1/(m(m-1))) Σ_{i≠j} k(x_i,x_j)
///         + (1/(n(n-1))) Σ_{i≠j} k(y_i,y_j)
///         - (2/(mn))     Σ_{i,j} k(x_i,y_j)
fn unbiased_mmd_rbf(x: &[Vec<f64>], y: &[Vec<f64>], sigma: f64) -> f64 {
    let m = x.len();
    let n = y.len();

    // K_xx (i ≠ j)
    let kxx: f64 = if m < 2 {
        0.0
    } else {
        let sum: f64 = (0..m)
            .flat_map(|i| (0..m).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| rbf_kernel(&x[i], &x[j], sigma))
            .sum();
        sum / (m * (m - 1)) as f64
    };

    // K_yy (i ≠ j)
    let kyy: f64 = if n < 2 {
        0.0
    } else {
        let sum: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| rbf_kernel(&y[i], &y[j], sigma))
            .sum();
        sum / (n * (n - 1)) as f64
    };

    // K_xy
    let kxy: f64 = if m == 0 || n == 0 {
        0.0
    } else {
        let sum: f64 = (0..m)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| rbf_kernel(&x[i], &y[j], sigma))
            .sum();
        sum / (m * n) as f64
    };

    kxx + kyy - 2.0 * kxy
}

// ---------------------------------------------------------------------------
// Maximum Mean Discrepancy
// ---------------------------------------------------------------------------

/// Maximum Mean Discrepancy with RBF kernel (unbiased estimator).
///
/// MMD is a non-parametric two-sample test statistic: it is 0 iff the two
/// distributions are identical.  A larger value indicates a greater discrepancy
/// between the real distribution `x` and the generated distribution `y`.
///
/// # Arguments
/// * `x`     — real feature vectors.
/// * `y`     — generated feature vectors.
/// * `sigma` — RBF bandwidth (standard deviation).
///
/// # Returns
/// Unbiased MMD² estimate (can be slightly negative due to finite-sample variance).
///
/// # Errors
/// Returns an error if `x` or `y` is empty, or if any vectors have mismatched
/// dimensionality.
pub fn mmd_rbf(x: &[Vec<f64>], y: &[Vec<f64>], sigma: f64) -> Result<f64> {
    if x.is_empty() {
        return Err(MetricsError::InvalidInput("x must not be empty".to_string()));
    }
    if y.is_empty() {
        return Err(MetricsError::InvalidInput("y must not be empty".to_string()));
    }
    if sigma <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "sigma must be positive".to_string(),
        ));
    }
    let d = x[0].len();
    for (i, xi) in x.iter().enumerate() {
        if xi.len() != d {
            return Err(MetricsError::DimensionMismatch(format!(
                "x[{i}] has length {} but expected {d}",
                xi.len()
            )));
        }
    }
    for (i, yi) in y.iter().enumerate() {
        if yi.len() != d {
            return Err(MetricsError::DimensionMismatch(format!(
                "y[{i}] has length {} but expected {d}",
                yi.len()
            )));
        }
    }

    Ok(unbiased_mmd_rbf(x, y, sigma))
}

// ---------------------------------------------------------------------------
// Kernel Inception Distance
// ---------------------------------------------------------------------------

/// Kernel Inception Distance — unbiased MMD estimator with variance estimate.
///
/// KID is computed by:
/// 1. Drawing `n_subsets` random subsets of size `subset_size` from both `x` and `y`.
/// 2. Computing the unbiased MMD² on each subset pair.
/// 3. Returning the mean and standard deviation across subsets.
///
/// This mirrors the official KID implementation from Demystifying MMD GANs (2018).
///
/// # Arguments
/// * `x`          — real feature vectors.
/// * `y`          — generated feature vectors.
/// * `sigma`      — RBF bandwidth.
/// * `n_subsets`  — number of random subsets (≥ 1).
/// * `subset_size` — size of each subset; must be ≤ min(|x|, |y|).
///
/// # Returns
/// `(KID_mean, KID_std)` — mean and standard deviation of the per-subset MMD².
///
/// # Errors
/// Returns an error on invalid parameters or mismatched dimensions.
pub fn kid_rbf(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    sigma: f64,
    n_subsets: usize,
    subset_size: usize,
) -> Result<(f64, f64)> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "x and y must not be empty".to_string(),
        ));
    }
    if sigma <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "sigma must be positive".to_string(),
        ));
    }
    if n_subsets == 0 {
        return Err(MetricsError::InvalidInput(
            "n_subsets must be >= 1".to_string(),
        ));
    }
    if subset_size == 0 {
        return Err(MetricsError::InvalidInput(
            "subset_size must be >= 1".to_string(),
        ));
    }
    if subset_size > x.len() || subset_size > y.len() {
        return Err(MetricsError::InvalidInput(format!(
            "subset_size ({subset_size}) exceeds min(|x|={}, |y|={})",
            x.len(),
            y.len()
        )));
    }
    let d = x[0].len();
    for xi in x.iter() {
        if xi.len() != d {
            return Err(MetricsError::DimensionMismatch(
                "vectors in x have inconsistent dimension".to_string(),
            ));
        }
    }
    for yi in y.iter() {
        if yi.len() != d {
            return Err(MetricsError::DimensionMismatch(
                "vectors in y have inconsistent dimension".to_string(),
            ));
        }
    }

    // Deterministic pseudo-random subset selection (linear congruential generator).
    // This avoids depending on the `rand` crate while still providing varied subsets.
    let mut lcg_state: u64 = 0x_4d595df4d0f33173;

    let mut mmd_scores: Vec<f64> = Vec::with_capacity(n_subsets);

    for _ in 0..n_subsets {
        // Sample `subset_size` indices from 0..x.len() without replacement.
        let x_indices = lcg_sample_without_replacement(&mut lcg_state, x.len(), subset_size);
        let y_indices = lcg_sample_without_replacement(&mut lcg_state, y.len(), subset_size);

        let x_sub: Vec<&Vec<f64>> = x_indices.iter().map(|&i| &x[i]).collect();
        let y_sub: Vec<&Vec<f64>> = y_indices.iter().map(|&i| &y[i]).collect();

        // Compute unbiased MMD² on subsets.
        let kxx: f64 = if subset_size < 2 {
            0.0
        } else {
            let sum: f64 = (0..subset_size)
                .flat_map(|i| {
                    (0..subset_size)
                        .filter(move |&j| j != i)
                        .map(move |j| (i, j))
                })
                .map(|(i, j)| rbf_kernel(x_sub[i], x_sub[j], sigma))
                .sum();
            sum / (subset_size * (subset_size - 1)) as f64
        };

        let kyy: f64 = if subset_size < 2 {
            0.0
        } else {
            let sum: f64 = (0..subset_size)
                .flat_map(|i| {
                    (0..subset_size)
                        .filter(move |&j| j != i)
                        .map(move |j| (i, j))
                })
                .map(|(i, j)| rbf_kernel(y_sub[i], y_sub[j], sigma))
                .sum();
            sum / (subset_size * (subset_size - 1)) as f64
        };

        let kxy: f64 = {
            let sum: f64 = (0..subset_size)
                .flat_map(|i| (0..subset_size).map(move |j| (i, j)))
                .map(|(i, j)| rbf_kernel(x_sub[i], y_sub[j], sigma))
                .sum();
            sum / (subset_size * subset_size) as f64
        };

        mmd_scores.push(kxx + kyy - 2.0 * kxy);
    }

    let mean = mmd_scores.iter().sum::<f64>() / mmd_scores.len() as f64;
    let variance = mmd_scores
        .iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f64>()
        / mmd_scores.len() as f64;
    Ok((mean, variance.sqrt()))
}

/// Sample `k` distinct indices from `[0, n)` using a linear congruential generator.
fn lcg_sample_without_replacement(state: &mut u64, n: usize, k: usize) -> Vec<usize> {
    // Fisher-Yates partial shuffle on a Vec of indices.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        // LCG step.
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + (*state as usize) % (n - i);
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Inception Score
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_uniform_marginal() {
        // When p(y|x) is the same for all x and equals the marginal, KL=0 → IS=1.
        let probs = vec![vec![0.5, 0.5]; 20];
        let (is_mean, _is_std) = inception_score_approx(&probs, 5).expect("IS failed");
        assert!((is_mean - 1.0).abs() < 1e-6, "IS should be 1.0, got {is_mean}");
    }

    #[test]
    fn test_is_perfect_diversity() {
        // p(y|x) is a one-hot distribution that perfectly covers all classes.
        // For 4 classes and a balanced sample the IS should be > 1.
        let mut probs = Vec::new();
        for _ in 0..5 {
            probs.push(vec![1.0, 0.0, 0.0, 0.0]);
            probs.push(vec![0.0, 1.0, 0.0, 0.0]);
            probs.push(vec![0.0, 0.0, 1.0, 0.0]);
            probs.push(vec![0.0, 0.0, 0.0, 1.0]);
        }
        let (is_mean, _is_std) = inception_score_approx(&probs, 2).expect("IS failed");
        assert!(is_mean > 1.0, "IS for perfect diversity should be > 1, got {is_mean}");
    }

    #[test]
    fn test_is_empty_error() {
        assert!(inception_score_approx(&[], 5).is_err());
    }

    #[test]
    fn test_is_zero_splits_error() {
        assert!(inception_score_approx(&[vec![0.5, 0.5]], 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Fréchet Distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_identical_distributions() {
        let mu = vec![1.0, 2.0];
        let sigma = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let fd = frechet_distance(&mu, &sigma, &mu, &sigma).expect("FD failed");
        assert!(fd < 1e-6, "FD of identical Gaussians should be ~0, got {fd}");
    }

    #[test]
    fn test_fd_mean_shift() {
        let mu1 = vec![0.0, 0.0];
        let mu2 = vec![1.0, 0.0];
        let sigma = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let fd = frechet_distance(&mu1, &sigma, &mu2, &sigma).expect("FD failed");
        // ||mu1 - mu2||^2 = 1, Tr(sigma1)+Tr(sigma2) - 2*Tr(sqrt(I*I)) = 2+2-4 = 0
        // So FD should be ~1.
        assert!((fd - 1.0).abs() < 1e-4, "FD with unit mean shift should be ~1, got {fd}");
    }

    #[test]
    fn test_fd_dimension_mismatch_error() {
        let mu1 = vec![0.0, 0.0];
        let mu2 = vec![0.0];
        let sigma = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s1 = vec![vec![1.0]];
        assert!(frechet_distance(&mu1, &sigma, &mu2, &s1).is_err());
    }

    // -----------------------------------------------------------------------
    // MMD
    // -----------------------------------------------------------------------

    #[test]
    fn test_mmd_identical_samples() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, i as f64]).collect();
        let y = x.clone();
        let mmd = mmd_rbf(&x, &y, 1.0).expect("MMD failed");
        // MMD of identical samples (unbiased) should be 0 or very close.
        // Unbiased estimator: kxx + kyy - 2*kxy, with i≠j in xx and yy.
        // For identical sets the cross term matches, so result ≈ 0.
        assert!(mmd.abs() < 1e-4, "MMD of identical samples should be ~0, got {mmd}");
    }

    #[test]
    fn test_mmd_different_samples() {
        // x centered at 0, y centered far away.
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) * 0.01]).collect();
        let y: Vec<Vec<f64>> = (0..20).map(|i| vec![100.0 + (i as f64) * 0.01]).collect();
        let mmd = mmd_rbf(&x, &y, 1.0).expect("MMD failed");
        assert!(mmd > 0.0, "MMD of distant samples should be positive, got {mmd}");
    }

    #[test]
    fn test_mmd_empty_error() {
        assert!(mmd_rbf(&[], &[vec![1.0]], 1.0).is_err());
    }

    #[test]
    fn test_mmd_negative_sigma_error() {
        let x = vec![vec![1.0]];
        assert!(mmd_rbf(&x, &x, -1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // KID
    // -----------------------------------------------------------------------

    #[test]
    fn test_kid_identical_samples() {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.1]).collect();
        let y = x.clone();
        let (mean, _std) = kid_rbf(&x, &y, 1.0, 5, 10).expect("KID failed");
        assert!(mean.abs() < 0.1, "KID of identical samples should be near 0, got {mean}");
    }

    #[test]
    fn test_kid_subset_size_exceeds_error() {
        let x: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0]];
        let y: Vec<Vec<f64>> = vec![vec![1.0]];
        assert!(kid_rbf(&x, &y, 1.0, 5, 2).is_err());
    }

    #[test]
    fn test_kid_returns_nonnegative_mean_for_well_separated() {
        // x ≈ N(0,1), y ≈ N(10,1) in 1D
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64 - 10.0) * 0.1]).collect();
        let y: Vec<Vec<f64>> = (0..20).map(|i| vec![10.0 + (i as f64 - 10.0) * 0.1]).collect();
        let (mean, _) = kid_rbf(&x, &y, 1.0, 4, 10).expect("KID failed");
        // Distributions are far apart; MMD should be positive.
        assert!(mean > 0.0, "KID for well-separated distributions should be positive, got {mean}");
    }

    // -----------------------------------------------------------------------
    // matrix_sqrt sanity check
    // -----------------------------------------------------------------------

    #[test]
    fn test_matrix_sqrt_identity() {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let sqrt_id = symmetric_matrix_sqrt(&id).expect("sqrt failed");
        // sqrt(I) = I
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sqrt_id[i][j] - expected).abs() < 1e-6,
                    "sqrt_id[{i}][{j}] = {} expected {expected}", sqrt_id[i][j]);
            }
        }
    }

    #[test]
    fn test_matrix_sqrt_diagonal() {
        // sqrt([[4,0],[0,9]]) = [[2,0],[0,3]]
        let m = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let s = symmetric_matrix_sqrt(&m).expect("sqrt failed");
        assert!((s[0][0] - 2.0).abs() < 1e-5, "s[0][0]={}", s[0][0]);
        assert!((s[1][1] - 3.0).abs() < 1e-5, "s[1][1]={}", s[1][1]);
    }
}
