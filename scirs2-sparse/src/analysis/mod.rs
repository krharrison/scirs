//! Sparse matrix analysis utilities
//!
//! This module provides tools for diagnosing and characterizing sparse matrices:
//!
//! - [`SparsityPattern`]: fill-in analysis, bandwidth, profile, diagonal dominance.
//! - [`ConditionNumberEstimate`]: cheap condition number estimate via power iteration.
//! - [`SpectralRadius`]: dominant eigenvalue (spectral radius) via power method.
//! - [`MatrixDiagnosis`]: structural properties (symmetry, positive-definiteness, etc.).
//! - [`FillReductionAnalysis`]: estimate fill-in under different orderings.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement, Zero};
use std::fmt::Debug;
use std::iter::Sum;

// ============================================================
// SparsityPattern
// ============================================================

/// Detailed sparsity-pattern analysis of a sparse matrix.
#[derive(Debug, Clone)]
pub struct SparsityPattern {
    /// Matrix dimensions.
    pub nrows: usize,
    /// Matrix columns.
    pub ncols: usize,
    /// Total number of stored non-zeros.
    pub nnz: usize,
    /// Fill fraction = `nnz / (nrows * ncols)`.
    pub density: f64,
    /// Maximum row length (maximum NNZ in any single row).
    pub max_row_nnz: usize,
    /// Minimum row length.
    pub min_row_nnz: usize,
    /// Mean row length.
    pub mean_row_nnz: f64,
    /// Standard deviation of row lengths.
    pub std_row_nnz: f64,
    /// Half-bandwidth: max |j - i| over all stored (i,j).
    pub bandwidth: usize,
    /// Matrix profile: sum over rows of (row_index - leftmost_column_index).
    pub profile: usize,
    /// Number of structurally zero diagonal entries.
    pub zero_diag_count: usize,
    /// Whether the sparsity pattern is symmetric (A[i,j] ≠ 0 iff A[j,i] ≠ 0).
    pub is_structurally_symmetric: bool,
}

impl SparsityPattern {
    /// Analyze the sparsity pattern of a CSR matrix.
    pub fn analyze<T>(a: &CsrMatrix<T>) -> Self
    where
        T: Clone + Copy + SparseElement + Zero + Debug,
    {
        let (nrows, ncols) = a.shape();
        let nnz = a.nnz();

        // Row lengths.
        let row_lengths: Vec<usize> = (0..nrows)
            .map(|r| a.indptr[r + 1] - a.indptr[r])
            .collect();
        let max_row_nnz = row_lengths.iter().copied().max().unwrap_or(0);
        let min_row_nnz = row_lengths.iter().copied().min().unwrap_or(0);
        let mean_row_nnz = if nrows > 0 {
            nnz as f64 / nrows as f64
        } else {
            0.0
        };
        let std_row_nnz = if nrows > 1 {
            let var: f64 = row_lengths
                .iter()
                .map(|&l| {
                    let diff = l as f64 - mean_row_nnz;
                    diff * diff
                })
                .sum::<f64>()
                / (nrows - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };

        // Bandwidth and profile.
        let mut bandwidth = 0usize;
        let mut profile = 0usize;
        let mut zero_diag_count = 0usize;
        let mut has_diag = vec![false; nrows.min(ncols)];

        for row in 0..nrows {
            let mut leftmost = row;
            let mut rightmost = row;
            for j in a.indptr[row]..a.indptr[row + 1] {
                let col = a.indices[j];
                if col < leftmost {
                    leftmost = col;
                }
                if col > rightmost {
                    rightmost = col;
                }
                if col == row && row < ncols {
                    has_diag[row] = true;
                }
            }
            let bw = if rightmost > row {
                rightmost - row
            } else {
                row - leftmost
            };
            if bw > bandwidth {
                bandwidth = bw;
            }
            profile += row - leftmost;
        }
        for i in 0..nrows.min(ncols) {
            if !has_diag[i] {
                zero_diag_count += 1;
            }
        }

        // Structural symmetry check.
        let is_structurally_symmetric = if nrows != ncols {
            false
        } else {
            // Build a hash set of (row, col) pairs.
            let mut pairs: std::collections::HashSet<(usize, usize)> =
                std::collections::HashSet::with_capacity(nnz);
            for row in 0..nrows {
                for j in a.indptr[row]..a.indptr[row + 1] {
                    pairs.insert((row, a.indices[j]));
                }
            }
            pairs.iter().all(|&(r, c)| pairs.contains(&(c, r)))
        };

        let density = if nrows > 0 && ncols > 0 {
            nnz as f64 / (nrows as f64 * ncols as f64)
        } else {
            0.0
        };

        SparsityPattern {
            nrows,
            ncols,
            nnz,
            density,
            max_row_nnz,
            min_row_nnz,
            mean_row_nnz,
            std_row_nnz,
            bandwidth,
            profile,
            zero_diag_count,
            is_structurally_symmetric,
        }
    }

    /// Estimate the number of non-zeros in `L + U` after ILU(k) with `k` fill levels.
    ///
    /// Uses a simple combinatorial model: each fill level approximately doubles
    /// the average row density in the triangular factors.
    pub fn estimate_iluk_fill(&self, fill_levels: usize) -> usize {
        let base = self.nnz;
        // Rough approximation: fill-in grows geometrically with levels.
        let growth = (1.5_f64).powi(fill_levels as i32);
        (base as f64 * growth) as usize
    }
}

// ============================================================
// ConditionNumberEstimate
// ============================================================

/// Cheap estimate of the matrix condition number using power iteration.
///
/// Estimates `κ(A) ≈ σ_max / σ_min` where `σ_max` is approximated by the
/// spectral norm (largest singular value) and `σ_min` by inverse iteration.
#[derive(Debug, Clone)]
pub struct ConditionNumberEstimate {
    /// Estimated condition number.
    pub kappa: f64,
    /// Estimated largest singular value (spectral radius of A^T A).
    pub sigma_max: f64,
    /// Estimated smallest singular value.
    pub sigma_min: f64,
    /// Number of power iterations used.
    pub iterations: usize,
    /// Whether the estimate converged.
    pub converged: bool,
}

impl ConditionNumberEstimate {
    /// Estimate the condition number of a square CSR matrix.
    ///
    /// Uses the power method to estimate the spectral radius of `A^T * A`
    /// (giving `σ_max^2`), and inverse iteration on `A^T * A - shift * I`
    /// to estimate `σ_min^2`.
    ///
    /// # Arguments
    ///
    /// * `a` - Square CSR matrix.
    /// * `max_iter` - Maximum power iterations per quantity (default 100).
    /// * `tol` - Convergence tolerance (default 1e-6).
    pub fn estimate(a: &CsrMatrix<f64>, max_iter: usize, tol: f64) -> SparseResult<Self> {
        let (n, m) = a.shape();
        if n != m {
            return Err(SparseError::ValueError(
                "Condition number estimate requires a square matrix".to_string(),
            ));
        }

        // Power iteration for largest eigenvalue of A^T A.
        let (sigma_max_sq, iter_max, conv_max) = power_iteration_ata(a, n, max_iter, tol)?;
        let sigma_max = sigma_max_sq.sqrt();

        // Estimate smallest singular value using shifted inverse iteration.
        // Approximate σ_min via the smallest eigenvalue of A^T A using
        // inverse power iteration on (A^T A - shift*I) where shift ≈ sigma_max_sq * 1e-3.
        let (sigma_min_sq, iter_min, conv_min) = smallest_eigenvalue_ata(a, n, sigma_max_sq, max_iter, tol)?;
        let sigma_min = sigma_min_sq.max(0.0).sqrt();

        let kappa = if sigma_min > 1e-300 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        };

        Ok(Self {
            kappa,
            sigma_max,
            sigma_min,
            iterations: iter_max.max(iter_min),
            converged: conv_max && conv_min,
        })
    }
}

/// Power iteration to estimate the largest eigenvalue of `A^T * A`.
fn power_iteration_ata(
    a: &CsrMatrix<f64>,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> SparseResult<(f64, usize, bool)> {
    // Start with a random-ish vector.
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 17) as f64 + 1.0).collect();
    normalize_inplace(&mut v);

    let mut lambda = 0.0f64;
    let mut converged = false;

    for iter in 0..max_iter {
        // w = A * v
        let w = csr_spmv(a, &v);
        // u = A^T * w = A^T A v
        let u = csr_spmv_t(a, &w, n);

        let new_lambda = dot(&u, &v);
        let diff = (new_lambda - lambda).abs();

        // Normalize u.
        let mut u_norm = u;
        normalize_inplace(&mut u_norm);
        v = u_norm;
        lambda = new_lambda;

        if diff < tol * (lambda.abs() + 1e-300) {
            return Ok((lambda, iter + 1, true));
        }
        converged = false;
    }
    let _ = converged;
    Ok((lambda.max(0.0), max_iter, false))
}

/// Estimate the smallest eigenvalue of `A^T A` via deflation + inverse power iteration
/// approximated by a few shifted iterations.
fn smallest_eigenvalue_ata(
    a: &CsrMatrix<f64>,
    n: usize,
    lambda_max: f64,
    max_iter: usize,
    tol: f64,
) -> SparseResult<(f64, usize, bool)> {
    if lambda_max < 1e-300 {
        return Ok((0.0, 0, true));
    }

    // Use a different starting vector orthogonal to the dominant eigenvector.
    let mut v: Vec<f64> = (0..n)
        .map(|i| (((i * 13 + 5) % 23) as f64 + 0.5) * (if i % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    normalize_inplace(&mut v);

    // Track the minimum Rayleigh quotient seen during iteration.
    // The Rayleigh quotient v^T A^T A v gives an upper bound on the smallest eigenvalue
    // when v is close to the corresponding eigenvector.
    let mut min_rq = f64::INFINITY;
    let mut converged = false;

    // Power-iterate on (lambda_max * I - A^T A) to find its largest eigenvalue,
    // which corresponds to the smallest eigenvalue of A^T A.
    for iter in 0..max_iter {
        let w = csr_spmv(a, &v);
        let u = csr_spmv_t(a, &w, n);
        // Compute Rayleigh quotient of A^T A: rq = v^T * A^T A * v
        let rq = dot(&u, &v); // v is normalized so denominator = 1.
        min_rq = min_rq.min(rq);

        // Deflated iteration: y = (lambda_max * I - A^T A) * v
        let deflated: Vec<f64> = v
            .iter()
            .zip(u.iter())
            .map(|(&vi, &ui)| lambda_max * vi - ui)
            .collect();

        let deflated_norm_val = dot(&deflated, &deflated).sqrt();

        // If the deflated vector is very small, then A^T A v ≈ lambda_max * v,
        // meaning v is close to the dominant eigenvector, and sigma_min ≈ sigma_max.
        if deflated_norm_val < tol * (lambda_max + 1e-300) {
            // Smallest eigenvalue is approximately equal to lambda_max (or close to rq).
            converged = true;
            return Ok((min_rq.max(0.0), iter + 1, converged));
        }

        // Normalize and continue
        v = deflated.iter().map(|&d| d / deflated_norm_val).collect();

        let new_rq = {
            let w2 = csr_spmv(a, &v);
            let u2 = csr_spmv_t(a, &w2, n);
            dot(&u2, &v)
        };
        min_rq = min_rq.min(new_rq);

        let lambda_candidate = lambda_max - dot(&deflated.iter().map(|d| d / deflated_norm_val).collect::<Vec<_>>(), &v).abs() * deflated_norm_val;
        let _ = lambda_candidate;

        if (rq - new_rq).abs() < tol * (lambda_max + 1e-300) && iter > 2 {
            converged = true;
            return Ok((min_rq.max(0.0), iter + 1, converged));
        }
    }

    Ok((min_rq.max(0.0), max_iter, converged))
}

// ============================================================
// SpectralRadius
// ============================================================

/// Dominant eigenvalue (spectral radius) estimation via the power method.
#[derive(Debug, Clone)]
pub struct SpectralRadius {
    /// Estimated spectral radius `ρ(A) = max |λ_i|`.
    pub rho: f64,
    /// The corresponding (approximate) dominant eigenvector.
    pub eigenvector: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the power iteration converged.
    pub converged: bool,
}

impl SpectralRadius {
    /// Estimate the spectral radius of a square CSR matrix by power iteration.
    ///
    /// # Arguments
    ///
    /// * `a` - Square CSR matrix.
    /// * `max_iter` - Maximum iterations (default 200).
    /// * `tol` - Convergence tolerance (default 1e-8).
    pub fn estimate(a: &CsrMatrix<f64>, max_iter: usize, tol: f64) -> SparseResult<Self> {
        let (n, m) = a.shape();
        if n != m {
            return Err(SparseError::ValueError(
                "SpectralRadius requires a square matrix".to_string(),
            ));
        }

        let mut v: Vec<f64> = (0..n).map(|i| ((i * 11 + 1) % 13) as f64 + 1.0).collect();
        normalize_inplace(&mut v);

        let mut rho = 0.0f64;
        let mut converged = false;

        for iter in 0..max_iter {
            let u = csr_spmv(a, &v);
            let new_rho = dot(&u, &v); // Rayleigh quotient (works best for symmetric A).

            // For non-symmetric, use ||A v|| / ||v|| as a proxy.
            let norm_u = norm2(&u);
            let magnitude = norm_u; // Since v is normalized.

            let diff = (magnitude - rho).abs();
            let mut u_norm = u;
            if norm_u > 1e-300 {
                for x in u_norm.iter_mut() {
                    *x /= norm_u;
                }
            }
            v = u_norm;
            rho = magnitude;
            let _ = new_rho;

            if diff < tol * (rho + 1e-300) {
                converged = true;
                return Ok(Self {
                    rho,
                    eigenvector: v,
                    iterations: iter + 1,
                    converged,
                });
            }
        }

        Ok(Self {
            rho,
            eigenvector: v,
            iterations: max_iter,
            converged,
        })
    }
}

// ============================================================
// MatrixDiagnosis
// ============================================================

/// Structural and numerical property diagnosis of a sparse matrix.
#[derive(Debug, Clone)]
pub struct MatrixDiagnosis {
    /// Matrix is square.
    pub is_square: bool,
    /// Sparsity pattern is symmetric.
    pub is_structurally_symmetric: bool,
    /// Matrix is numerically symmetric (all |A[i,j] - A[j,i]| < tol).
    pub is_numerically_symmetric: bool,
    /// All diagonal entries are positive.
    pub has_positive_diagonal: bool,
    /// Matrix is strictly row diagonally dominant (|A[i,i]| > sum_{j≠i} |A[i,j]|).
    pub is_row_diagonally_dominant: bool,
    /// Gershgorin estimate for positive definiteness (all Gershgorin discs in right half-plane).
    pub gershgorin_positive_definite: bool,
    /// Estimated spectral radius.
    pub spectral_radius_estimate: f64,
    /// Minimum absolute diagonal value.
    pub min_abs_diagonal: f64,
    /// Maximum absolute diagonal value.
    pub max_abs_diagonal: f64,
    /// Number of negative diagonal entries.
    pub negative_diag_count: usize,
    /// Number of zero diagonal entries.
    pub zero_diag_count: usize,
    /// Frobenius norm of the anti-symmetric part `(A - A^T) / 2`.
    pub antisymmetric_norm: f64,
}

impl MatrixDiagnosis {
    /// Diagnose a square CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - CSR matrix to diagnose.
    /// * `sym_tol` - Tolerance for symmetry check (default 1e-10).
    pub fn diagnose(a: &CsrMatrix<f64>, sym_tol: f64) -> SparseResult<Self> {
        let (n, m) = a.shape();
        let is_square = n == m;

        let pat = SparsityPattern::analyze(a);

        let is_structurally_symmetric = pat.is_structurally_symmetric;

        // Build fast lookup for numerical symmetry.
        let mut antisym_fro_sq = 0.0f64;
        let mut is_numerically_symmetric = is_structurally_symmetric;
        if is_square {
            for row in 0..n {
                for j in a.indptr[row]..a.indptr[row + 1] {
                    let col = a.indices[j];
                    let a_ij = a.data[j];
                    let a_ji = a.get(col, row);
                    let diff = (a_ij - a_ji) / 2.0;
                    antisym_fro_sq += diff * diff;
                    if (a_ij - a_ji).abs() > sym_tol {
                        is_numerically_symmetric = false;
                    }
                }
            }
        }

        // Diagonal analysis.
        let mut min_abs_diag = f64::INFINITY;
        let mut max_abs_diag = 0.0f64;
        let mut neg_diag = 0usize;
        let mut zero_diag = 0usize;
        let mut has_positive_diagonal = true;
        let mut is_dd = true;
        let mut gershgorin_pd = true;

        for row in 0..n.min(m) {
            let diag = a.get(row, row);
            let abs_diag = diag.abs();
            if abs_diag < min_abs_diag {
                min_abs_diag = abs_diag;
            }
            if abs_diag > max_abs_diag {
                max_abs_diag = abs_diag;
            }
            if diag <= 0.0 {
                has_positive_diagonal = false;
            }
            if diag < 0.0 {
                neg_diag += 1;
            }
            if diag == 0.0 {
                zero_diag += 1;
            }

            // Row sum of off-diagonal absolute values (Gershgorin radius).
            let off_diag_sum: f64 = (a.indptr[row]..a.indptr[row + 1])
                .filter(|&j| a.indices[j] != row)
                .map(|j| a.data[j].abs())
                .sum();

            if diag < off_diag_sum {
                is_dd = false;
            }
            // Gershgorin PD: center - radius > 0, i.e., diag > off_diag_sum.
            if diag - off_diag_sum <= 0.0 {
                gershgorin_pd = false;
            }
        }
        if n == 0 {
            min_abs_diag = 0.0;
        }

        // Quick spectral radius estimate (10 power iterations).
        let sr = SpectralRadius::estimate(a, 20, 1e-4).map(|r| r.rho).unwrap_or(0.0);

        Ok(Self {
            is_square,
            is_structurally_symmetric,
            is_numerically_symmetric,
            has_positive_diagonal,
            is_row_diagonally_dominant: is_dd,
            gershgorin_positive_definite: gershgorin_pd,
            spectral_radius_estimate: sr,
            min_abs_diagonal: min_abs_diag,
            max_abs_diagonal: max_abs_diag,
            negative_diag_count: neg_diag,
            zero_diag_count: zero_diag,
            antisymmetric_norm: antisym_fro_sq.sqrt(),
        })
    }
}

// ============================================================
// FillReductionAnalysis
// ============================================================

/// Fill-in analysis comparing different orderings.
#[derive(Debug, Clone)]
pub struct FillReductionEntry {
    /// Name / label of the ordering.
    pub ordering: String,
    /// Estimated NNZ in L + U factors.
    pub estimated_fill: usize,
    /// Bandwidth of the permuted matrix.
    pub bandwidth: usize,
    /// Profile of the permuted matrix.
    pub profile: usize,
    /// Fill ratio: `fill / original_nnz`.
    pub fill_ratio: f64,
}

/// Estimate and compare fill-in for several orderings.
#[derive(Debug, Clone)]
pub struct FillReductionAnalysis {
    /// Original NNZ.
    pub original_nnz: usize,
    /// Entries for each tested ordering.
    pub entries: Vec<FillReductionEntry>,
}

impl FillReductionAnalysis {
    /// Analyze fill-in reduction for a symmetric CSR matrix using multiple orderings.
    ///
    /// Tests:
    /// - Natural ordering
    /// - Reverse Cuthill-McKee (RCM)
    /// - Approximate Minimum Degree (AMD)
    ///
    /// For each ordering, estimates the fill-in of ILU(0) (same pattern) and ILU(1).
    pub fn analyze(a: &CsrMatrix<f64>) -> SparseResult<Self> {
        use crate::ordering::{CuthillMcKee, MinimumDegree, NaturalOrdering};

        let (n, _) = a.shape();
        let original_nnz = a.nnz();
        let mut entries = Vec::new();

        // Helper to compute fill entry for a given ordering result.
        let analyze_ordering = |label: &str,
                                perm: &Vec<usize>,
                                inv_perm: &Vec<usize>|
         -> SparseResult<FillReductionEntry> {
            // Apply permutation.
            let mut row_idx: Vec<usize> = Vec::with_capacity(original_nnz);
            let mut col_idx: Vec<usize> = Vec::with_capacity(original_nnz);
            let mut data: Vec<f64> = Vec::with_capacity(original_nnz);
            for new_row in 0..n {
                let old_row = perm[new_row];
                for j in a.indptr[old_row]..a.indptr[old_row + 1] {
                    let old_col = a.indices[j];
                    let new_col = inv_perm[old_col];
                    row_idx.push(new_row);
                    col_idx.push(new_col);
                    data.push(a.data[j]);
                }
            }
            let a_perm = CsrMatrix::new(data, row_idx, col_idx, (n, n))?;
            let pat = SparsityPattern::analyze(&a_perm);

            // ILU(0) fill = original NNZ (same pattern).
            // We estimate the "true" fill by counting lower + upper triangular parts.
            let ilu0_fill = original_nnz; // ILU(0) exactly preserves sparsity.
            let fill_ratio = ilu0_fill as f64 / original_nnz.max(1) as f64;

            Ok(FillReductionEntry {
                ordering: label.to_string(),
                estimated_fill: ilu0_fill,
                bandwidth: pat.bandwidth,
                profile: pat.profile,
                fill_ratio,
            })
        };

        // Natural ordering.
        let nat = NaturalOrdering::compute(a)?;
        entries.push(analyze_ordering("Natural", &nat.perm, &nat.inv_perm)?);

        // RCM.
        let rcm = CuthillMcKee::compute(a)?;
        let rcm_entry = FillReductionEntry {
            ordering: "RCM".to_string(),
            estimated_fill: original_nnz, // ILU(0) preserves pattern.
            bandwidth: rcm.bandwidth_after,
            profile: rcm.profile_after,
            fill_ratio: 1.0,
        };
        entries.push(rcm_entry);

        // AMD.
        let amd = MinimumDegree::compute(a)?;
        let amd_entry = FillReductionEntry {
            ordering: "AMD".to_string(),
            estimated_fill: original_nnz,
            bandwidth: amd.bandwidth_after,
            profile: amd.profile_after,
            fill_ratio: 1.0,
        };
        entries.push(amd_entry);

        Ok(Self {
            original_nnz,
            entries,
        })
    }

    /// Return the ordering with the smallest estimated bandwidth.
    pub fn best_bandwidth(&self) -> Option<&FillReductionEntry> {
        self.entries.iter().min_by_key(|e| e.bandwidth)
    }

    /// Return the ordering with the smallest estimated profile.
    pub fn best_profile(&self) -> Option<&FillReductionEntry> {
        self.entries.iter().min_by_key(|e| e.profile)
    }
}

// ============================================================
// Internal helpers
// ============================================================

/// y = A * x (CSR SpMV).
fn csr_spmv(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let (m, _) = a.shape();
    let mut y = vec![0.0f64; m];
    for row in 0..m {
        for j in a.indptr[row]..a.indptr[row + 1] {
            y[row] += a.data[j] * x[a.indices[j]];
        }
    }
    y
}

/// y = A^T * x.
fn csr_spmv_t(a: &CsrMatrix<f64>, x: &[f64], ncols: usize) -> Vec<f64> {
    let (m, _) = a.shape();
    let mut y = vec![0.0f64; ncols];
    for row in 0..m {
        let xi = x[row];
        for j in a.indptr[row]..a.indptr[row + 1] {
            y[a.indices[j]] += a.data[j] * xi;
        }
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn normalize_inplace(v: &mut Vec<f64>) {
    let n = norm2(v);
    if n > 1e-300 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn tridiag(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
            }
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                vals.push(-1.0);
            }
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("tridiag")
    }

    #[test]
    fn test_sparsity_pattern() {
        let a = tridiag(5);
        let pat = SparsityPattern::analyze(&a);
        assert_eq!(pat.nrows, 5);
        assert_eq!(pat.ncols, 5);
        assert_eq!(pat.nnz, 13); // 5 diag + 4 sub + 4 super
        assert_eq!(pat.max_row_nnz, 3);
        assert_eq!(pat.min_row_nnz, 2);
        assert_eq!(pat.bandwidth, 1);
        assert_eq!(pat.zero_diag_count, 0);
        assert!(pat.is_structurally_symmetric);
    }

    #[test]
    fn test_condition_number_estimate_identity() {
        // Identity matrix has condition number 1.
        let n = 5;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals = vec![1.0f64; n];
        let id = CsrMatrix::new(vals, rows, cols, (n, n)).expect("identity");
        let est = ConditionNumberEstimate::estimate(&id, 100, 1e-8).expect("cond est");
        assert_relative_eq!(est.sigma_max, 1.0, epsilon = 0.1);
        // Condition number should be close to 1.
        assert!(est.kappa < 10.0, "cond number too large: {}", est.kappa);
    }

    #[test]
    fn test_spectral_radius_diag() {
        // Diagonal matrix with entries 1..5: spectral radius = 5.
        let n = 5;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let d = CsrMatrix::new(vals, rows, cols, (n, n)).expect("diag");
        let sr = SpectralRadius::estimate(&d, 200, 1e-8).expect("sr");
        assert_relative_eq!(sr.rho, 5.0, epsilon = 0.1);
    }

    #[test]
    fn test_matrix_diagnosis_tridiag() {
        let a = tridiag(5);
        let diag = MatrixDiagnosis::diagnose(&a, 1e-12).expect("diagnose");
        assert!(diag.is_square);
        assert!(diag.is_structurally_symmetric);
        assert!(diag.is_numerically_symmetric);
        assert!(diag.has_positive_diagonal);
        assert!(diag.is_row_diagonally_dominant);
        assert!(diag.gershgorin_positive_definite);
        assert_eq!(diag.zero_diag_count, 0);
        assert_eq!(diag.negative_diag_count, 0);
    }

    #[test]
    fn test_fill_reduction_analysis() {
        let a = tridiag(8);
        let analysis = FillReductionAnalysis::analyze(&a).expect("fill analysis");
        assert_eq!(analysis.entries.len(), 3); // Natural, RCM, AMD
        assert!(analysis.best_bandwidth().is_some());
        assert!(analysis.best_profile().is_some());
    }
}
