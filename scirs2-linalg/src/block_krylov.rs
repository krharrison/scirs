//! Block Krylov subspace methods and subspace iteration with deflation.
//!
//! This module implements:
//! - **Block Lanczos** for finding the top-k eigenvalues/eigenvectors of symmetric matrices.
//!   Uses a block three-term recurrence to build a block-tridiagonal projected matrix,
//!   then extracts Ritz pairs.
//! - **Subspace iteration with deflation** for finding the dominant eigenspace of
//!   a symmetric matrix, with converged eigenvalues deflated to improve remaining convergence.
//!
//! # References
//!
//! - Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations*, 4th ed. Chapter 9.
//! - Saad, Y. (2011). *Numerical Methods for Large Eigenvalue Problems*, 2nd ed.

use crate::error::{LinalgError, LinalgResult};

/// Configuration for Block Krylov eigensolvers.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BlockKrylovConfig {
    /// Number of vectors in each block (block size), default 4.
    pub block_size: usize,
    /// Number of block Krylov steps (Lanczos steps), default 10.
    pub n_steps: usize,
    /// Convergence tolerance for residuals, default 1e-10.
    pub tol: f64,
    /// Number of eigenvalues to find, default 6.
    pub n_eigenvalues: usize,
    /// Maximum number of restarts (for restarted variants), default 5.
    pub max_restarts: usize,
}

impl Default for BlockKrylovConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            n_steps: 10,
            tol: 1e-10,
            n_eigenvalues: 6,
            max_restarts: 5,
        }
    }
}

/// Result of a Block Krylov eigensolver.
#[derive(Debug, Clone)]
pub struct BlockKrylovResult {
    /// Computed eigenvalues (sorted in descending order of magnitude).
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors stored as columns (each entry is one eigenvector of length n).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Residual norms for each eigenpair: ‖A v_i - λ_i v_i‖.
    pub residuals: Vec<f64>,
    /// Total number of matrix-vector products performed.
    pub n_matvecs: usize,
    /// Whether each eigenpair converged to within the tolerance.
    pub converged: Vec<bool>,
}

/// Block Krylov and subspace iteration eigensolvers.
pub struct BlockKrylov;

impl BlockKrylov {
    /// Find the top-k eigenvalues/eigenvectors of a symmetric n×n matrix using Block Lanczos.
    ///
    /// # Arguments
    /// - `a`: symmetric n×n matrix stored in row-major order (length n*n).
    /// - `n`: matrix dimension.
    /// - `config`: algorithm parameters.
    ///
    /// # Returns
    /// [`BlockKrylovResult`] with up to `config.n_eigenvalues` eigenpairs.
    pub fn eigs_symmetric(
        a: &[f64],
        n: usize,
        config: &BlockKrylovConfig,
    ) -> LinalgResult<BlockKrylovResult> {
        if a.len() != n * n {
            return Err(LinalgError::ShapeError(format!(
                "expected {} elements, got {}",
                n * n,
                a.len()
            )));
        }
        if n == 0 {
            return Err(LinalgError::DimensionError("n must be > 0".into()));
        }

        let k = config.block_size.max(1);
        let m_steps = config.n_steps.max(1);
        let n_eig = config.n_eigenvalues.min(n);

        // We build a block Lanczos basis of size k * m_steps (at most n columns).
        let max_basis_cols = (k * m_steps).min(n);
        let actual_steps = max_basis_cols / k;

        let mut n_matvecs = 0usize;

        // Initialise V_0: k random orthonormal columns.
        let mut basis: Vec<Vec<f64>> = Self::random_orthonormal_block(n, k);

        // Block Lanczos recurrence storage.
        // alpha[j] is the j-th k×k diagonal block of T.
        // beta[j] is the j-th k×k off-diagonal block (below diagonal).
        let mut alpha_blocks: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut beta_blocks: Vec<Vec<Vec<f64>>> = Vec::new();

        // Full basis: collect all block columns as individual vectors.
        let mut full_basis: Vec<Vec<f64>> = Vec::new();

        let mut v_prev: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
        let mut v_curr = basis.clone();

        for step in 0..actual_steps {
            // W = A * V_curr  (block matvec: n×k)
            let mut w = Self::block_matvec(a, &v_curr, n);
            n_matvecs += k;

            // Subtract V_{step-1} * beta_prev^T
            if step > 0 {
                let beta_prev = &beta_blocks[step - 1];
                // w -= v_prev * beta_prev^T  (beta_prev is k×k)
                for col in 0..k {
                    for i in 0..n {
                        for j in 0..k {
                            w[col][i] -= v_prev[j][i] * beta_prev[j][col];
                        }
                    }
                }
            }

            // alpha_j = V_curr^T * W  (k×k)
            let alpha_j = Self::block_inner_product(&v_curr, &w, n);
            alpha_blocks.push(alpha_j.clone());

            // W̃ = W - V_curr * alpha_j
            for col in 0..k {
                for i in 0..n {
                    for j in 0..k {
                        w[col][i] -= v_curr[j][i] * alpha_j[j][col];
                    }
                }
            }

            // Re-orthogonalise W̃ against all previous basis vectors.
            {
                let prev_basis: Vec<Vec<f64>> = full_basis.clone();
                Self::block_gram_schmidt(&mut w, &prev_basis, n);
            }

            // Add v_curr to full basis.
            for vc in v_curr.iter().take(k) {
                full_basis.push(vc.clone());
            }

            if step + 1 < actual_steps {
                // QR decomposition of W̃ to get beta_{step+1} and V_{step+1}.
                let (q_block, r_block) = Self::block_qr(&w, n, k);
                beta_blocks.push(r_block);
                v_prev = v_curr;
                v_curr = q_block;
            }
        }

        // Now eigendecompose the block tridiagonal T_m.
        let (ritz_vals, ritz_vecs_t) =
            Self::block_tridiag_eig(&alpha_blocks, &beta_blocks, actual_steps, k);

        // Ritz vectors in original space: V_full * ritz_vecs_t
        let basis_cols = full_basis.len();
        let n_ritz = ritz_vals.len().min(n_eig);

        let mut eigenvalues = Vec::with_capacity(n_ritz);
        let mut eigenvectors = Vec::with_capacity(n_ritz);
        let mut residuals = Vec::with_capacity(n_ritz);
        let mut converged = Vec::with_capacity(n_ritz);

        for r in 0..n_ritz {
            let lam = ritz_vals[r];
            // Compute Ritz vector: sum over basis cols.
            let mut v = vec![0.0_f64; n];
            let coeff_col = &ritz_vecs_t[r]; // length = basis_cols
            let nc = coeff_col.len().min(basis_cols);
            for (j, &c) in coeff_col.iter().take(nc).enumerate() {
                for i in 0..n {
                    v[i] += full_basis[j][i] * c;
                }
            }
            // Normalise.
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-14 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }

            // Residual: ‖A*v - λ*v‖
            let av = Self::matvec(a, &v, n);
            let res: f64 = av
                .iter()
                .zip(v.iter())
                .map(|(avi, vi)| (avi - lam * vi).powi(2))
                .sum::<f64>()
                .sqrt();

            let conv = res < config.tol;
            eigenvalues.push(lam);
            eigenvectors.push(v);
            residuals.push(res);
            converged.push(conv);
        }

        Ok(BlockKrylovResult {
            eigenvalues,
            eigenvectors,
            residuals,
            n_matvecs,
            converged,
        })
    }

    /// Subspace iteration with deflation for symmetric matrices.
    ///
    /// Applies repeated power iteration on a k-dimensional subspace.
    /// Converged eigenpairs are deflated (Schur complement) so remaining
    /// vectors can converge faster.
    ///
    /// # Arguments
    /// - `a`: symmetric n×n row-major matrix.
    /// - `n`: matrix dimension.
    /// - `k`: number of eigenvalues to find.
    /// - `config`: algorithm parameters.
    pub fn subspace_iteration(
        a: &[f64],
        n: usize,
        k: usize,
        config: &BlockKrylovConfig,
    ) -> LinalgResult<BlockKrylovResult> {
        if a.len() != n * n {
            return Err(LinalgError::ShapeError(format!(
                "expected {} elements, got {}",
                n * n,
                a.len()
            )));
        }
        let k = k.min(n);
        if k == 0 {
            return Err(LinalgError::DimensionError("k must be > 0".into()));
        }

        let max_iter = config.n_steps * config.max_restarts;
        let mut n_matvecs = 0usize;

        // Initialise: random orthonormal basis Q (n×k stored as k column vectors).
        let mut q_cols = Self::random_orthonormal_block(n, k);

        let mut eigenvalues = vec![0.0_f64; k];
        let mut converged_flags = vec![false; k];
        let mut residuals = vec![f64::INFINITY; k];

        // A_deflated = A - sum of already-converged Rayleigh-Ritz pairs.
        // We handle deflation by keeping track of converged pairs and projecting them out.
        let mut conv_vecs: Vec<Vec<f64>> = Vec::new();
        let mut conv_vals: Vec<f64> = Vec::new();

        for _iter in 0..max_iter {
            // Z = A * Q  (apply deflation)
            let mut z_cols = Self::block_matvec(a, &q_cols, n);
            n_matvecs += k;

            // Deflate converged pairs: Z -= sum_i λ_i * (v_i * v_i^T) * Q
            for (cv, &clam) in conv_vecs.iter().zip(conv_vals.iter()) {
                for col in 0..k {
                    let dot: f64 = cv.iter().zip(q_cols[col].iter()).map(|(a, b)| a * b).sum();
                    for i in 0..n {
                        z_cols[col][i] -= clam * dot * cv[i];
                    }
                }
            }

            // Orthonormalise Z to get new Q.
            Self::block_gram_schmidt_inplace(&mut z_cols, n);
            q_cols = z_cols;

            // Compute Rayleigh quotients and residuals.
            let mut all_conv = true;
            for col in 0..k {
                if converged_flags[col] {
                    continue;
                }
                let v = &q_cols[col];
                let av = Self::matvec(a, v, n);
                let lam: f64 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();
                eigenvalues[col] = lam;

                let res: f64 = av
                    .iter()
                    .zip(v.iter())
                    .map(|(avi, vi)| (avi - lam * vi).powi(2))
                    .sum::<f64>()
                    .sqrt();
                residuals[col] = res;

                if res < config.tol {
                    converged_flags[col] = true;
                    conv_vecs.push(v.clone());
                    conv_vals.push(lam);
                } else {
                    all_conv = false;
                }
            }
            if all_conv {
                break;
            }
        }

        // Final residual pass for any remaining unconverged.
        for col in 0..k {
            if !converged_flags[col] {
                let v = &q_cols[col];
                let av = Self::matvec(a, v, n);
                let lam: f64 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();
                eigenvalues[col] = lam;
                let res: f64 = av
                    .iter()
                    .zip(v.iter())
                    .map(|(avi, vi)| (avi - lam * vi).powi(2))
                    .sum::<f64>()
                    .sqrt();
                residuals[col] = res;
                converged_flags[col] = res < config.tol;
            }
        }

        // Sort by eigenvalue (descending).
        let mut pairs: Vec<(f64, Vec<f64>, f64, bool)> = eigenvalues
            .into_iter()
            .zip(q_cols)
            .zip(residuals)
            .zip(converged_flags)
            .map(|(((lam, v), res), conv)| (lam, v, res, conv))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(BlockKrylovResult {
            eigenvalues: pairs.iter().map(|(l, _, _, _)| *l).collect(),
            eigenvectors: pairs.iter().map(|(_, v, _, _)| v.clone()).collect(),
            residuals: pairs.iter().map(|(_, _, r, _)| *r).collect(),
            n_matvecs,
            converged: pairs.iter().map(|(_, _, _, c)| *c).collect(),
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Dense matrix-vector product: y = A * x.
    fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0_f64; n];
        for (i, yi) in y.iter_mut().enumerate().take(n) {
            let row_start = i * n;
            let mut s = 0.0_f64;
            for j in 0..n {
                s += a[row_start + j] * x[j];
            }
            *yi = s;
        }
        y
    }

    /// Block matrix-vector product: C = A * B where B is a list of k column vectors.
    fn block_matvec(a: &[f64], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        b.iter().map(|col| Self::matvec(a, col, n)).collect()
    }

    /// Block inner product: M = U^T * V  (k_u × k_v matrix as Vec<Vec<f64>>).
    fn block_inner_product(u: &[Vec<f64>], v: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        let ku = u.len();
        let kv = v.len();
        let mut m = vec![vec![0.0_f64; kv]; ku];
        for i in 0..ku {
            for j in 0..kv {
                let dot: f64 = (0..n).map(|l| u[i][l] * v[j][l]).sum();
                m[i][j] = dot;
            }
        }
        m
    }

    /// Initialise a block of k orthonormal random vectors (pseudo-random, deterministic seed).
    fn random_orthonormal_block(n: usize, k: usize) -> Vec<Vec<f64>> {
        // Use a simple deterministic LCG to avoid external rand dependency.
        let mut state = 0x123456789abcdef0_u64;
        let lcg_next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (*s >> 33) as f64 / (u32::MAX as f64);
            bits * 2.0 - 1.0
        };

        let mut vecs: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..n).map(|_| lcg_next(&mut state)).collect())
            .collect();
        Self::block_gram_schmidt_inplace(&mut vecs, n);
        vecs
    }

    /// In-place block Gram-Schmidt orthonormalisation of `vecs`.
    fn block_gram_schmidt_inplace(vecs: &mut [Vec<f64>], n: usize) {
        let k = vecs.len();
        for i in 0..k {
            // Orthogonalise against previous vectors.
            for j in 0..i {
                // Safe: we split the borrow by index.
                let dot: f64 = (0..n).map(|l| vecs[j][l] * vecs[i][l]).sum();
                // We must avoid aliasing – copy the j-th vector.
                let vj: Vec<f64> = vecs[j].clone();
                for l in 0..n {
                    vecs[i][l] -= dot * vj[l];
                }
            }
            // Normalise.
            let norm = vecs[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-14 {
                for x in vecs[i].iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    /// Orthogonalise block `vecs` against external `basis` vectors, then normalise.
    fn block_gram_schmidt(vecs: &mut [Vec<f64>], basis: &[Vec<f64>], n: usize) {
        for vec in vecs.iter_mut() {
            // Two-pass classical Gram-Schmidt for numerical stability.
            for _ in 0..2 {
                for bv in basis.iter() {
                    let dot: f64 = (0..n).map(|l| bv[l] * vec[l]).sum();
                    for l in 0..n {
                        vec[l] -= dot * bv[l];
                    }
                }
            }
        }
        // Also orthonormalise within block.
        let k = vecs.len();
        for i in 0..k {
            // Orthogonalise against previous in block.
            for j in 0..i {
                let vj = vecs[j].clone();
                let dot: f64 = (0..n).map(|l| vj[l] * vecs[i][l]).sum();
                for l in 0..n {
                    vecs[i][l] -= dot * vj[l];
                }
            }
            let norm = vecs[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-14 {
                for x in vecs[i].iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    /// Thin QR of a block of column vectors (each of length n).
    /// Returns (Q_cols, R) where Q_cols are k orthonormal n-vectors and R is k×k upper triangular.
    fn block_qr(cols: &[Vec<f64>], n: usize, k: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut q = cols.to_vec();
        let mut r = vec![vec![0.0_f64; k]; k];

        for i in 0..k {
            // Orthogonalise against previous columns.
            for j in 0..i {
                let qj = q[j].clone();
                let dot: f64 = (0..n).map(|l| qj[l] * q[i][l]).sum();
                r[j][i] = dot;
                for l in 0..n {
                    q[i][l] -= dot * qj[l];
                }
            }
            // Norm (diagonal of R).
            let norm = q[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            r[i][i] = norm;
            if norm > 1e-14 {
                for x in q[i].iter_mut() {
                    *x /= norm;
                }
            }
        }
        (q, r)
    }

    /// Eigendecomposition of a block-tridiagonal symmetric matrix T of size (m*k × m*k).
    ///
    /// Returns (eigenvalues, eigenvectors_as_row_vectors) both length m*k, sorted descending.
    fn block_tridiag_eig(
        alpha: &[Vec<Vec<f64>>],
        beta: &[Vec<Vec<f64>>],
        m: usize,
        k: usize,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let dim = m * k;
        if dim == 0 {
            return (vec![], vec![]);
        }

        // Build the dense T matrix (dim × dim).
        let mut t = vec![vec![0.0_f64; dim]; dim];

        for step in 0..m {
            let row_off = step * k;
            let col_off = step * k;
            // Diagonal block alpha[step]: k×k.
            if step < alpha.len() {
                for r in 0..k {
                    for c in 0..k {
                        t[row_off + r][col_off + c] = alpha[step][r][c];
                    }
                }
            }
            // Sub-diagonal block beta[step]: k×k  (beta[step] is at row step+1, col step).
            if step + 1 < m && step < beta.len() {
                let r_off2 = (step + 1) * k;
                let c_off2 = step * k;
                for r in 0..k {
                    for c in 0..k {
                        t[r_off2 + r][c_off2 + c] = beta[step][r][c];
                        t[c_off2 + c][r_off2 + r] = beta[step][r][c]; // symmetric
                    }
                }
            }
        }

        // Symmetric QR iteration to find all eigenvalues/eigenvectors of T.
        Self::symmetric_qr_eig(&t, dim)
    }

    /// Symmetric QR eigendecomposition of a small dense symmetric matrix.
    /// Returns (eigenvalues, eigenvectors as list of row-vectors).
    fn symmetric_qr_eig(a: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        if n == 0 {
            return (vec![], vec![]);
        }
        if n == 1 {
            return (vec![a[0][0]], vec![vec![1.0]]);
        }

        // Start from a copy.
        let mut mat: Vec<Vec<f64>> = a.to_vec();
        // Accumulate eigenvectors in Q.
        let mut q_acc: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0_f64; n];
                row[i] = 1.0;
                row
            })
            .collect();

        // Tridiagonalise via Householder (if n > 2).
        Self::tridiagonalise(&mut mat, &mut q_acc, n);

        // QR iteration on tridiagonal.
        let max_iter = 30 * n;
        let eps = 1e-13_f64;

        for _ in 0..max_iter {
            // Check for deflation: zero out sub-diagonals below eps.
            let mut p = 0; // start of active sub-matrix
            while p < n {
                let mut q_end = p;
                while q_end + 1 < n
                    && mat[q_end + 1][q_end].abs()
                        > eps * (mat[q_end][q_end].abs() + mat[q_end + 1][q_end + 1].abs())
                {
                    q_end += 1;
                }
                if q_end == p {
                    p += 1;
                    continue;
                }
                // Apply QR step to submatrix [p..=q_end].
                let sz = q_end - p + 1;
                // Wilkinson shift.
                let d = (mat[q_end - 1][q_end - 1] - mat[q_end][q_end]) / 2.0;
                let sign_d = if d >= 0.0 { 1.0 } else { -1.0 };
                let mu = mat[q_end][q_end]
                    - sign_d * mat[q_end][q_end - 1].powi(2)
                        / (d.abs() + (d * d + mat[q_end][q_end - 1].powi(2)).sqrt());

                // Francis implicit QR step on submatrix.
                Self::implicit_qr_step(&mut mat, &mut q_acc, n, p, p + sz - 1, mu);
                break;
            }

            // Check convergence.
            let mut conv = true;
            for i in 1..n {
                if mat[i][i - 1].abs() > eps * (mat[i - 1][i - 1].abs() + mat[i][i].abs()) {
                    conv = false;
                    break;
                }
            }
            if conv {
                break;
            }
        }

        // Extract eigenvalues (diagonal).
        let mut eig_vals: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
        // Eigenvectors are columns of q_acc.
        let mut eig_vecs: Vec<Vec<f64>> = (0..n)
            .map(|col| (0..n).map(|row| q_acc[row][col]).collect())
            .collect();

        // Sort descending.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eig_vals[b]
                .partial_cmp(&eig_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_vals: Vec<f64> = indices.iter().map(|&i| eig_vals[i]).collect();
        let sorted_vecs: Vec<Vec<f64>> = indices.iter().map(|&i| eig_vecs[i].clone()).collect();
        eig_vals = sorted_vals;
        eig_vecs = sorted_vecs;

        (eig_vals, eig_vecs)
    }

    /// Householder tridiagonalisation of a symmetric matrix in-place.
    fn tridiagonalise(a: &mut [Vec<f64>], q: &mut [Vec<f64>], n: usize) {
        if n <= 2 {
            return;
        }
        for col in 0..n - 2 {
            // Build Householder reflector for column `col`, rows col+1..n.
            let mut x: Vec<f64> = (col + 1..n).map(|i| a[i][col]).collect();
            let xnorm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if xnorm < 1e-14 {
                continue;
            }
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            x[0] += sign * xnorm;
            let hlen = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if hlen < 1e-14 {
                continue;
            }
            for v in x.iter_mut() {
                *v /= hlen;
            }

            // Apply H = I - 2 v v^T from both sides to submatrix.
            // p = submatrix a[col+1..n][col+1..n]
            let sz = n - col - 1;
            // p_vec = A_sub * v
            let mut p_vec = vec![0.0_f64; sz];
            for i in 0..sz {
                for j in 0..sz {
                    p_vec[i] += a[col + 1 + i][col + 1 + j] * x[j];
                }
            }
            // K = v^T * p_vec
            let k_scalar: f64 = x.iter().zip(p_vec.iter()).map(|(a, b)| a * b).sum();
            // w = p_vec - K * v
            let mut w_vec: Vec<f64> = p_vec
                .iter()
                .zip(x.iter())
                .map(|(p, v)| p - k_scalar * v)
                .collect();

            // Update A_sub: A -= 2 * (v w^T + w v^T)
            for i in 0..sz {
                for j in 0..sz {
                    a[col + 1 + i][col + 1 + j] -= 2.0 * (x[i] * w_vec[j] + w_vec[i] * x[j]);
                }
            }
            // Update the col-th row/col.
            let new_val = -sign * xnorm;
            a[col + 1][col] = new_val;
            a[col][col + 1] = new_val;
            for ai in a.iter_mut().take(n).skip(col + 2) {
                ai[col] = 0.0;
            }
            for val in a[col].iter_mut().take(n).skip(col + 2) {
                *val = 0.0;
            }

            // Accumulate reflector into Q.
            // Q_sub (rows, columns col+1..n) update: Q -= 2 * (Q_sub * v) v^T
            let mut qv = vec![0.0_f64; n]; // Q * v (full rows)
            for i in 0..n {
                for j in 0..sz {
                    qv[i] += q[i][col + 1 + j] * x[j];
                }
            }
            for i in 0..n {
                for j in 0..sz {
                    q[i][col + 1 + j] -= 2.0 * qv[i] * x[j];
                }
            }
        }
    }

    /// Single implicit QR step with shift `mu` on the tridiagonal submatrix [lo..=hi].
    fn implicit_qr_step(
        a: &mut [Vec<f64>],
        q: &mut [Vec<f64>],
        n: usize,
        lo: usize,
        hi: usize,
        mu: f64,
    ) {
        // Shifted first element for bulge-chasing.
        let mut x = a[lo][lo] - mu;
        let mut z = a[lo + 1][lo];

        for k in lo..hi {
            // Givens rotation to eliminate a[k+1][k].
            let (c, s) = Self::givens(x, z);
            // Apply G from left and right to tridiagonal.
            // Left: rows k, k+1 of A.
            // Left: rows k, k+1 of A - need to handle borrow carefully
            {
                let (left, right) = a.split_at_mut(k + 1);
                let row_k = &mut left[k];
                let row_k1 = &mut right[0];
                for j in 0..n {
                    let ak = row_k[j];
                    let ak1 = row_k1[j];
                    row_k[j] = c * ak + s * ak1;
                    row_k1[j] = -s * ak + c * ak1;
                }
            }
            // Right: cols k, k+1 of A.
            for ai in a.iter_mut().take(n) {
                let aik = ai[k];
                let aik1 = ai[k + 1];
                ai[k] = c * aik + s * aik1;
                ai[k + 1] = -s * aik + c * aik1;
            }
            // Accumulate in Q (for eigenvectors): Q columns k, k+1.
            for qi in q.iter_mut().take(n) {
                let qik = qi[k];
                let qik1 = qi[k + 1];
                qi[k] = c * qik + s * qik1;
                qi[k + 1] = -s * qik + c * qik1;
            }
            if k + 1 < hi {
                x = a[k + 1][k];
                z = a[k + 2][k];
            }
        }
    }

    /// Compute Givens rotation coefficients (c, s) such that [c s; -s c]^T [a; b] = [r; 0].
    #[inline]
    fn givens(a: f64, b: f64) -> (f64, f64) {
        if b.abs() < 1e-15 {
            return (1.0, 0.0);
        }
        let r = a.hypot(b);
        (a / r, b / r)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a flat row-major n×n diagonal matrix from a slice of diagonal values.
    fn diag_matrix(diag: &[f64]) -> (Vec<f64>, usize) {
        let n = diag.len();
        let mut a = vec![0.0_f64; n * n];
        for (i, &d) in diag.iter().enumerate() {
            a[i * n + i] = d;
        }
        (a, n)
    }

    /// Symmetric tridiagonal matrix [main; sub].
    fn tridiag_sym(n: usize, main: f64, sub: f64) -> Vec<f64> {
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = main;
            if i + 1 < n {
                a[i * n + i + 1] = sub;
                a[(i + 1) * n + i] = sub;
            }
        }
        a
    }

    #[test]
    fn test_block_krylov_config_default() {
        let cfg = BlockKrylovConfig::default();
        assert_eq!(cfg.block_size, 4);
        assert_eq!(cfg.n_steps, 10);
        assert_eq!(cfg.n_eigenvalues, 6);
        assert_eq!(cfg.max_restarts, 5);
        assert!(cfg.tol < 1e-9);
    }

    #[test]
    fn test_block_krylov_diagonal_5x5() {
        // 5×5 diagonal matrix: eigenvalues 5,4,3,2,1.
        let diag_vals = [5.0_f64, 4.0, 3.0, 2.0, 1.0];
        let (a, n) = diag_matrix(&diag_vals);
        let cfg = BlockKrylovConfig {
            n_eigenvalues: 3,
            block_size: 2,
            n_steps: 8,
            tol: 1e-8,
            ..Default::default()
        };

        let res = BlockKrylov::eigs_symmetric(&a, n, &cfg).expect("Block Lanczos failed");
        assert!(!res.eigenvalues.is_empty());
        // Largest eigenvalue should be close to 5.
        assert_relative_eq!(res.eigenvalues[0], 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_block_krylov_symmetric_tridiag() {
        // 6×6 symmetric tridiagonal.
        let n = 6;
        let a = tridiag_sym(n, 4.0, -1.0);
        let cfg = BlockKrylovConfig {
            n_eigenvalues: 3,
            block_size: 2,
            n_steps: 10,
            tol: 1e-6,
            ..Default::default()
        };

        let res = BlockKrylov::eigs_symmetric(&a, n, &cfg).expect("Block Lanczos failed");
        assert!(!res.eigenvalues.is_empty());
        // Eigenvalues of this matrix are known: 4 + 2*cos(j*pi/(n+1)) for j=1..n.
        // Largest is ~ 4 + 2*cos(pi/7) ≈ 5.8.
        assert!(res.eigenvalues[0] > 4.0, "largest eigenvalue should be > 4");
    }

    #[test]
    fn test_block_krylov_residuals() {
        let diag_vals = [10.0_f64, 8.0, 5.0, 3.0, 1.0];
        let (a, n) = diag_matrix(&diag_vals);
        let cfg = BlockKrylovConfig {
            n_eigenvalues: 2,
            block_size: 1,
            n_steps: 15,
            tol: 1e-6,
            ..Default::default()
        };

        let res = BlockKrylov::eigs_symmetric(&a, n, &cfg).expect("Block Lanczos failed");
        // For converged pairs, residual < tol.
        for (i, (&res_norm, &conv)) in res.residuals.iter().zip(res.converged.iter()).enumerate() {
            if conv {
                assert!(
                    res_norm < cfg.tol * 10.0,
                    "eigenpair {i}: residual {res_norm} exceeds tol"
                );
            }
        }
    }

    #[test]
    fn test_block_krylov_block_size_1() {
        // Block size 1 = standard (single-vector) Lanczos.
        let diag_vals = [7.0_f64, 5.0, 3.0];
        let (a, n) = diag_matrix(&diag_vals);
        let cfg = BlockKrylovConfig {
            block_size: 1,
            n_eigenvalues: 2,
            n_steps: 10,
            tol: 1e-6,
            ..Default::default()
        };

        let res = BlockKrylov::eigs_symmetric(&a, n, &cfg).expect("Block Lanczos failed");
        assert!(!res.eigenvalues.is_empty());
        assert!(res.eigenvalues[0] > 4.0);
    }

    #[test]
    fn test_subspace_iteration_diagonal() {
        let diag_vals = [6.0_f64, 4.0, 2.0, 1.0];
        let (a, n) = diag_matrix(&diag_vals);
        let cfg = BlockKrylovConfig {
            n_steps: 30,
            max_restarts: 3,
            tol: 1e-5,
            ..Default::default()
        };

        let res =
            BlockKrylov::subspace_iteration(&a, n, 2, &cfg).expect("Subspace iteration failed");
        assert_eq!(res.eigenvalues.len(), 2);
        // Largest eigenvalue ≈ 6.
        assert_relative_eq!(res.eigenvalues[0], 6.0, epsilon = 0.5);
    }
}
