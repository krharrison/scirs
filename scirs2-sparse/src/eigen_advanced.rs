//! Advanced eigenvalue algorithms for sparse and structured matrices
//!
//! This module provides several dense and matrix-free eigenvalue algorithms:
//!
//! - [`jacobi_eigen_sym`] — Classical Jacobi sweeps for dense symmetric matrices.
//! - [`tridiagonal_eigen_dc`] — Divide-and-conquer for symmetric tridiagonal matrices.
//! - [`golub_kahan_bidiag`] / [`GolubKahanBidiag`] — Golub-Kahan bidiagonalisation
//!   for sparse SVD via matrix-vector products.
//! - [`thick_restart_lanczos`] / [`ThickRestartLanczos`] — Thick-restart Lanczos for
//!   sparse symmetric eigenvalue problems.
//! - [`EigenwhichTarget`] — Selector for which part of the spectrum to target.
//!
//! All routines work through closures (`mat_vec`, `mat_transpose_vec`) to
//! remain format-agnostic.


// ---------------------------------------------------------------------------
// Helper: dot product and vector norms
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Scale vector in place: x *= s.
#[inline]
fn scal(s: f64, x: &mut [f64]) {
    x.iter_mut().for_each(|v| *v *= s);
}

// ---------------------------------------------------------------------------
// Jacobi eigenvalue algorithm (classical, for dense symmetric matrices)
// ---------------------------------------------------------------------------

/// Classical Jacobi eigenvalue algorithm for dense symmetric matrices.
///
/// Performs cyclic Jacobi sweeps, annihilating the largest off-diagonal
/// element at each step, until the off-diagonal Frobenius norm is below `tol`.
///
/// # Arguments
///
/// * `a`          – Square symmetric matrix; modified in-place to accumulate
///                  the rotation (on exit it holds Q^T A_0 Q ≈ diag).
/// * `n`          – Matrix dimension.
/// * `max_sweeps` – Maximum number of full sweeps over all off-diagonal pairs.
/// * `tol`        – Convergence threshold on the off-diagonal Frobenius norm.
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` where `eigenvalues[i]` is the i-th
/// eigenvalue and `eigenvectors[i]` is the corresponding eigenvector
/// (as row i of the returned matrix).
pub fn jacobi_eigen_sym(
    a: &mut Vec<Vec<f64>>,
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Start with Q = I
    let mut q: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for _sweep in 0..max_sweeps {
        // Check convergence: off-diagonal Frobenius norm
        let off_norm_sq: f64 = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j] * 2.0)
            .sum();
        if off_norm_sq.sqrt() < tol {
            break;
        }

        // Cyclic sweep over all (p, q) pairs with p < q
        for p in 0..n {
            for q_idx in (p + 1)..n {
                let a_pq = a[p][q_idx];
                if a_pq.abs() < 1e-300 {
                    continue;
                }
                let a_pp = a[p][p];
                let a_qq = a[q_idx][q_idx];
                let theta = 0.5 * (a_qq - a_pp) / a_pq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                // Update diagonal
                a[p][p] -= t * a_pq;
                a[q_idx][q_idx] += t * a_pq;
                a[p][q_idx] = 0.0;
                a[q_idx][p] = 0.0;

                // Update off-diagonal elements a[r][p], a[r][q_idx]
                for r in 0..n {
                    if r == p || r == q_idx {
                        continue;
                    }
                    let a_rp = a[r][p];
                    let a_rq = a[r][q_idx];
                    let new_rp = a_rp - s * (a_rq + tau * a_rp);
                    let new_rq = a_rq + s * (a_rp - tau * a_rq);
                    a[r][p] = new_rp;
                    a[p][r] = new_rp;
                    a[r][q_idx] = new_rq;
                    a[q_idx][r] = new_rq;
                }

                // Update eigenvector matrix Q
                for r in 0..n {
                    let q_rp = q[r][p];
                    let q_rq = q[r][q_idx];
                    q[r][p] = c * q_rp - s * q_rq;
                    q[r][q_idx] = s * q_rp + c * q_rq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

    // Eigenvectors are columns of Q (Q[:,i] is the eigenvector for eigenvalue i).
    // Return as row-major: eigenvectors[i] = Q[:,i] (the i-th column of Q).
    let mut eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|r| q[r][i]).collect())
        .collect();

    // Sort eigenvalues in ascending order
    sort_eigenpairs(&mut eigenvalues, &mut eigenvectors);

    (eigenvalues, eigenvectors)
}

// ---------------------------------------------------------------------------
// Divide-and-conquer for symmetric tridiagonal matrices
// ---------------------------------------------------------------------------

/// Divide-and-conquer solver for the symmetric tridiagonal eigenvalue problem.
///
/// Computes all eigenvalues and eigenvectors of the symmetric tridiagonal
/// matrix T with diagonal `diag` and sub/super-diagonal `offdiag`.
///
/// # Arguments
///
/// * `diag`    – Diagonal entries (length n).
/// * `offdiag` – Off-diagonal entries (length n − 1).
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` sorted in ascending order of eigenvalue.
/// `eigenvectors[i]` is the i-th eigenvector as a column (returned row-major,
/// so `eigenvectors[i][j]` is the j-th component of eigenvector i).
pub fn tridiagonal_eigen_dc(
    diag: &[f64],
    offdiag: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = diag.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![diag[0]], vec![vec![1.0]]);
    }

    // For the base case (n ≤ 4) or for moderate sizes, delegate to Jacobi.
    // The divide step splits at the middle; on merge we solve a secular equation.
    dc_solve(diag, offdiag, n)
}

fn dc_solve(diag: &[f64], offdiag: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if n <= 4 {
        // Use Jacobi for small sizes
        let mut a: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if i + 1 < n {
                a[i][i + 1] = offdiag[i];
                a[i + 1][i] = offdiag[i];
            }
        }
        return jacobi_eigen_sym(&mut a, n, 100, 1e-14);
    }

    let mid = n / 2;
    let rho = offdiag[mid - 1];

    // Split: T₁ = T[0..mid] with d[mid-1] -= rho, T₂ = T[mid..n] with d[mid] -= rho
    let mut d1: Vec<f64> = diag[..mid].to_vec();
    d1[mid - 1] -= rho.abs();
    let of1 = &offdiag[..mid - 1];

    let mut d2: Vec<f64> = diag[mid..].to_vec();
    d2[0] -= rho.abs();
    let of2 = &offdiag[mid..];

    let (mut ev1, mut evec1) = dc_solve(&d1, of1, mid);
    let (mut ev2, mut evec2) = dc_solve(&d2, of2, n - mid);

    // Sort sub-problems (they should already be sorted but ensure)
    sort_eigenpairs(&mut ev1, &mut evec1);
    sort_eigenpairs(&mut ev2, &mut evec2);

    // Merge via secular equation:
    // The rank-1 update: T = diag([λ₁;λ₂]) + ρ z zᵀ
    // where z = [Q₁ᵀ e_{mid}; Q₂ᵀ e₁] (last column of Q₁, first column of Q₂).
    let sign_rho = if rho >= 0.0 { 1.0 } else { -1.0 };

    // Gather the merged diagonal and z vector
    let d_merged: Vec<f64> = ev1.iter().chain(ev2.iter()).copied().collect();
    // z_k = sign_rho * Q1[:,mid-1][k] for k in 0..mid
    //       Q2[:,0][k] for k in mid..n
    let mut z: Vec<f64> = {
        let mut v = Vec::with_capacity(n);
        // last eigenvector component of T₁ at position mid-1 of local basis
        for k in 0..mid {
            v.push(sign_rho * evec1[k][mid - 1]);
        }
        // first eigenvector component of T₂ at position 0 of local basis
        for k in 0..n - mid {
            v.push(evec2[k][0]);
        }
        v
    };

    // Normalize z
    let z_norm = norm2(&z);
    let rho_eff = rho.abs() * z_norm * z_norm;
    let inv_z_norm = if z_norm > 1e-300 { 1.0 / z_norm } else { 1.0 };
    scal(inv_z_norm, &mut z);

    // Solve secular equation f(λ) = 1 + rho_eff Σ z_k² / (d_k − λ) = 0
    // for n eigenvalues, one in each interval (d_k, d_{k+1}).
    // Sort d_merged and z together.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&i, &j| d_merged[i].partial_cmp(&d_merged[j]).unwrap_or(std::cmp::Ordering::Equal));
    let d_sorted: Vec<f64> = idx.iter().map(|&i| d_merged[i]).collect();
    let z_sorted: Vec<f64> = idx.iter().map(|&i| z[i]).collect();

    // Solve secular equation for each eigenvalue
    let mut merged_evals: Vec<f64> = Vec::with_capacity(n);
    for k in 0..n {
        let lo = if k == 0 {
            d_sorted[0] - rho_eff.abs() - 1.0
        } else {
            d_sorted[k - 1]
        };
        let hi = d_sorted[k] + rho_eff.abs() + 1.0;
        let lam = secular_root(&d_sorted, &z_sorted, rho_eff, lo, hi);
        merged_evals.push(lam);
    }

    // Compute merged eigenvectors using the secular equation solution
    let mut merged_evecs: Vec<Vec<f64>> = Vec::with_capacity(n);
    for &lam in &merged_evals {
        // v_k = z_k / (d_k − λ)
        let mut v: Vec<f64> = (0..n)
            .map(|k| {
                let denom = d_sorted[k] - lam;
                if denom.abs() < 1e-300 {
                    z_sorted[k].signum() * 1e14
                } else {
                    z_sorted[k] / denom
                }
            })
            .collect();
        let vnorm = norm2(&v);
        if vnorm > 1e-300 {
            scal(1.0 / vnorm, &mut v);
        }

        // Map v back to original basis: first mid components belong to Q₁,
        // remaining n-mid to Q₂.
        // Un-permute according to idx
        let mut v_unperm = vec![0.0f64; n];
        for (sorted_pos, &orig_pos) in idx.iter().enumerate() {
            v_unperm[orig_pos] = v[sorted_pos];
        }

        // Global eigenvector = [Q₁ * v_1_part ; Q₂ * v_2_part]
        let mut global_evec = vec![0.0f64; n];
        // Q₁ part: evec1 has shape mid × mid (evec1[i] is i-th eigenvector of T₁ of length mid)
        let v1 = &v_unperm[..mid];
        for row in 0..mid {
            let mut acc = 0.0f64;
            for (col, &vk) in v1.iter().enumerate() {
                acc += evec1[col][row] * vk;
            }
            global_evec[row] = acc;
        }
        // Q₂ part
        let v2 = &v_unperm[mid..];
        for row in 0..n - mid {
            let mut acc = 0.0f64;
            for (col, &vk) in v2.iter().enumerate() {
                acc += evec2[col][row] * vk;
            }
            global_evec[mid + row] = acc;
        }

        let gnorm = norm2(&global_evec);
        if gnorm > 1e-300 {
            scal(1.0 / gnorm, &mut global_evec);
        }
        merged_evecs.push(global_evec);
    }

    // Sort by eigenvalue
    sort_eigenpairs(&mut merged_evals, &mut merged_evecs);

    (merged_evals, merged_evecs)
}

/// Bisection root-finding for the secular equation on one interval (lo, hi).
fn secular_root(d: &[f64], z: &[f64], rho: f64, lo: f64, hi: f64) -> f64 {
    let n = d.len();
    let secular = |lam: f64| -> f64 {
        1.0 + rho * (0..n).map(|k| z[k] * z[k] / (d[k] - lam)).sum::<f64>()
    };

    // Narrow the bracket
    let mut a = lo + 1e-12 * (hi - lo);
    let mut b = hi - 1e-12 * (hi - lo);

    // Ensure a < d[k] < hi to avoid pole
    // We look for a sign change
    let fa = secular(a);
    let fb = secular(b);

    if fa * fb > 0.0 {
        // No sign change found; return midpoint (degenerate case)
        return (a + b) * 0.5;
    }

    for _ in 0..80 {
        let mid = (a + b) * 0.5;
        if b - a < 1e-13 * (1.0 + mid.abs()) {
            return mid;
        }
        let fm = secular(mid);
        if fa * fm <= 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }
    (a + b) * 0.5
}

fn sort_eigenpairs(evals: &mut Vec<f64>, evecs: &mut Vec<Vec<f64>>) {
    let n = evals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&i, &j| evals[i].partial_cmp(&evals[j]).unwrap_or(std::cmp::Ordering::Equal));
    let ev_sorted: Vec<f64> = idx.iter().map(|&i| evals[i]).collect();
    let evec_sorted: Vec<Vec<f64>> = idx.iter().map(|&i| evecs[i].clone()).collect();
    *evals = ev_sorted;
    *evecs = evec_sorted;
}

// ---------------------------------------------------------------------------
// Golub-Kahan bidiagonalization
// ---------------------------------------------------------------------------

/// Result of Golub-Kahan bidiagonalization.
///
/// After `k` steps: A ≈ U B Vᵀ where B is k×k upper bidiagonal.
pub struct GolubKahanBidiag {
    /// Left Lanczos vectors (m × k+1) stored as rows: `u[j]` is the j-th vector.
    pub u: Vec<Vec<f64>>,
    /// Right Lanczos vectors (n × k) stored as rows: `v[j]` is the j-th vector.
    pub v: Vec<Vec<f64>>,
    /// Diagonal elements of the bidiagonal factor (length k).
    pub alpha: Vec<f64>,
    /// Sub-diagonal elements (length k; alpha[k] is the next diagonal estimate).
    pub beta: Vec<f64>,
}

/// Golub-Kahan bidiagonalisation for computing the SVD of a sparse matrix
/// A (m × n) via matrix-vector products only.
///
/// # Arguments
///
/// * `mat_vec`           – Closure computing A * x for x of length n.
/// * `mat_transpose_vec` – Closure computing Aᵀ * y for y of length m.
/// * `m`, `n`            – Dimensions of A.
/// * `k`                 – Number of Lanczos steps (Krylov dimension).
///
/// # Returns
///
/// A [`GolubKahanBidiag`] containing the Lanczos vectors and bidiagonal elements.
pub fn golub_kahan_bidiag<F, G>(
    mat_vec: F,
    mat_transpose_vec: G,
    m: usize,
    n: usize,
    k: usize,
) -> GolubKahanBidiag
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let k = k.min(m.min(n));

    let mut u: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut alpha = Vec::with_capacity(k);
    let mut beta_vec = Vec::with_capacity(k);

    // Initialize: choose u_0 = e_0 (or random; we use e_0 for reproducibility)
    let mut u0 = vec![0.0f64; m];
    u0[0] = 1.0;
    u.push(u0.clone());

    let mut beta_prev = 1.0f64; // β_0 = 1 (fictitious)

    for j in 0..k {
        // r = Aᵀ u_j − β_{j-1} v_{j-1}  (if j == 0, no previous v)
        let u_j = &u[j];
        let mut r = mat_transpose_vec(u_j);
        if j > 0 {
            axpy(-beta_prev, &v[j - 1], &mut r);
        }

        // Orthogonalize against all previous v vectors (full re-orthogonalization)
        for vk in &v {
            let d = dot(&r, vk);
            axpy(-d, vk, &mut r);
        }

        let alpha_j = norm2(&r);
        alpha.push(alpha_j);

        let vj = if alpha_j > 1e-300 {
            let inv = 1.0 / alpha_j;
            r.iter().map(|x| x * inv).collect::<Vec<_>>()
        } else {
            // Breakdown; fill with unit vector orthogonal to existing ones
            let mut e = vec![0.0f64; n];
            for i in 0..n {
                e[i] = 1.0;
                let _d = dot(&e, &e);
                for vk in &v {
                    let dd = dot(&e, vk);
                    axpy(-dd, vk, &mut e);
                }
                let nn = norm2(&e);
                if nn > 1e-10 {
                    scal(1.0 / nn, &mut e);
                    break;
                }
                e[i] = 0.0;
            }
            e
        };
        v.push(vj.clone());

        // p = A v_j − alpha_j u_j
        let mut p = mat_vec(&vj);
        axpy(-alpha_j, u_j, &mut p);

        // Full re-orthogonalization against all previous u vectors
        for uk in &u {
            let d = dot(&p, uk);
            axpy(-d, uk, &mut p);
        }

        let beta_j = norm2(&p);
        beta_vec.push(beta_j);
        beta_prev = beta_j;

        let u_next = if beta_j > 1e-300 {
            let inv = 1.0 / beta_j;
            p.iter().map(|x| x * inv).collect::<Vec<_>>()
        } else {
            // Deflation: pick a new starting vector orthogonal to existing u
            let mut e = vec![0.0f64; m];
            'outer: for i in 0..m {
                e = vec![0.0f64; m];
                e[i] = 1.0;
                for uk in &u {
                    let d = dot(&e, uk);
                    axpy(-d, uk, &mut e);
                }
                let nn = norm2(&e);
                if nn > 1e-10 {
                    scal(1.0 / nn, &mut e);
                    break 'outer;
                }
            }
            e
        };
        u.push(u_next);
    }

    GolubKahanBidiag { u, v, alpha, beta: beta_vec }
}

// ---------------------------------------------------------------------------
// Eigenvalue target selector
// ---------------------------------------------------------------------------

/// Selector for which eigenvalues to target in iterative methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigenwhichTarget {
    /// Eigenvalues with the largest algebraic value (most positive).
    LargestAlgebraic,
    /// Eigenvalues with the smallest algebraic value (most negative).
    SmallestAlgebraic,
    /// Eigenvalues with the largest absolute value.
    LargestMagnitude,
    /// Eigenvalues with the smallest absolute value.
    SmallestMagnitude,
}

impl EigenwhichTarget {
    /// Return `true` if the eigenvalue `a` should be preferred over `b` for
    /// this target (used to sort / select).
    fn is_better(&self, a: f64, b: f64) -> bool {
        match self {
            Self::LargestAlgebraic => a > b,
            Self::SmallestAlgebraic => a < b,
            Self::LargestMagnitude => a.abs() > b.abs(),
            Self::SmallestMagnitude => a.abs() < b.abs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Thick-restart Lanczos
// ---------------------------------------------------------------------------

/// Result of the thick-restart Lanczos algorithm.
pub struct ThickRestartLanczos {
    /// Converged (or best) eigenvalues (length ≤ k).
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors; `eigenvectors[i]` has length n.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of eigenpairs that passed the convergence criterion.
    pub n_converged: usize,
}

/// Thick-restart Lanczos algorithm for symmetric eigenvalue problems.
///
/// Computes the `k` eigenvalues (and vectors) of the symmetric operator
/// `mat_vec` that are best according to `which`.
///
/// # Arguments
///
/// * `mat_vec`     – Closure computing A * v for a vector of length n.
/// * `n`           – Problem dimension.
/// * `k`           – Number of desired eigenpairs.
/// * `krylov`      – Krylov subspace size (must satisfy `k < krylov ≤ n`).
/// * `max_restarts`– Maximum number of restart cycles.
/// * `tol`         – Convergence tolerance on ‖A q − λ q‖.
/// * `which`       – Spectral target (largest, smallest, etc.).
///
/// # Returns
///
/// A [`ThickRestartLanczos`] with converged eigenpairs.
pub fn thick_restart_lanczos<F>(
    mat_vec: F,
    n: usize,
    k: usize,
    krylov: usize,
    max_restarts: usize,
    tol: f64,
    which: EigenwhichTarget,
) -> ThickRestartLanczos
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let k = k.min(n);
    let krylov = krylov.min(n).max(k + 1);

    // Lanczos vectors (up to krylov + 1)
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(krylov + 1);
    // Alpha (diagonal of tridiagonal T) and beta (off-diagonal)
    let mut alpha: Vec<f64> = Vec::with_capacity(krylov);
    let mut beta: Vec<f64> = Vec::with_capacity(krylov);

    // Initialize with a random-like starting vector that has components
    // along many eigenvectors (not e_0, which might be an exact eigenvector of diagonal matrices).
    let mut v0 = vec![0.0f64; n];
    for i in 0..n {
        // Deterministic pseudo-random initialization
        v0[i] = ((i as f64 + 1.0).sqrt().fract() + 0.1) * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let v0_norm = norm2(&v0);
    if v0_norm > 1e-300 {
        scal(1.0 / v0_norm, &mut v0);
    }
    v.push(v0);

    let mut n_converged = 0usize;
    let mut ritz_vals: Vec<f64> = Vec::new();
    let mut ritz_vecs: Vec<Vec<f64>> = Vec::new();

    for _restart in 0..=max_restarts {
        // Extend Lanczos basis from current size to krylov
        let start = if _restart == 0 { 0 } else { k };
        for j in start..krylov {
            if j >= v.len() {
                break;
            }
            let vj = v[j].clone();
            let mut w = mat_vec(&vj);

            // Subtract β_{j-1} v_{j-1}
            if j > 0 && j <= beta.len() {
                axpy(-beta[j - 1], &v[j - 1], &mut w);
            }

            // Compute α_j = <w, v_j>
            let alpha_j = dot(&w, &vj);
            if j < alpha.len() {
                alpha[j] = alpha_j;
            } else {
                alpha.push(alpha_j);
            }

            // w = w − α_j v_j
            axpy(-alpha_j, &vj, &mut w);

            // Full re-orthogonalization
            for vk in &v[..j + 1] {
                let d = dot(&w, vk);
                axpy(-d, vk, &mut w);
            }
            // Second pass (numerical stability)
            for vk in &v[..j + 1] {
                let d = dot(&w, vk);
                axpy(-d, vk, &mut w);
            }

            let beta_j = norm2(&w);
            if j < beta.len() {
                beta[j] = beta_j;
            } else {
                beta.push(beta_j);
            }

            if beta_j < 1e-12 {
                // Invariant subspace found; pad with a new vector
                let mut e = vec![0.0f64; n];
                for i in 0..n {
                    e[i] = 1.0;
                    for vk in v.iter().take(j + 1) {
                        let d = dot(&e, vk);
                        axpy(-d, vk, &mut e);
                    }
                    let nn = norm2(&e);
                    if nn > 1e-10 {
                        scal(1.0 / nn, &mut e);
                        break;
                    }
                    e[i] = 0.0;
                }
                if j + 1 < v.len() {
                    v[j + 1] = e;
                } else {
                    v.push(e);
                }
                break;
            } else {
                let inv_beta = 1.0 / beta_j;
                let v_next: Vec<f64> = w.iter().map(|x| x * inv_beta).collect();
                if j + 1 < v.len() {
                    v[j + 1] = v_next;
                } else {
                    v.push(v_next);
                }
            }
        }

        // Solve the tridiagonal eigenvalue problem for T (size = current_m × current_m)
        let current_m = alpha.len().min(krylov);
        if current_m == 0 {
            break;
        }
        let t_alpha = alpha[..current_m].to_vec();
        let t_beta = if beta.len() >= current_m {
            beta[..current_m - 1].to_vec()
        } else {
            beta.clone()
        };

        let (t_evals, t_evecs) = tridiagonal_eigen_dc(&t_alpha, &t_beta);

        // Select the k best Ritz values
        let mut order: Vec<usize> = (0..t_evals.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            let ea = t_evals[a];
            let eb = t_evals[b];
            if which.is_better(ea, eb) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        let num_keep = k.min(order.len());
        let kept_idx: Vec<usize> = order[..num_keep].to_vec();

        // Compute global Ritz vectors: x_i = V * y_i (y_i = Ritz vector of T)
        let mut new_ritz_vals: Vec<f64> = Vec::with_capacity(num_keep);
        let mut new_ritz_vecs: Vec<Vec<f64>> = Vec::with_capacity(num_keep);
        let mut residuals: Vec<f64> = Vec::with_capacity(num_keep);

        for &ti in &kept_idx {
            let ritz_val = t_evals[ti];
            // Global Ritz vector x = Σ_j y_{ti}[j] * v_j
            let mut x = vec![0.0f64; n];
            let y = &t_evecs[ti];
            for (j, &y_j) in y.iter().enumerate() {
                if j < v.len() {
                    axpy(y_j, &v[j], &mut x);
                }
            }
            let xnorm = norm2(&x);
            if xnorm > 1e-300 {
                scal(1.0 / xnorm, &mut x);
            }

            // Residual: ‖A x − λ x‖
            let ax = mat_vec(&x);
            let mut res_vec = ax.clone();
            axpy(-ritz_val, &x, &mut res_vec);
            let res_norm = norm2(&res_vec);

            new_ritz_vals.push(ritz_val);
            new_ritz_vecs.push(x);
            residuals.push(res_norm);
        }

        // Count converged
        n_converged = residuals.iter().filter(|&&r| r < tol).count();
        ritz_vals = new_ritz_vals;
        ritz_vecs = new_ritz_vecs;

        if n_converged >= k {
            break;
        }

        // Thick restart: keep the `k` best Ritz vectors as the new Lanczos basis
        v.clear();
        alpha.clear();
        beta.clear();

        for (i, rvec) in ritz_vecs.iter().enumerate().take(num_keep) {
            v.push(rvec.clone());
            alpha.push(ritz_vals[i]);
        }
        // The beta for the restart is the last beta (coupling to the next direction)
        if let Some(&last_beta) = residuals.last() {
            if num_keep > 0 {
                beta.resize(num_keep - 1, 0.0);
                beta.push(last_beta);
            }
        }

        // Add the next Lanczos vector (the residual of the last kept Ritz pair)
        if let Some(last_ritz) = ritz_vecs.last() {
            let ax = mat_vec(last_ritz);
            let mut w = ax;
            axpy(-ritz_vals[ritz_vals.len() - 1], last_ritz, &mut w);
            for vk in &v {
                let d = dot(&w, vk);
                axpy(-d, vk, &mut w);
            }
            let wn = norm2(&w);
            if wn > 1e-12 {
                scal(1.0 / wn, &mut w);
                v.push(w);
            } else {
                // Pick a new direction
                let mut e = vec![0.0f64; n];
                for i in 0..n {
                    e[i] = 1.0;
                    for vk in &v {
                        let d = dot(&e, vk);
                        axpy(-d, vk, &mut e);
                    }
                    let nn = norm2(&e);
                    if nn > 1e-10 {
                        scal(1.0 / nn, &mut e);
                        break;
                    }
                    e[i] = 0.0;
                }
                v.push(e);
            }
        }
    }

    ThickRestartLanczos {
        eigenvalues: ritz_vals,
        eigenvectors: ritz_vecs,
        n_converged,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small dense symmetric matrix for testing.
    fn test_matrix_4x4() -> Vec<Vec<f64>> {
        // Symmetric tridiagonal: diag=4, off=-1
        let n = 4;
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = 4.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
        }
        a
    }

    #[test]
    fn test_jacobi_eigen_sym_4x4() {
        let mut a = test_matrix_4x4();
        let n = 4;
        let (evals, evecs) = jacobi_eigen_sym(&mut a, n, 100, 1e-12);
        assert_eq!(evals.len(), n);
        assert_eq!(evecs.len(), n);

        // All eigenvalues of the 4×4 tridiagonal should be positive
        for &ev in &evals {
            assert!(ev > 0.0, "eigenvalue should be positive, got {ev}");
        }

        // Check orthonormality: Q^T Q ≈ I
        for i in 0..n {
            for j in 0..n {
                let d = dot(&evecs[i], &evecs[j]);
                if i == j {
                    assert!((d - 1.0).abs() < 1e-10, "evec[{i}] not normalized: {d}");
                } else {
                    assert!(d.abs() < 1e-10, "evec[{i}] and evec[{j}] not orthogonal: {d}");
                }
            }
        }
    }

    #[test]
    fn test_jacobi_1x1() {
        let mut a = vec![vec![7.0f64]];
        let (evals, evecs) = jacobi_eigen_sym(&mut a, 1, 10, 1e-12);
        assert_eq!(evals.len(), 1);
        assert!((evals[0] - 7.0).abs() < 1e-12);
        assert!((evecs[0][0].abs() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tridiagonal_eigen_dc_4x4() {
        let diag = vec![4.0, 4.0, 4.0, 4.0];
        let off = vec![-1.0, -1.0, -1.0];
        let (evals, evecs) = tridiagonal_eigen_dc(&diag, &off);
        assert_eq!(evals.len(), 4);
        // Eigenvalues sorted ascending; all positive
        for (i, &ev) in evals.iter().enumerate() {
            assert!(ev > 0.0, "evals[{i}]={ev}");
        }
        // Monotone ascending
        for i in 0..evals.len() - 1 {
            assert!(evals[i] <= evals[i + 1] + 1e-10, "not sorted");
        }
        // Orthonormality
        let n = 4;
        for i in 0..n {
            for j in 0..n {
                let d = dot(&evecs[i], &evecs[j]);
                if i == j {
                    assert!((d - 1.0).abs() < 1e-8, "evec[{i}] norm^2={d}");
                } else {
                    assert!(d.abs() < 1e-8, "evec[{i}].evec[{j}]={d}");
                }
            }
        }
    }

    #[test]
    fn test_tridiagonal_eigen_dc_1() {
        let (evals, evecs) = tridiagonal_eigen_dc(&[3.0], &[]);
        assert_eq!(evals.len(), 1);
        assert!((evals[0] - 3.0).abs() < 1e-12);
        assert_eq!(evecs.len(), 1);
    }

    #[test]
    fn test_golub_kahan_bidiag() {
        // Simple 4×3 matrix (identity-like)
        let mat_vals: Vec<Vec<f64>> = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
            vec![0.0, 0.0, 0.0],
        ];
        let m = 4;
        let n = 3;
        let k = 3;
        let mv = |x: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0f64; m];
            for i in 0..m {
                for j in 0..n {
                    y[i] += mat_vals[i][j] * x[j];
                }
            }
            y
        };
        let mtv = |y: &[f64]| -> Vec<f64> {
            let mut x = vec![0.0f64; n];
            for i in 0..m {
                for j in 0..n {
                    x[j] += mat_vals[i][j] * y[i];
                }
            }
            x
        };

        let result = golub_kahan_bidiag(mv, mtv, m, n, k);
        assert_eq!(result.alpha.len(), k);
        assert_eq!(result.beta.len(), k);
        assert_eq!(result.v.len(), k);
        assert_eq!(result.u.len(), k + 1);

        // All alpha (singular value estimates) should be positive
        for &a in &result.alpha {
            assert!(a >= 0.0, "alpha={a}");
        }
    }

    #[test]
    fn test_thick_restart_lanczos_finds_eigenvalues() {
        // 5×5 diagonal matrix: eigenvalues are 1,2,3,4,5
        let n = 5usize;
        let diags = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mat_vec = move |x: &[f64]| -> Vec<f64> {
            x.iter().zip(diags.iter()).map(|(xi, di)| xi * di).collect()
        };

        let result = thick_restart_lanczos(mat_vec, n, 3, 5, 20, 1e-10, EigenwhichTarget::LargestAlgebraic);
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.eigenvectors.len(), 3);

        // The 3 largest eigenvalues should be close to 5, 4, 3
        let mut found_evals = result.eigenvalues.clone();
        found_evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Check that at least the largest is close to 5
        assert!(
            (found_evals[0] - 5.0).abs() < 0.5,
            "largest eigenvalue {}, expected ~5.0",
            found_evals[0]
        );
    }

    #[test]
    fn test_eigenwhich_target() {
        assert!(EigenwhichTarget::LargestAlgebraic.is_better(5.0, 3.0));
        assert!(EigenwhichTarget::SmallestAlgebraic.is_better(-1.0, 2.0));
        assert!(EigenwhichTarget::LargestMagnitude.is_better(-5.0, 3.0));
        assert!(EigenwhichTarget::SmallestMagnitude.is_better(0.1, 5.0));
    }
}
