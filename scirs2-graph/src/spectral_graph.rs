//! Spectral graph theory — adjacency-matrix-based API.
//!
//! This module exposes pure-matrix versions of spectral algorithms that accept
//! `Array2<f64>` weighted adjacency matrices directly.  It complements the
//! typed-graph spectral module (`spectral.rs`) which operates on `Graph<N,E,Ix>`.
//!
//! ## Algorithms
//! - **Graph Laplacian** L = D − A
//! - **Normalized Laplacian** L̃ = D^{−1/2} L D^{−1/2}
//! - **Fiedler vector** (spectral bisection via inverse iteration)
//! - **Spectral clustering** (k smallest eigenvectors + Lloyd k-means)
//! - **Graph Fourier Transform** (project signal onto Laplacian eigenvectors)
//! - **Effective resistance** and resistance matrix (via pseudo-inverse)
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::spectral_graph::{graph_laplacian, fiedler_vector};
//!
//! let adj = Array2::<f64>::from_shape_vec((4,4), vec![
//!     0.,1.,0.,1., 1.,0.,1.,0., 0.,1.,0.,1., 1.,0.,1.,0.,
//! ]).unwrap();
//! let l = graph_laplacian(&adj);
//! let (lambda2, fv) = fiedler_vector(&adj).unwrap();
//! println!("Fiedler value = {lambda2}");
//! ```

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, SeedableRng, StdRng};

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Graph Laplacian  L = D − A
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the combinatorial graph Laplacian L = D − A.
///
/// `D` is the diagonal degree matrix with `D_{ii} = sum_j A_{ij}`.
/// The Laplacian is symmetric positive semi-definite; its smallest eigenvalue
/// is 0 (constant eigenvector) and the second-smallest (Fiedler value) measures
/// graph connectivity.
pub fn graph_laplacian(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let mut lap = Array2::zeros((n, n));
    for i in 0..n {
        let deg: f64 = adj.row(i).sum();
        lap[[i, i]] = deg;
        for j in 0..n {
            if i != j {
                lap[[i, j]] = -adj[[i, j]];
            }
        }
    }
    lap
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalized Laplacian  L̃ = D^{−1/2} L D^{−1/2}
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the symmetric normalized Laplacian L̃ = D^{−1/2} (D − A) D^{−1/2}.
///
/// Isolated nodes (zero degree) are left as zero rows/columns.
/// Eigenvalues lie in [0, 2]; the largest is exactly 2 iff the graph is bipartite.
pub fn normalized_laplacian(adj: &Array2<f64>) -> Result<Array2<f64>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph("adjacency matrix must be square".into()));
    }

    // D^{-1/2} diagonal
    let d_inv_sqrt: Vec<f64> = (0..n)
        .map(|i| {
            let deg = adj.row(i).sum();
            if deg > 0.0 {
                1.0 / deg.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    let l = graph_laplacian(adj);
    let mut l_norm = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            l_norm[[i, j]] = d_inv_sqrt[i] * l[[i, j]] * d_inv_sqrt[j];
        }
    }
    Ok(l_norm)
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigenvalue helpers (power iteration / inverse iteration)
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix-vector product: result_i = sum_j M[i,j] * v[j].
fn matvec(m: &Array2<f64>, v: &[f64]) -> Vec<f64> {
    let n = m.nrows();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            out[i] += m[[i, j]] * v[j];
        }
    }
    out
}

/// L2 norm of a slice.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize a slice in-place; returns the norm before normalization.
fn normalize_inplace(v: &mut Vec<f64>) -> f64 {
    let n = vec_norm(v);
    if n > 1e-300 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    n
}

/// Project `v` away from `basis` vectors (modified Gram-Schmidt deflation).
fn deflate(v: &mut Vec<f64>, basis: &[Vec<f64>]) {
    for b in basis {
        let dot: f64 = v.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
        for (vi, bi) in v.iter_mut().zip(b.iter()) {
            *vi -= dot * bi;
        }
    }
}

/// Power iteration to find the largest eigenvalue/vector of a symmetric matrix.
/// Returns `(eigenvalue, eigenvector)`.
fn power_iteration(m: &Array2<f64>, max_iter: usize, tol: f64, seed: u64) -> (f64, Vec<f64>) {
    let n = m.nrows();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
    normalize_inplace(&mut v);

    let mut lambda = 0.0f64;
    for _ in 0..max_iter {
        let mv = matvec(m, &v);
        let new_lambda: f64 = mv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let mut new_v = mv;
        normalize_inplace(&mut new_v);
        if (new_lambda - lambda).abs() < tol {
            return (new_lambda, new_v);
        }
        lambda = new_lambda;
        v = new_v;
    }
    (lambda, v)
}

/// Inverse iteration with shift `sigma` to find the eigenvalue/vector closest
/// to `sigma`.  Solves `(M - sigma I) v_{k+1} = v_k` using Gaussian elimination.
///
/// Returns `(eigenvalue, eigenvector)`.
fn inverse_iteration(
    m: &Array2<f64>,
    sigma: f64,
    prev_evecs: &[Vec<f64>],
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> std::result::Result<(f64, Vec<f64>), String> {
    let n = m.nrows();
    if n == 0 {
        return Err("empty matrix".into());
    }

    // Build shifted matrix A = M - sigma * I
    let mut a = m.to_owned();
    for i in 0..n {
        a[[i, i]] -= sigma;
    }

    // LU decomposition (partial pivoting) of A
    let (lu, piv) = lu_decompose(&a)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
    deflate(&mut v, prev_evecs);
    normalize_inplace(&mut v);

    let mut eigenvalue = sigma;

    for _ in 0..max_iter {
        // Solve A w = v  (LU solve)
        let mut w = lu_solve(&lu, &piv, &v)?;
        deflate(&mut w, prev_evecs);
        let norm = normalize_inplace(&mut w);
        if norm < 1e-300 {
            break;
        }
        // Rayleigh quotient with original matrix
        let mv = matvec(m, &w);
        let new_eigenvalue: f64 = mv.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        let diff: f64 = w.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = w;
        if (new_eigenvalue - eigenvalue).abs() < tol && diff < tol {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }

    Ok((eigenvalue, v))
}

/// LU decomposition with partial pivoting.
/// Returns `(LU, pivot_indices)` where `LU` stores L in the lower triangle
/// (unit diagonal) and U in the upper triangle.
fn lu_decompose(a: &Array2<f64>) -> std::result::Result<(Vec<Vec<f64>>, Vec<usize>), String> {
    let n = a.nrows();
    let mut lu: Vec<Vec<f64>> = (0..n).map(|i| a.row(i).to_vec()).collect();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[i][k].abs() > max_val {
                max_val = lu[i][k].abs();
                max_row = i;
            }
        }
        if max_val < 1e-300 {
            return Err(format!("singular matrix at column {k}"));
        }
        lu.swap(k, max_row);
        piv.swap(k, max_row);

        for i in (k + 1)..n {
            lu[i][k] /= lu[k][k];
            for j in (k + 1)..n {
                let factor = lu[i][k] * lu[k][j];
                lu[i][j] -= factor;
            }
        }
    }
    Ok((lu, piv))
}

/// Solve LU x = b using the factorization from `lu_decompose`.
fn lu_solve(
    lu: &[Vec<f64>],
    piv: &[usize],
    b: &[f64],
) -> std::result::Result<Vec<f64>, String> {
    let n = lu.len();
    // Apply row permutation to b
    let mut x: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        x[i] = b[piv[i]];
    }
    // Forward substitution (L unit lower)
    for i in 1..n {
        for j in 0..i {
            x[i] -= lu[i][j] * x[j];
        }
    }
    // Back substitution (U upper)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu[i][j] * x[j];
        }
        if lu[i][i].abs() < 1e-300 {
            return Err(format!("zero pivot at {i}"));
        }
        x[i] /= lu[i][i];
    }
    Ok(x)
}

/// Compute the `k` smallest eigenvalues and eigenvectors of a symmetric matrix
/// using inverse iteration with progressive deflation.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by eigenvalue (ascending).
fn smallest_k_eigen(
    m: &Array2<f64>,
    k: usize,
    seed: u64,
) -> std::result::Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let n = m.nrows();
    if k == 0 {
        return Ok((vec![], vec![]));
    }
    if k > n {
        return Err(format!("k={k} > n={n}"));
    }

    // Estimate largest eigenvalue via power iteration for shift selection
    let (lambda_max, _) = power_iteration(m, 200, 1e-8, seed);

    let mut eigenvalues: Vec<f64> = Vec::with_capacity(k);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Start slightly below 0 to capture the zero eigenvalue of Laplacians
    let mut shift = -1e-4;

    for idx in 0..k {
        // Pick a shift below the expected next eigenvalue
        // For Laplacian, eigenvalues are 0, lambda2, lambda3, ...
        let trial_shift = if idx == 0 {
            -1e-4 // just below 0
        } else {
            // Use a shift below the previous eigenvalue + small gap
            eigenvalues[idx - 1] - 1e-3
        };
        shift = trial_shift.min(shift);

        // Try inverse iteration; if singular (eigen = 0 causes singular shift), nudge
        let result = inverse_iteration(m, shift, &eigenvectors, 150, 1e-8, seed + idx as u64);
        let (eval, evec) = match result {
            Ok(r) => r,
            Err(_) => {
                // Try a small offset
                inverse_iteration(
                    m,
                    shift + 1e-6,
                    &eigenvectors,
                    150,
                    1e-8,
                    seed + idx as u64 + 1000,
                )?
            }
        };

        // Clamp negative near-zero eigenvalues of Laplacians to 0
        let clamped = if eval.abs() < 1e-6 { 0.0 } else { eval };
        eigenvalues.push(clamped);
        eigenvectors.push(evec);
        // For next shift, start well above the current eigenvalue
        shift = clamped + (lambda_max - clamped) * 0.01;
    }

    Ok((eigenvalues, eigenvectors))
}

// ─────────────────────────────────────────────────────────────────────────────
// Fiedler vector
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Fiedler value (λ₂) and Fiedler vector of a graph.
///
/// The Fiedler value is the **second-smallest** eigenvalue of the combinatorial
/// Laplacian L = D − A.  A positive Fiedler value indicates a connected graph.
/// The Fiedler vector provides a spectral bisection: sign of each component gives
/// the two-way partition.
///
/// # Returns
/// `(fiedler_value, fiedler_vector)` where the vector has unit L2-norm.
pub fn fiedler_vector(adj: &Array2<f64>) -> Result<(f64, Vec<f64>)> {
    let n = adj.nrows();
    if n < 2 {
        return Err(GraphError::InvalidGraph(
            "need at least 2 nodes for Fiedler vector".into(),
        ));
    }

    let lap = graph_laplacian(adj);
    let (evals, evecs) = smallest_k_eigen(&lap, 2, 12345).map_err(|e| GraphError::LinAlgError {
        operation: "fiedler_vector".into(),
        details: e,
    })?;

    if evals.len() < 2 {
        return Err(GraphError::LinAlgError {
            operation: "fiedler_vector".into(),
            details: "could not compute second eigenvalue".into(),
        });
    }

    Ok((evals[1], evecs[1].clone()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral clustering
// ─────────────────────────────────────────────────────────────────────────────

/// Result of spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralClusterResult {
    /// Cluster assignment (0-indexed) for each node.
    pub assignments: Vec<usize>,
    /// Eigenvalues of the normalized Laplacian used for embedding.
    pub eigenvalues: Vec<f64>,
    /// n × k embedding matrix (one row per node).
    pub embedding: Array2<f64>,
}

/// Spectral clustering: embed via k smallest eigenvectors of L̃, then k-means.
///
/// Uses the normalized Laplacian `L̃ = D^{-1/2} L D^{-1/2}` for the embedding.
/// After computing the k-dimensional spectral embedding, Lloyd's algorithm
/// (k-means) groups the nodes into `k` clusters.
///
/// # Arguments
/// * `adj`  – Symmetric weighted adjacency matrix (n × n).
/// * `k`    – Number of clusters.
/// * `seed` – RNG seed for k-means initialization.
pub fn spectral_clustering(adj: &Array2<f64>, k: usize, seed: u64) -> Result<SpectralClusterResult> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if k == 0 || k > n {
        return Err(GraphError::InvalidParameter {
            param: "k".into(),
            value: k.to_string(),
            expected: "1..=n".into(),
            context: "spectral_clustering".into(),
        });
    }

    let l_norm = normalized_laplacian(adj)?;
    let (evals, evecs) =
        smallest_k_eigen(&l_norm, k, seed).map_err(|e| GraphError::LinAlgError {
            operation: "spectral_clustering".into(),
            details: e,
        })?;

    // Build n×k embedding matrix (rows = nodes, cols = eigenvectors)
    let mut embedding = Array2::zeros((n, k));
    for col in 0..k {
        if col < evecs.len() {
            for row in 0..n {
                embedding[[row, col]] = evecs[col][row];
            }
        }
    }

    // Lloyd's k-means on the embedding
    let assignments = kmeans_lloyd(&embedding, k, 300, seed)?;

    Ok(SpectralClusterResult {
        assignments,
        eigenvalues: evals,
        embedding,
    })
}

/// Lloyd's k-means algorithm on rows of an n×d matrix.
/// Returns cluster assignments of length n.
fn kmeans_lloyd(data: &Array2<f64>, k: usize, max_iter: usize, seed: u64) -> Result<Vec<usize>> {
    let n = data.nrows();
    let d = data.ncols();
    if n == 0 || k == 0 {
        return Ok(vec![]);
    }

    // k-means++ style initialization: pick k distinct rows as initial centroids
    let mut rng = StdRng::seed_from_u64(seed);

    // Pick first centroid uniformly
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first_idx = rng.random_range(0..n);
    centroids.push(data.row(first_idx).to_vec());

    // Pick remaining centroids proportional to squared distance from nearest centroid
    for _ in 1..k {
        let dists: Vec<f64> = (0..n)
            .map(|i| {
                let row = data.row(i);
                centroids
                    .iter()
                    .map(|c| {
                        row.iter()
                            .zip(c.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        let mut r = rng.random::<f64>() * total;
        let mut chosen = n - 1;
        for (i, &d_val) in dists.iter().enumerate() {
            r -= d_val;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(data.row(chosen).to_vec());
    }

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assignment step
        let mut changed = false;
        for i in 0..n {
            let row = data.row(i);
            let mut best_dist = f64::INFINITY;
            let mut best_c = 0;
            for (ci, centroid) in centroids.iter().enumerate() {
                let dist: f64 = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_c = ci;
                }
            }
            if assignments[i] != best_c {
                changed = true;
                assignments[i] = best_c;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids
        let mut sums = vec![vec![0.0f64; d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..d {
                sums[c][j] += data[[i, j]];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c][j] = sums[c][j] / counts[c] as f64;
                }
            }
        }
    }

    Ok(assignments)
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Fourier Transform
// ─────────────────────────────────────────────────────────────────────────────

/// Graph Fourier Transform: project a nodal signal onto eigenvectors of L.
///
/// Returns the frequency-domain coefficients `x̂ = U^T x` where `U` is the
/// matrix of eigenvectors of the combinatorial Laplacian, sorted by eigenvalue.
/// The coefficient `x̂[k]` represents the amplitude of the k-th graph frequency.
///
/// # Arguments
/// * `adj`    – Symmetric weighted adjacency matrix (n × n).
/// * `signal` – Nodal signal of length n.
pub fn graph_fourier_transform(adj: &Array2<f64>, signal: &[f64]) -> Result<Vec<f64>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if signal.len() != n {
        return Err(GraphError::InvalidParameter {
            param: "signal".into(),
            value: signal.len().to_string(),
            expected: format!("{n}"),
            context: "graph_fourier_transform".into(),
        });
    }

    let lap = graph_laplacian(adj);
    let (_, evecs) = smallest_k_eigen(&lap, n, 42).map_err(|e| GraphError::LinAlgError {
        operation: "graph_fourier_transform".into(),
        details: e,
    })?;

    // x̂[k] = <u_k, x>
    let coeffs: Vec<f64> = evecs
        .iter()
        .map(|uk| uk.iter().zip(signal.iter()).map(|(a, b)| a * b).sum())
        .collect();

    Ok(coeffs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Effective resistance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Moore-Penrose pseudo-inverse of a symmetric PSD matrix.
///
/// Uses the eigendecomposition: L^+ = sum_{k: λ_k > tol} (1/λ_k) u_k u_k^T.
fn pinv_symmetric(m: &Array2<f64>, tol: f64) -> std::result::Result<Array2<f64>, String> {
    let n = m.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    let (evals, evecs) = smallest_k_eigen(m, n, 999)?;

    let mut pinv = Array2::zeros((n, n));
    for (k, &lambda) in evals.iter().enumerate() {
        if lambda.abs() <= tol {
            continue;
        }
        let uk = &evecs[k];
        for i in 0..n {
            for j in 0..n {
                pinv[[i, j]] += uk[i] * uk[j] / lambda;
            }
        }
    }
    Ok(pinv)
}

/// Effective resistance R_{ij} = (e_i − e_j)^T L^+ (e_i − e_j)
///                              = L^+_{ii} + L^+_{jj} − 2 L^+_{ij}.
///
/// Requires a connected graph; isolated nodes yield infinite resistance to
/// all other nodes.
///
/// # Arguments
/// * `adj` – Symmetric weighted adjacency matrix.
/// * `i`   – Source node index.
/// * `j`   – Target node index.
pub fn effective_resistance(adj: &Array2<f64>, i: usize, j: usize) -> Result<f64> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if i >= n || j >= n {
        return Err(GraphError::InvalidParameter {
            param: "i or j".into(),
            value: format!("({i},{j})"),
            expected: format!("< {n}"),
            context: "effective_resistance".into(),
        });
    }
    if i == j {
        return Ok(0.0);
    }

    let lap = graph_laplacian(adj);
    let lp = pinv_symmetric(&lap, 1e-9).map_err(|e| GraphError::LinAlgError {
        operation: "effective_resistance".into(),
        details: e,
    })?;

    let r = lp[[i, i]] + lp[[j, j]] - 2.0 * lp[[i, j]];
    Ok(r.max(0.0))
}

/// Compute the all-pairs effective resistance matrix R where R_{ij} is the
/// effective resistance between nodes i and j.
///
/// Diagonal entries are 0.  Uses a single pseudo-inverse computation.
pub fn resistance_matrix(adj: &Array2<f64>) -> Result<Array2<f64>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph("adjacency matrix must be square".into()));
    }

    let lap = graph_laplacian(adj);
    let lp = pinv_symmetric(&lap, 1e-9).map_err(|e| GraphError::LinAlgError {
        operation: "resistance_matrix".into(),
        details: e,
    })?;

    let mut r_mat = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let r = lp[[i, i]] + lp[[j, j]] - 2.0 * lp[[i, j]];
                r_mat[[i, j]] = r.max(0.0);
            }
        }
    }
    Ok(r_mat)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn path_adj(n: usize) -> Array2<f64> {
        let mut adj = Array2::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    fn complete_adj(n: usize) -> Array2<f64> {
        let mut adj = Array2::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        adj
    }

    /// Complete bipartite graph K_{m,n}.
    fn complete_bipartite(m: usize, n: usize) -> Array2<f64> {
        let total = m + n;
        let mut adj = Array2::zeros((total, total));
        for i in 0..m {
            for j in m..total {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        adj
    }

    // ── graph_laplacian ──────────────────────────────────────────────────────

    #[test]
    fn test_laplacian_path3() {
        // P3: 0-1-2
        let adj = path_adj(3);
        let l = graph_laplacian(&adj);
        // Row sums must be zero (Laplacian property)
        for i in 0..3 {
            let row_sum: f64 = l.row(i).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
        // Diagonal = degree
        assert!((l[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((l[[1, 1]] - 2.0).abs() < 1e-12);
        assert!((l[[2, 2]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_laplacian_complete4() {
        let adj = complete_adj(4);
        let l = graph_laplacian(&adj);
        for i in 0..4 {
            let row_sum: f64 = l.row(i).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
            assert!((l[[i, i]] - 3.0).abs() < 1e-12);
        }
    }

    // ── normalized_laplacian ─────────────────────────────────────────────────

    #[test]
    fn test_normalized_laplacian_diagonal_ones() {
        // For regular graphs (all nodes same degree), diagonal of L̃ = 1
        let adj = complete_adj(4); // all have degree 3
        let l_norm = normalized_laplacian(&adj).expect("norm lap");
        for i in 0..4 {
            assert!(
                (l_norm[[i, i]] - 1.0).abs() < 1e-10,
                "diagonal[{i}] = {}",
                l_norm[[i, i]]
            );
        }
    }

    #[test]
    fn test_normalized_laplacian_symmetric() {
        let adj = path_adj(5);
        let l_norm = normalized_laplacian(&adj).expect("norm lap");
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (l_norm[[i, j]] - l_norm[[j, i]]).abs() < 1e-12,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    // ── fiedler_vector ───────────────────────────────────────────────────────

    #[test]
    fn test_fiedler_value_positive_connected() {
        // A connected graph must have a positive Fiedler value
        let adj = path_adj(6);
        let (lambda2, fv) = fiedler_vector(&adj).expect("fiedler");
        assert!(lambda2 > 1e-6, "Fiedler value should be positive: {lambda2}");
        assert_eq!(fv.len(), 6);
        // Verify the vector is a unit vector
        let norm: f64 = fv.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Fiedler vector should be unit norm: {norm}");
    }

    #[test]
    fn test_fiedler_bisection() {
        // For a path graph P4, the Fiedler vector has opposite signs on the two halves
        let adj = path_adj(4);
        let (_lambda2, fv) = fiedler_vector(&adj).expect("fiedler");
        // The Fiedler vector of P4 should split the path into two halves
        // (the two groups should have opposite mean sign)
        let left_mean = (fv[0] + fv[1]) / 2.0;
        let right_mean = (fv[2] + fv[3]) / 2.0;
        assert!(
            left_mean * right_mean < 0.0,
            "Fiedler vector should bisect: left={left_mean}, right={right_mean}"
        );
    }

    #[test]
    fn test_fiedler_error_single_node() {
        let adj = Array2::<f64>::zeros((1, 1));
        assert!(fiedler_vector(&adj).is_err());
    }

    // ── spectral_clustering ──────────────────────────────────────────────────

    #[test]
    fn test_spectral_clustering_shapes() {
        let adj = path_adj(6);
        let result = spectral_clustering(&adj, 2, 42).expect("spectral clustering");
        assert_eq!(result.assignments.len(), 6);
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.embedding.nrows(), 6);
        assert_eq!(result.embedding.ncols(), 2);
    }

    #[test]
    fn test_spectral_clustering_two_components() {
        // Two disjoint paths: nodes 0-1-2 and nodes 3-4-5
        let mut adj = Array2::zeros((6, 6));
        adj[[0, 1]] = 1.0; adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0; adj[[2, 1]] = 1.0;
        adj[[3, 4]] = 1.0; adj[[4, 3]] = 1.0;
        adj[[4, 5]] = 1.0; adj[[5, 4]] = 1.0;

        let result = spectral_clustering(&adj, 2, 7).expect("spectral clustering");
        // The two components must be in different clusters
        let c0 = result.assignments[0];
        let c3 = result.assignments[3];
        assert_ne!(c0, c3, "disconnected components should be in different clusters");
        // All nodes in the same component should be in the same cluster
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[1], result.assignments[2]);
        assert_eq!(result.assignments[3], result.assignments[4]);
        assert_eq!(result.assignments[4], result.assignments[5]);
    }

    #[test]
    fn test_spectral_clustering_invalid_k() {
        let adj = path_adj(4);
        assert!(spectral_clustering(&adj, 0, 0).is_err());
        assert!(spectral_clustering(&adj, 5, 0).is_err());
    }

    // ── graph_fourier_transform ──────────────────────────────────────────────

    #[test]
    fn test_gft_length() {
        let adj = path_adj(4);
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let coeffs = graph_fourier_transform(&adj, &signal).expect("gft");
        assert_eq!(coeffs.len(), 4);
    }

    #[test]
    fn test_gft_constant_signal() {
        // A constant signal = DC component, only zeroth frequency has nonzero coefficient
        let adj = complete_adj(4);
        let signal = vec![1.0; 4];
        let coeffs = graph_fourier_transform(&adj, &signal).expect("gft");
        assert_eq!(coeffs.len(), 4);
        // The DC component (k=0) should be large; others near 0
        let max_coeff = coeffs.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_coeff > 0.5, "DC component should dominate: {coeffs:?}");
    }

    // ── effective_resistance ─────────────────────────────────────────────────

    #[test]
    fn test_effective_resistance_self() {
        let adj = path_adj(4);
        let r = effective_resistance(&adj, 1, 1).expect("eff res");
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_effective_resistance_path3() {
        // For a path P3: 0-1-2
        // R(0,1) = 1, R(1,2) = 1, R(0,2) = 2 (series)
        let adj = path_adj(3);
        let r01 = effective_resistance(&adj, 0, 1).expect("r01");
        let r12 = effective_resistance(&adj, 1, 2).expect("r12");
        let r02 = effective_resistance(&adj, 0, 2).expect("r02");
        assert!((r01 - 1.0).abs() < 1e-4, "R(0,1) = {r01}");
        assert!((r12 - 1.0).abs() < 1e-4, "R(1,2) = {r12}");
        assert!((r02 - 2.0).abs() < 1e-4, "R(0,2) = {r02}");
    }

    #[test]
    fn test_effective_resistance_complete_graph() {
        // For K_n with unit weights, R(i,j) = 2/n for all i≠j
        let n = 4;
        let adj = complete_adj(n);
        let r = effective_resistance(&adj, 0, 1).expect("eff res");
        let expected = 2.0 / n as f64;
        assert!(
            (r - expected).abs() < 1e-4,
            "K{n} effective resistance = {r}, expected {expected}"
        );
    }

    // ── resistance_matrix ────────────────────────────────────────────────────

    #[test]
    fn test_resistance_matrix_shape() {
        let adj = path_adj(4);
        let r_mat = resistance_matrix(&adj).expect("res mat");
        assert_eq!(r_mat.nrows(), 4);
        assert_eq!(r_mat.ncols(), 4);
    }

    #[test]
    fn test_resistance_matrix_symmetric() {
        let adj = path_adj(5);
        let r_mat = resistance_matrix(&adj).expect("res mat");
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (r_mat[[i, j]] - r_mat[[j, i]]).abs() < 1e-8,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_resistance_matrix_zero_diagonal() {
        let adj = complete_adj(4);
        let r_mat = resistance_matrix(&adj).expect("res mat");
        for i in 0..4 {
            assert!(r_mat[[i, i]].abs() < 1e-10, "diagonal[{i}] = {}", r_mat[[i, i]]);
        }
    }

    #[test]
    fn test_resistance_matrix_path3_values() {
        let adj = path_adj(3);
        let r_mat = resistance_matrix(&adj).expect("res mat");
        // R(0,1)=1, R(1,2)=1, R(0,2)=2
        assert!((r_mat[[0, 1]] - 1.0).abs() < 1e-4, "R(0,1) = {}", r_mat[[0, 1]]);
        assert!((r_mat[[1, 2]] - 1.0).abs() < 1e-4, "R(1,2) = {}", r_mat[[1, 2]]);
        assert!((r_mat[[0, 2]] - 2.0).abs() < 1e-4, "R(0,2) = {}", r_mat[[0, 2]]);
    }

    #[test]
    fn test_bipartite_laplacian_max_eigenvalue() {
        // For K_{2,2} (complete bipartite), largest eigenvalue of L_norm = 2.0
        let adj = complete_bipartite(2, 2);
        let l_norm = normalized_laplacian(&adj).expect("norm lap");
        // Largest eigenvalue via power iteration
        let (lambda_max, _) = power_iteration(&l_norm, 500, 1e-10, 77);
        assert!(
            (lambda_max - 2.0).abs() < 0.05,
            "K_{{2,2}} max eigenvalue of L_norm = {lambda_max}, expected 2.0"
        );
    }
}
