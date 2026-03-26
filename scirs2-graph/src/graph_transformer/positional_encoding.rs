//! Positional encodings for graph transformers.
//!
//! Implements:
//! - **LapPE** – Laplacian eigenvector positional encoding
//! - **RWPE**  – Random-walk landing probability positional encoding
//! - **APSP**  – All-pairs shortest paths (BFS) used by Graphormer

// ============================================================================
// Laplacian PE
// ============================================================================

/// Compute the graph Laplacian L = D - A and return its diagonal and the
/// symmetric normalised version needed for eigenvector computation.
///
/// Returns `(degree, laplacian_rows)` where `laplacian_rows[i][j]` = L[i][j].
fn build_laplacian(adj: &[Vec<usize>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = adj.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    let mut deg = vec![0.0_f64; n];
    for (i, nbrs) in adj.iter().enumerate() {
        deg[i] = nbrs.len() as f64;
        l[i][i] = deg[i];
        for &j in nbrs {
            l[i][j] -= 1.0;
        }
    }
    (deg, l)
}

/// Estimate the spectral radius of L by Gershgorin circles (max row sum of abs).
fn spectral_radius_estimate(l: &[Vec<f64>]) -> f64 {
    l.iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
        .max(1.0)
}

/// One matrix–vector multiplication: y = M · x
fn matvec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Normalise a vector to unit length; returns `false` if the norm is too small.
fn normalise_inplace(v: &mut [f64]) -> bool {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

/// Deflect `v` by subtracting its projections onto all vectors in `basis`.
fn deflate(v: &mut [f64], basis: &[Vec<f64>]) {
    for b in basis {
        let dot: f64 = v.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
        for (vi, bi) in v.iter_mut().zip(b.iter()) {
            *vi -= dot * bi;
        }
    }
}

/// Sign-normalise: make the first non-zero element of `v` positive.
fn sign_normalise(v: &mut [f64]) {
    for &val in v.iter() {
        if val.abs() > 1e-12 {
            if val < 0.0 {
                for x in v.iter_mut() {
                    *x = -*x;
                }
            }
            break;
        }
    }
}

/// Compute the **k** smallest non-trivial eigenvectors of the graph Laplacian L
/// using power iteration on the shifted matrix `M = I - L / sigma` where `sigma`
/// is a spectral-radius estimate.  The trivial all-ones eigenvector is skipped.
///
/// Returns an `n × k` matrix (outer vec = nodes, inner vec = PE dimensions).
/// If `k > n - 1` the result is zero-padded.
pub fn laplacian_pe(adj: &[Vec<usize>], k: usize) -> Vec<Vec<f64>> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![vec![0.0_f64; k]; n];
    if k == 0 || n <= 1 {
        return out;
    }

    let (_deg, l) = build_laplacian(adj);
    let sigma = spectral_radius_estimate(&l);

    // Build shifted matrix M = I - L / sigma  (eigenvectors of M correspond to
    // eigenvectors of L; smallest eigenvalues of L → largest eigenvalues of M)
    let mut m = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = if i == j {
                1.0 - l[i][j] / sigma
            } else {
                -l[i][j] / sigma
            };
        }
    }

    // The constant vector 1/√n is the trivial eigenvector with eigenvalue 1
    // (L·1 = 0, so M·1 = 1). We add it to the basis so power iteration skips it.
    let trivial: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
    let mut basis: Vec<Vec<f64>> = vec![trivial];

    // Simple LCG seeded deterministically
    let mut lcg: u64 = 0x5851_f42d_4c95_7f2d;

    let max_vectors = k.min(n - 1);
    for col in 0..max_vectors {
        // Initialise with pseudo-random vector
        let mut v: Vec<f64> = (0..n)
            .map(|_| {
                lcg = lcg
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = (lcg >> 33) as i32;
                (bits as f64) / (i32::MAX as f64)
            })
            .collect();
        deflate(&mut v, &basis);
        if !normalise_inplace(&mut v) {
            // Fallback: use canonical basis vector
            v = vec![0.0; n];
            v[col + 1] = 1.0;
            deflate(&mut v, &basis);
            if !normalise_inplace(&mut v) {
                break;
            }
        }

        // Power iteration (target: next largest eigenvalue of M → smallest of L)
        for _iter in 0..200 {
            let mv = matvec(&m, &v);
            let mut v_new = mv;
            deflate(&mut v_new, &basis);
            if !normalise_inplace(&mut v_new) {
                break;
            }
            // Convergence check
            let diff: f64 = v_new.iter().zip(v.iter()).map(|(a, b)| (a - b).abs()).sum();
            let diff_neg: f64 = v_new.iter().zip(v.iter()).map(|(a, b)| (a + b).abs()).sum();
            v = v_new;
            if diff < 1e-10 || diff_neg < 1e-10 {
                break;
            }
        }
        sign_normalise(&mut v);
        for i in 0..n {
            out[i][col] = v[i];
        }
        basis.push(v);
    }
    out
}

// ============================================================================
// Random-Walk PE
// ============================================================================

/// Compute **RWPE**: random-walk landing probabilities for `k` steps.
///
/// The random-walk matrix is `P = D⁻¹ A`.  For each step `t = 1..=k` we record
/// `diag(P^t)` — the probability of landing back at the starting node after
/// exactly `t` steps.
///
/// Returns an `n × k` matrix.
pub fn rwpe(adj: &[Vec<usize>], k: usize) -> Vec<Vec<f64>> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![vec![0.0_f64; k]; n];
    if k == 0 {
        return out;
    }

    // Compute degree and P = D⁻¹ A stored as dense matrix
    let deg: Vec<f64> = adj.iter().map(|nbrs| nbrs.len() as f64).collect();
    // p_mat[i][j] = P[i][j]
    let mut p_mat = vec![vec![0.0_f64; n]; n];
    for (i, nbrs) in adj.iter().enumerate() {
        if deg[i] > 0.0 {
            for &j in nbrs {
                p_mat[i][j] = 1.0 / deg[i];
            }
        }
    }

    // Iteratively compute P^t; start with P^0 = I
    let mut pt = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        pt[i][i] = 1.0;
    }

    for step in 0..k {
        // pt_new = pt · p_mat  (next power)
        let mut pt_new = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    pt_new[i][j] += pt[i][l] * p_mat[l][j];
                }
            }
        }
        pt = pt_new;
        // Record diagonal
        for i in 0..n {
            out[i][step] = pt[i][i];
        }
    }
    out
}

// ============================================================================
// All-pairs shortest paths (BFS)
// ============================================================================

/// Compute all-pairs shortest-path distances using BFS from each node.
///
/// Disconnected pairs receive the sentinel value `usize::MAX`.
pub fn all_pairs_shortest_path(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for src in 0..n {
        dist[src][src] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if dist[src][v] == usize::MAX {
                    dist[src][v] = dist[src][u] + 1;
                    queue.push_back(v);
                }
            }
        }
    }
    dist
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_adj() -> Vec<Vec<usize>> {
        vec![vec![1, 2], vec![0, 2], vec![0, 1]]
    }

    fn path4_adj() -> Vec<Vec<usize>> {
        // 0-1-2-3
        vec![vec![1], vec![0, 2], vec![1, 3], vec![2]]
    }

    // ---- LapPE ---------------------------------------------------------------

    #[test]
    fn test_laplacian_pe_shape() {
        let adj = triangle_adj();
        let pe = laplacian_pe(&adj, 2);
        assert_eq!(pe.len(), 3);
        for row in &pe {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn test_laplacian_pe_orthogonal() {
        let adj = path4_adj();
        let pe = laplacian_pe(&adj, 2);
        // Columns should be approximately orthogonal
        let n = pe.len();
        let dot: f64 = (0..n).map(|i| pe[i][0] * pe[i][1]).sum();
        assert!(dot.abs() < 0.15, "columns not orthogonal: dot={dot}");
    }

    #[test]
    fn test_pe_sign_normalization() {
        let adj = path4_adj();
        let pe = laplacian_pe(&adj, 2);
        // For each column, the first non-zero element should be positive
        for col in 0..2 {
            let first_nonzero = (0..pe.len()).find(|&i| pe[i][col].abs() > 1e-9);
            if let Some(idx) = first_nonzero {
                assert!(
                    pe[idx][col] > 0.0,
                    "column {col} first nonzero is negative: {}",
                    pe[idx][col]
                );
            }
        }
    }

    // ---- RWPE ----------------------------------------------------------------

    #[test]
    fn test_rwpe_shape() {
        let adj = triangle_adj();
        let pe = rwpe(&adj, 3);
        assert_eq!(pe.len(), 3);
        for row in &pe {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_rwpe_probabilities_bounded() {
        let adj = path4_adj();
        let pe = rwpe(&adj, 4);
        for row in &pe {
            for &v in row {
                assert!((-1e-10..=1.0 + 1e-10).contains(&v), "out of [0,1]: {v}");
            }
        }
    }

    // ---- APSP ----------------------------------------------------------------

    #[test]
    fn test_apsp_complete_graph() {
        // K4
        let adj = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        let d = all_pairs_shortest_path(&adj);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(d[i][j], 0);
                } else {
                    assert_eq!(d[i][j], 1);
                }
            }
        }
    }

    #[test]
    fn test_apsp_path_graph() {
        // 0-1-2-3
        let adj = path4_adj();
        let d = all_pairs_shortest_path(&adj);
        assert_eq!(d[0][3], 3);
        assert_eq!(d[1][3], 2);
        assert_eq!(d[0][2], 2);
    }

    #[test]
    fn test_apsp_disconnected() {
        // Two separate edges: 0-1  and  2-3
        let adj = vec![vec![1], vec![0], vec![3], vec![2]];
        let d = all_pairs_shortest_path(&adj);
        assert_eq!(d[0][2], usize::MAX);
        assert_eq!(d[0][3], usize::MAX);
        assert_eq!(d[0][1], 1);
        assert_eq!(d[2][3], 1);
    }
}
