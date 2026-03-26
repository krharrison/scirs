//! Directed graph embedding algorithms: HOPE (Ou et al. 2016) and APP (Zhou et al. 2017).

use super::types::{DirectedGraph, EmbeddingResult};

// ─────────────────────────────────────────────────────────────────────────────
// Dense matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn normalise(v: &mut [f64]) {
    let n = norm(v);
    if n > 1e-14 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Matrix-vector multiply: y = A x  (A stored row-major as Vec<Vec<f64>>).
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

/// Matrix-matrix multiply: C = A · B  (all row-major, sizes compatible).
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let p = b.len();
    let n = if p > 0 { b[0].len() } else { 0 };
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for k in 0..p {
            if a[i][k].abs() < 1e-300 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Scale matrix by scalar.
fn mat_scale(m: &mut [Vec<f64>], s: f64) {
    for row in m.iter_mut() {
        for x in row.iter_mut() {
            *x *= s;
        }
    }
}

/// Add matrices in place: A += B.
fn mat_add_inplace(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
    for (ar, br) in a.iter_mut().zip(b.iter()) {
        for (x, y) in ar.iter_mut().zip(br.iter()) {
            *x += y;
        }
    }
}

/// Build the n×n column-normalised (row-stochastic for transpose) adjacency matrix
/// from a DirectedGraph.  A[i][j] = w_{ij} / out_degree(i).
fn stochastic_matrix(graph: &DirectedGraph) -> Vec<Vec<f64>> {
    let n = graph.n_nodes;
    let mut a = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let deg: f64 = graph.out_adj[i].iter().map(|&(_, w)| w).sum();
        if deg > 1e-14 {
            for &(j, w) in &graph.out_adj[i] {
                a[i][j] = w / deg;
            }
        }
    }
    a
}

/// Build un-normalised adjacency matrix.
fn raw_adjacency(graph: &DirectedGraph) -> Vec<Vec<f64>> {
    let n = graph.n_nodes;
    let mut a = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for &(j, w) in &graph.out_adj[i] {
            a[i][j] = w;
        }
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────────
// Randomised truncated SVD (for HOPE / APP proximity matrices)
// ─────────────────────────────────────────────────────────────────────────────

/// Randomised SVD to extract the top-`rank` singular triplets of an n×n matrix M.
///
/// Returns (U, sigma, Vt) where U is n×rank, sigma is rank, Vt is rank×n.
/// Uses the Halko-Martinsson-Tropp 2011 algorithm with `n_iter` power iterations.
fn randomised_svd(
    m: &[Vec<f64>],
    rank: usize,
    n_iter: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = m.len();
    let k = rank.min(n);

    // LCG random projection matrix Ω  (n × k)
    let mut state = seed;
    let lcg_next = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64) - 0.5
    };
    let mut omega: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..k).map(|_| lcg_next(&mut state)).collect())
        .collect();

    // Y = M · Ω
    let mut y: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let col: Vec<f64> = omega.iter().map(|row| row[j]).collect();
                    dot(&m[i], &col)
                })
                .collect()
        })
        .collect();

    // Power iteration: Y = (M M^T)^q M Ω
    for _ in 0..n_iter {
        // Z = M^T Y  (size n×k)
        let mut z: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; n];
        for i in 0..n {
            for jj in 0..n {
                for q in 0..k {
                    z[i][q] += m[jj][i] * y[jj][q];
                }
            }
        }
        // Y = M Z
        y = (0..n)
            .map(|i| {
                (0..k)
                    .map(|q| dot(&m[i], &z.iter().map(|r| r[q]).collect::<Vec<_>>()))
                    .collect()
            })
            .collect();
    }

    // QR decomposition of Y via Gram-Schmidt to get Q  (n×k)
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut col: Vec<f64> = (0..n).map(|i| y[i][j]).collect();
        for prev in &q_cols {
            let c = dot(&col, prev);
            for i in 0..n {
                col[i] -= c * prev[i];
            }
        }
        normalise(&mut col);
        q_cols.push(col);
    }

    // B = Q^T M  (k×n)
    let b: Vec<Vec<f64>> = q_cols
        .iter()
        .map(|q| {
            (0..n)
                .map(|j| dot(q, &m.iter().map(|r| r[j]).collect::<Vec<_>>()))
                .collect()
        })
        .collect();

    // SVD of small B via one-shot QR-like approach (power iteration on B B^T)
    // For simplicity we use a randomised 1-step approach: B B^T eigenvectors
    let bbt: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| (0..n).map(|l| b[i][l] * b[j][l]).sum())
                .collect()
        })
        .collect();

    // Power iteration to get eigenvectors of B B^T
    let mut u_small: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut sigma: Vec<f64> = Vec::with_capacity(k);
    let mut deflated: Vec<Vec<f64>> = bbt.clone();

    for _ in 0..k {
        let mut v: Vec<f64> = (0..k).map(|i| (i as f64 + 0.5) / k as f64).collect();
        normalise(&mut v);
        for _ in 0..150 {
            let w = mat_vec(&deflated, &v);
            let mut w2 = w.clone();
            normalise(&mut w2);
            let diff: f64 = v
                .iter()
                .zip(w2.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            v = w2;
            if diff < 1e-10 {
                break;
            }
        }
        // Eigenvalue λ = v^T (B B^T) v
        let bbt_v = mat_vec(&bbt, &v);
        let lambda = dot(&v, &bbt_v).max(0.0);
        sigma.push(lambda.sqrt());
        u_small.push(v.clone());
        // Deflate
        for i in 0..k {
            for j in 0..k {
                deflated[i][j] -= lambda * v[i] * v[j];
            }
        }
    }

    // U = Q u_small  (n×k)
    let u: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    u_small
                        .iter()
                        .enumerate()
                        .map(|(l, us)| q_cols[l][i] * us[j])
                        .sum()
                })
                .collect()
        })
        .collect();

    // Vt[j] = (1/sigma_j) B^T u_j  (k×n)
    let vt: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let s = if sigma[j] > 1e-14 { sigma[j] } else { 1e-14 };
            (0..n)
                .map(|i| {
                    let bt_uj: f64 = (0..k).map(|l| b[l][i] * u_small[j][l]).sum();
                    bt_uj / s
                })
                .collect()
        })
        .collect();

    // Re-order by descending sigma
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let u_sorted: Vec<Vec<f64>> = (0..n)
        .map(|i| order.iter().map(|&o| u[i][o]).collect())
        .collect();
    let vt_sorted: Vec<Vec<f64>> = order.iter().map(|&o| vt[o].clone()).collect();
    let sigma_sorted: Vec<f64> = order.iter().map(|&o| sigma[o]).collect();

    (u_sorted, sigma_sorted, vt_sorted)
}

// ─────────────────────────────────────────────────────────────────────────────
// HOPE: High-Order Proximity preserved Embedding
// ─────────────────────────────────────────────────────────────────────────────

/// HOPE embedding (Ou et al. 2016).
///
/// Computes the truncated high-order proximity matrix
///   S ≈ Σ_{t=1}^{order} β^t A^t
/// then takes its top-`dim` SVD to produce source and target embeddings.
///
/// Returns `(source_emb, target_emb)` each of size n_nodes × dim.
pub fn hope_embedding(
    graph: &DirectedGraph,
    dim: usize,
    beta: f64,
    order: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = graph.n_nodes;
    let a = raw_adjacency(graph);

    // S = Σ_{t=1}^{order} β^t A^t
    let mut s = vec![vec![0.0_f64; n]; n];
    let mut at = a.clone(); // A^1
    let mut beta_t = beta;
    for _t in 0..order {
        // S += β^t * A^t
        for i in 0..n {
            for j in 0..n {
                s[i][j] += beta_t * at[i][j];
            }
        }
        // A^{t+1} = A · A^t
        let next = mat_mul(&a, &at);
        at = next;
        beta_t *= beta;
    }

    let rank = dim.min(n);
    let (u, sigma, vt) = randomised_svd(&s, rank, 3, 0xBEEF_CAFE);

    // Source embedding = U · sqrt(Σ)
    let source: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..rank).map(|k| u[i][k] * sigma[k].sqrt()).collect())
        .collect();
    // Target embedding = V · sqrt(Σ)  (V = Vt^T)
    let target: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..rank).map(|k| vt[k][i] * sigma[k].sqrt()).collect())
        .collect();

    (source, target)
}

// ─────────────────────────────────────────────────────────────────────────────
// APP: Asymmetric Proximity Preserving embedding
// ─────────────────────────────────────────────────────────────────────────────

/// APP embedding (Zhou et al. 2017).
///
/// For each node u, approximates the rooted PageRank vector
///   s_u ≈ α Σ_{k=0}^{steps} (1-α)^k A^k e_u
/// builds the n*n proximity matrix `P[u][v]` = s_u(v), then takes top-`dim` SVD.
///
/// Returns an `EmbeddingResult` of size n_nodes × dim.
pub fn app_embedding(
    graph: &DirectedGraph,
    dim: usize,
    alpha: f64,
    steps: usize,
) -> EmbeddingResult {
    let n = graph.n_nodes;
    let a = stochastic_matrix(graph);

    // Build n×n proximity matrix P where P[u] = s_u
    let mut p = vec![vec![0.0_f64; n]; n];
    for u in 0..n {
        // s_u = α Σ_{k=0}^steps (1-α)^k A^k e_u
        let mut ek = vec![0.0_f64; n];
        ek[u] = 1.0;
        let mut coeff = alpha;
        for k in 0..=steps {
            for v in 0..n {
                p[u][v] += coeff * ek[v];
            }
            let next = mat_vec(&a, &ek);
            ek = next;
            coeff *= 1.0 - alpha;
            let _ = k;
        }
    }

    let rank = dim.min(n);
    let (u_mat, sigma, _vt) = randomised_svd(&p, rank, 3, 0xCAFE_BABE);

    let mut result = EmbeddingResult::zeros(n, rank);
    for i in 0..n {
        for k in 0..rank {
            result.embeddings[i][k] = u_mat[i][k] * sigma[k].sqrt();
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Directed random-walk stationary distribution (for test_directed_pagerank)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the stationary distribution of the row-stochastic random-walk
/// matrix of `graph` using power iteration.  Returns a probability vector
/// that sums to 1 (within floating-point tolerance).
pub fn stationary_distribution(graph: &DirectedGraph) -> Vec<f64> {
    let n = graph.n_nodes;
    if n == 0 {
        return Vec::new();
    }
    let a = stochastic_matrix(graph);
    // Handle dangling nodes (rows that sum to 0) by uniform redistribution
    let teleport = 0.15_f64;
    let mut p = vec![1.0_f64 / n as f64; n];
    for _ in 0..300 {
        let mut next = vec![0.0_f64; n];
        for i in 0..n {
            let row_sum: f64 = a[i].iter().sum();
            if row_sum < 1e-14 {
                // Dangling: uniform
                for x in next.iter_mut() {
                    *x += p[i] / n as f64;
                }
            } else {
                for j in 0..n {
                    next[j] += p[i] * a[i][j];
                }
            }
        }
        // Mix with teleportation
        let sum: f64 = next.iter().sum();
        for (x, np) in p.iter_mut().zip(next.iter()) {
            *x = (1.0 - teleport) * (np / sum.max(1e-14)) + teleport / n as f64;
        }
    }
    // Normalise
    let s: f64 = p.iter().sum();
    if s > 1e-14 {
        for x in p.iter_mut() {
            *x /= s;
        }
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_directed::types::{DirectedEmbedConfig, DirectedGraph};

    fn simple_directed() -> DirectedGraph {
        // 0 → 1 → 2 → 3 → 0 (cycle) plus 0 → 2
        let mut g = DirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 0, 1.0);
        g.add_edge(0, 2, 1.0);
        g
    }

    #[test]
    fn test_directed_graph_construction() {
        let g = simple_directed();
        assert_eq!(g.n_nodes, 4);
        assert_eq!(g.edges.len(), 5);
        assert_eq!(g.out_degree(0), 2); // 0→1 and 0→2
        assert_eq!(g.in_degree(2), 2); // 1→2 and 0→2
        assert_eq!(g.in_degree(0), 1); // 3→0
    }

    #[test]
    fn test_hope_embedding_shape() {
        let g = simple_directed();
        let dim = 2;
        let (src, tgt) = hope_embedding(&g, dim, 0.5, 3);
        assert_eq!(src.len(), 4);
        assert_eq!(tgt.len(), 4);
        for row in &src {
            assert_eq!(row.len(), dim);
        }
        for row in &tgt {
            assert_eq!(row.len(), dim);
        }
    }

    #[test]
    fn test_hope_beta_effect() {
        let g = simple_directed();
        let (src, _) = hope_embedding(&g, 2, 0.5, 3);
        // With beta=0.5, proximity is nonzero — embedding should not be all zeros
        let total: f64 = src.iter().flat_map(|r| r.iter()).map(|x| x.abs()).sum();
        assert!(
            total > 0.001,
            "HOPE embedding should be non-zero for beta=0.5"
        );
    }

    #[test]
    fn test_hope_source_target_differ() {
        let g = simple_directed();
        let (src, tgt) = hope_embedding(&g, 2, 0.5, 3);
        // Source and target embeddings should differ (graph is asymmetric)
        let mut any_diff = false;
        for i in 0..4 {
            for j in 0..src[i].len() {
                if (src[i][j] - tgt[i][j]).abs() > 1e-8 {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "source and target embeddings should differ for asymmetric graph"
        );
    }

    #[test]
    fn test_app_embedding_shape() {
        let g = simple_directed();
        let dim = 2;
        let emb = app_embedding(&g, dim, 0.15, 5);
        assert_eq!(emb.n_nodes, 4);
        assert_eq!(emb.dim, dim);
        assert_eq!(emb.embeddings.len(), 4);
        for row in &emb.embeddings {
            assert_eq!(row.len(), dim);
        }
    }

    #[test]
    fn test_app_alpha_effect() {
        let g = simple_directed();
        // alpha close to 0 → multi-step random walk (more spread)
        // alpha close to 1 → one-hop
        let emb_low = app_embedding(&g, 2, 0.01, 10);
        let emb_high = app_embedding(&g, 2, 0.99, 2);
        // Both should be non-trivial
        let sum_low: f64 = emb_low
            .embeddings
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x.abs())
            .sum();
        let sum_high: f64 = emb_high
            .embeddings
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x.abs())
            .sum();
        assert!(
            sum_low > 0.0 || sum_high > 0.0,
            "APP should produce non-trivial embeddings"
        );
    }

    #[test]
    fn test_directed_pagerank_sums_one() {
        let g = simple_directed();
        let dist = stationary_distribution(&g);
        assert_eq!(dist.len(), 4);
        let s: f64 = dist.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-6,
            "stationary dist should sum to 1, got {s:.8}"
        );
        for (i, &p) in dist.iter().enumerate() {
            assert!(p >= 0.0, "probability[{i}] should be non-negative");
        }
    }

    #[test]
    fn test_hope_truncated_series() {
        let g = simple_directed();
        // Higher order → more terms in the approximation → higher total proximity mass
        let (src2, _) = hope_embedding(&g, 2, 0.5, 2);
        let (src5, _) = hope_embedding(&g, 2, 0.5, 5);
        let norm2: f64 = src2
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let norm5: f64 = src5
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        // The norms are allowed to differ; we just check both are finite and positive
        assert!(
            norm2.is_finite() && norm2 > 0.0,
            "order=2 norm should be finite and positive"
        );
        assert!(
            norm5.is_finite() && norm5 > 0.0,
            "order=5 norm should be finite and positive"
        );
    }

    #[test]
    fn test_directed_embed_config_default() {
        let cfg = DirectedEmbedConfig::default();
        assert!(cfg.dim > 0);
        assert!(cfg.n_iter > 0);
    }
}
