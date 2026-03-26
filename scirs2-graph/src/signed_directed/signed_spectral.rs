//! Signed spectral embedding: Signed Laplacian, SPONGE (Cucuringu 2019),
//! signed ratio-cut clustering, and status-theory score (Leskovec 2010).

use super::types::{EmbeddingResult, SignedGraph};

// ─────────────────────────────────────────────────────────────────────────────
// Internal maths helpers
// ─────────────────────────────────────────────────────────────────────────────

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

fn deflate_all(v: &mut [f64], basis: &[Vec<f64>]) {
    for b in basis {
        let c = dot(v, b);
        for (x, y) in v.iter_mut().zip(b.iter()) {
            *x -= c * y;
        }
    }
}

fn mat_vec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, x)).collect()
}

fn add_tau_identity(m: &mut [Vec<f64>], tau: f64) {
    for i in 0..m.len() {
        m[i][i] += tau;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conjugate Gradient solver
// ─────────────────────────────────────────────────────────────────────────────

fn cg_solve(m: &[Vec<f64>], b: &[f64], max_iter: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    let mut r: Vec<f64> = b.to_vec();
    let mut p = r.clone();
    let mut rsold = dot(&r, &r);
    for _ in 0..max_iter {
        if rsold < 1e-28 {
            break;
        }
        let ap = mat_vec(m, &p);
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-300 {
            break;
        }
        let alpha = rsold / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rsnew = dot(&r, &r);
        let beta = rsnew / rsold;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// Signed Laplacian matrices
// ─────────────────────────────────────────────────────────────────────────────

/// Build the **signed Laplacian** L = D_|A| - A  (all edges treated as ±1).
pub fn signed_laplacian(graph: &SignedGraph) -> Vec<Vec<f64>> {
    let n = graph.n_nodes;
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for &j in &graph.pos_adj[i] {
            l[i][j] -= 1.0;
            l[i][i] += 1.0;
        }
        for &j in &graph.neg_adj[i] {
            l[i][j] += 1.0; // -a_ij = -(-1) = +1
            l[i][i] += 1.0;
        }
    }
    l
}

/// Build the **positive Laplacian** L+ = D+ - A+ (positive edges only).
pub fn positive_laplacian(graph: &SignedGraph) -> Vec<Vec<f64>> {
    let n = graph.n_nodes;
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for &j in &graph.pos_adj[i] {
            l[i][j] -= 1.0;
            l[i][i] += 1.0;
        }
    }
    l
}

/// Build the **negative Laplacian** L- = D- - A- (negative edges only).
pub fn negative_laplacian(graph: &SignedGraph) -> Vec<Vec<f64>> {
    let n = graph.n_nodes;
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for &j in &graph.neg_adj[i] {
            l[i][j] -= 1.0;
            l[i][i] += 1.0;
        }
    }
    l
}

// ─────────────────────────────────────────────────────────────────────────────
// Simultaneous iteration (subspace iteration / block power method)
// to find the k SMALLEST eigenvectors of a symmetric matrix.
//
// Strategy: find k+over eigenvectors of A^{-1} (largest of inverse = smallest
// of A), then sort by ascending eigenvalue and return the top `k` after
// filtering trivially flat ones.
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `k` eigenvectors of `mat` with the **smallest** eigenvalues using
/// subspace (simultaneous) inverse iteration with Gram-Schmidt reorthogonalisation.
///
/// `flat_thresh` controls how "flat" (near-constant) a vector must be before
/// it is discarded and an extra dimension is substituted.
fn subspace_smallest_eigvecs(
    mat: &[Vec<f64>],
    k: usize,
    max_outer: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = mat.len();
    let over = (k + 2).min(n); // extra vectors for stability

    // Regularise: A' = mat + ε I  (ensures CG convergence; eigenvectors unchanged)
    let eps = 1e-6_f64;
    let mut a_reg = mat.to_vec();
    for i in 0..n {
        a_reg[i][i] += eps;
    }

    // LCG random initial subspace Q (n × over), column-orthonormal
    let mut state = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64) - 0.5
    };

    let mut q: Vec<Vec<f64>> = (0..over)
        .map(|_| (0..n).map(|_| lcg(&mut state)).collect())
        .collect();

    // Gram-Schmidt orthonormalise q
    gram_schmidt_inplace(&mut q);

    // Subspace iteration: Q_{k+1} = orth(A^{-1} Q_k)
    for _iter in 0..max_outer {
        let q_old = q.clone();
        // Solve A' z_j = q_j for each column
        let mut z: Vec<Vec<f64>> = q
            .iter()
            .map(|qj| cg_solve(&a_reg, qj, 4 * n + 80))
            .collect();
        gram_schmidt_inplace(&mut z);
        // Convergence: max column change
        let delta: f64 = z
            .iter()
            .zip(q_old.iter())
            .map(|(zj, qj)| {
                zj.iter()
                    .zip(qj.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max);
        q = z;
        if delta < 1e-9 * n as f64 {
            break;
        }
    }

    // Compute Rayleigh quotients λ_j = q_j^T mat q_j to get eigenvalues
    let mut pairs: Vec<(f64, Vec<f64>)> = q
        .iter()
        .map(|qj| {
            let mq = mat_vec(mat, qj);
            let lambda = dot(qj, &mq);
            (lambda, qj.clone())
        })
        .collect();

    // Sort ascending by eigenvalue
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let lambdas: Vec<f64> = pairs.iter().map(|(l, _)| *l).collect();
    let vecs: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    (vecs, lambdas)
}

fn gram_schmidt_inplace(vecs: &mut [Vec<f64>]) {
    let k = vecs.len();
    for j in 0..k {
        // Orthogonalise vecs[j] against vecs[0..j]
        for i in 0..j {
            let vi = vecs[i].clone();
            let vj = &mut vecs[j];
            let c = dot(vj, &vi);
            for (x, y) in vj.iter_mut().zip(vi.iter()) {
                *x -= c * y;
            }
        }
        normalise(&mut vecs[j]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPONGE embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Detect a "flat" (nearly constant) vector: variance < thresh.
fn is_flat(v: &[f64], thresh: f64) -> bool {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    var < thresh
}

/// SPONGE (Signed Positive Over Negative Generalised Eigenproblem) embedding.
///
/// Approximates the SPONGE generalised eigenproblem
///   (L+ + τI) v = λ (L- + τI) v
/// by working with the signed Laplacian L_|A| = L+ + L- and extracting the
/// non-trivial smallest eigenvectors (Fiedler-like vectors that encode signed
/// community structure).
///
/// Returns an `EmbeddingResult` (n_nodes × dim).
pub fn sponge_embedding(graph: &SignedGraph, dim: usize, tau: f64) -> EmbeddingResult {
    let n = graph.n_nodes;

    // M = L+ + τ·(L-)  is the SPONGE numerator/denominator proxy.
    // For the spectral embedding we use the signed Laplacian L_|A| + τ I
    // whose nontrivial small eigenvectors separate balanced communities.
    let mut lsigned = signed_laplacian(graph);
    add_tau_identity(&mut lsigned, tau);

    // We request more vectors than needed so we can discard flat ones
    let n_req = (dim * 2 + 2).min(n);
    let (all_vecs, _lambdas) = subspace_smallest_eigvecs(&lsigned, n_req, 400, 0xFEED_FACE);

    // Filter flat (constant) vectors, then take the first `dim` informative ones
    const FLAT_THRESH: f64 = 1e-6;
    let selected: Vec<Vec<f64>> = all_vecs
        .into_iter()
        .filter(|v| !is_flat(v, FLAT_THRESH))
        .take(dim)
        .collect();

    let actual_dim = selected.len().max(1);
    let mut result = EmbeddingResult::zeros(n, actual_dim);
    for (k, evec) in selected.iter().enumerate() {
        for i in 0..n {
            result.embeddings[i][k] = evec[i];
        }
    }
    result.dim = actual_dim;
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// LCG k-means with k-means++ initialisation
// ─────────────────────────────────────────────────────────────────────────────

fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let dim = data[0].len();

    let mut state = seed;
    let lcg = |s: &mut u64| -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    };

    // k-means++ initialisation
    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(k);
    centres.push(data[(lcg(&mut state) as usize) % n].clone());
    for _ in 1..k {
        let dists: Vec<f64> = data
            .iter()
            .map(|row| {
                centres
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
        let chosen = if total < 1e-14 {
            (lcg(&mut state) as usize) % n
        } else {
            let mut r = (lcg(&mut state) as f64 / u32::MAX as f64) * total;
            let mut idx = n - 1;
            for (i, &d) in dists.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        };
        centres.push(data[chosen].clone());
    }

    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, row) in data.iter().enumerate() {
            let best = centres
                .iter()
                .enumerate()
                .map(|(c, cen)| {
                    let d: f64 = row
                        .iter()
                        .zip(cen.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (c, d)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| c)
                .unwrap_or(0);
            if best != labels[i] {
                changed = true;
            }
            labels[i] = best;
        }
        if !changed {
            break;
        }
        let mut new_c = vec![vec![0.0_f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, row) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for d in 0..dim {
                new_c[c][d] += row[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    new_c[c][d] /= counts[c] as f64;
                }
            } else {
                centres[c] = data[(lcg(&mut state) as usize) % n].clone();
            }
        }
        if counts.iter().all(|&c| c > 0) {
            centres = new_c;
        }
    }
    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Signed ratio-cut clustering
// ─────────────────────────────────────────────────────────────────────────────

/// Signed ratio-cut clustering via signed Laplacian eigenvectors + k-means.
///
/// Runs k-means multiple times with different seeds and returns the partition
/// with the lowest within-cluster sum-of-squares (WCSS).
pub fn signed_ratio_cut(graph: &SignedGraph, k: usize) -> Vec<usize> {
    let emb = sponge_embedding(graph, k, 0.1);
    let data = &emb.embeddings;
    let dim = emb.dim;

    // Run 10 restarts and keep the best
    let seeds: [u64; 10] = [
        0xABCD_1234,
        0xDEAD_BEEF,
        0xCAFE_BABE,
        0x1234_5678,
        0x9ABC_DEF0,
        0xFEED_FACE,
        0xBADD_CAFE,
        0xD00D_F00D,
        0xABBA_ABBA,
        0x1357_9BDF,
    ];

    let wcss = |labels: &[usize]| -> f64 {
        let mut centres = vec![vec![0.0_f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in labels.iter().enumerate() {
            counts[c] += 1;
            for d in 0..dim {
                centres[c][d] += data[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centres[c][d] /= counts[c] as f64;
                }
            }
        }
        labels
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                data[i]
                    .iter()
                    .zip(centres[c].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .sum()
    };

    let mut best_labels = kmeans(data, k, 300, seeds[0]);
    let mut best_wcss = wcss(&best_labels);
    for &seed in &seeds[1..] {
        let labels = kmeans(data, k, 300, seed);
        let w = wcss(&labels);
        if w < best_wcss {
            best_wcss = w;
            best_labels = labels;
        }
    }
    best_labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Status-theory score
// ─────────────────────────────────────────────────────────────────────────────

/// Compute status scores using Leskovec et al. 2010 status theory.
///
/// Edge u→(+1)→v means v has higher status; u→(-1)→v means v has lower status.
pub fn status_score(graph: &SignedGraph) -> Vec<f64> {
    let n = graph.n_nodes;
    if n == 0 {
        return Vec::new();
    }
    let mut scores = vec![0.0_f64; n];

    for _iter in 0..500 {
        let prev = scores.clone();
        for edge in &graph.edges {
            let u = edge.src;
            let v = edge.dst;
            let sign = edge.sign as f64;
            let residual = sign - (scores[v] - scores[u]);
            scores[v] += 0.25 * residual;
            scores[u] -= 0.25 * residual;
        }
        let mean = scores.iter().sum::<f64>() / n as f64;
        for s in scores.iter_mut() {
            *s -= mean;
        }
        let delta: f64 = scores
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        if delta < 1e-8 * n as f64 {
            break;
        }
    }
    scores
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_balanced_components() -> SignedGraph {
        let mut g = SignedGraph::new(6);
        // Group A (0,1,2): all positive
        g.add_edge(0, 1, 1);
        g.add_edge(0, 2, 1);
        g.add_edge(1, 2, 1);
        // Group B (3,4,5): all positive
        g.add_edge(3, 4, 1);
        g.add_edge(3, 5, 1);
        g.add_edge(4, 5, 1);
        // Cross-group: all negative (strongly imbalanced between groups)
        g.add_edge(0, 3, -1);
        g.add_edge(1, 4, -1);
        g.add_edge(2, 5, -1);
        g
    }

    #[test]
    fn test_signed_laplacian_psd() {
        let g = two_balanced_components();
        let l = signed_laplacian(&g);
        let n = l.len();
        for i in 0..n {
            assert!(l[i][i] >= 0.0, "diagonal[{i}] should be non-negative");
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (l[i][j] - l[j][i]).abs();
                assert!(diff < 1e-12, "L not symmetric at ({i},{j})");
            }
        }
        // PSD: x^T L x >= 0
        let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let lx = mat_vec(&l, &x);
        let quad: f64 = dot(&x, &lx);
        assert!(quad >= -1e-10, "signed Laplacian should be PSD, got {quad}");
    }

    #[test]
    fn test_sponge_embedding_shape() {
        let g = two_balanced_components();
        let dim = 2;
        let emb = sponge_embedding(&g, dim, 0.1);
        assert_eq!(emb.n_nodes, 6);
        assert_eq!(emb.embeddings.len(), 6);
        for row in &emb.embeddings {
            assert!(!row.is_empty(), "embedding row should not be empty");
        }
    }

    #[test]
    fn test_sponge_embedding_separates() {
        let g = two_balanced_components();
        let emb = sponge_embedding(&g, 2, 0.1);
        // Check that the two groups are separated in some embedding dimension
        let mut separated = false;
        for d in 0..emb.dim {
            let a_vals: Vec<f64> = emb.embeddings[0..3].iter().map(|v| v[d]).collect();
            let b_vals: Vec<f64> = emb.embeddings[3..6].iter().map(|v| v[d]).collect();
            let a_mean = a_vals.iter().sum::<f64>() / 3.0;
            let b_mean = b_vals.iter().sum::<f64>() / 3.0;
            if (a_mean - b_mean).abs() > 0.05 {
                separated = true;
                break;
            }
        }
        assert!(
            separated,
            "SPONGE failed to separate groups in any dimension. embeddings: {:?}",
            emb.embeddings
        );
    }

    #[test]
    fn test_signed_ratio_cut() {
        let g = two_balanced_components();
        let labels = signed_ratio_cut(&g, 2);
        assert_eq!(labels.len(), 6);
        // Within-group consistency
        assert_eq!(labels[0], labels[1], "nodes 0 and 1 should share a cluster");
        assert_eq!(labels[0], labels[2], "nodes 0 and 2 should share a cluster");
        assert_eq!(labels[3], labels[4], "nodes 3 and 4 should share a cluster");
        assert_eq!(labels[3], labels[5], "nodes 3 and 5 should share a cluster");
        assert_ne!(
            labels[0], labels[3],
            "nodes 0 and 3 should be in different clusters"
        );
    }

    #[test]
    fn test_status_score_directed() {
        let mut g = SignedGraph::new(3);
        g.add_edge(0, 1, 1);
        g.add_edge(1, 2, 1);
        let s = status_score(&g);
        assert_eq!(s.len(), 3);
        assert!(
            s[2] > s[1],
            "status(2) > status(1): {:.3} vs {:.3}",
            s[2],
            s[1]
        );
        assert!(
            s[1] > s[0],
            "status(1) > status(0): {:.3} vs {:.3}",
            s[1],
            s[0]
        );
    }

    #[test]
    fn test_balance_theory() {
        let mut g = SignedGraph::new(3);
        g.add_edge(0, 1, 1);
        g.add_edge(1, 2, 1);
        g.add_edge(0, 2, 1);
        let labels = signed_ratio_cut(&g, 2);
        let same_count = [(0, 1), (1, 2), (0, 2)]
            .iter()
            .filter(|&&(a, b)| labels[a] == labels[b])
            .count();
        assert!(
            same_count >= 1,
            "balance: some connected nodes should share a cluster"
        );
    }

    #[test]
    fn test_signed_embedding_config_default() {
        let cfg = crate::signed_directed::types::SignedEmbedConfig::default();
        assert!(cfg.dim > 0);
        assert!(cfg.n_iter > 0);
        assert!(cfg.lr > 0.0);
    }

    #[test]
    fn test_signed_graph_construction() {
        let g = two_balanced_components();
        assert_eq!(g.n_nodes, 6);
        assert_eq!(g.positive_edge_count(), 6);
        assert_eq!(g.negative_edge_count(), 3);
        // Each node in group A should have 2 pos neighbours + 1 neg neighbour
        assert_eq!(g.pos_adj[0].len(), 2);
        assert_eq!(g.neg_adj[0].len(), 1);
    }
}
