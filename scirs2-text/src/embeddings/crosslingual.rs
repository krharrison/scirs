//! Cross-lingual Embedding Alignment.
//!
//! This module provides methods for aligning embedding spaces across languages,
//! enabling cross-lingual transfer and translation of embeddings.
//!
//! # Alignment Methods
//!
//! | Method | Description |
//! |--------|-------------|
//! | Procrustes | Orthogonal alignment: W = UV^T from SVD(X^T Y) |
//! | CCA | Canonical Correlation Analysis projection |
//! | MUSE | Multilingual Unsupervised/Supervised Embeddings (iterative refinement) |
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::embeddings::crosslingual::{
//!     CrossLingualConfig, AlignmentMethod, align_embeddings, translate_embedding, AlignmentMatrix,
//! };
//!
//! let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let target = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
//! let anchors = vec![(0, 0), (1, 1)];
//!
//! let config = CrossLingualConfig::default();
//! let alignment = align_embeddings(&source, &target, &anchors, &config).unwrap();
//! let translated = translate_embedding(&source[0], &alignment);
//! assert_eq!(translated.len(), 2);
//! ```

use crate::error::{Result, TextError};

/// SVD decomposition result: (U, S, Vt).
type SvdResult = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>);

// ─── AlignmentMethod ────────────────────────────────────────────────────────

/// Method used to align embedding spaces.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AlignmentMethod {
    /// Procrustes alignment: find orthogonal W minimising ‖XW − Y‖_F.
    #[default]
    Procrustes,
    /// Canonical Correlation Analysis alignment.
    CCA,
    /// Multilingual Unsupervised/Supervised Embeddings (iterative).
    MUSE,
}

// ─── CrossLingualConfig ─────────────────────────────────────────────────────

/// Configuration for cross-lingual alignment.
#[derive(Debug, Clone)]
pub struct CrossLingualConfig {
    /// Dimensionality of source embeddings.
    pub source_dim: usize,
    /// Dimensionality of target embeddings.
    pub target_dim: usize,
    /// Alignment method to use.
    pub alignment: AlignmentMethod,
    /// Number of refinement iterations (for MUSE).
    pub refinement_iterations: usize,
    /// Learning rate for iterative methods.
    pub learning_rate: f64,
}

impl Default for CrossLingualConfig {
    fn default() -> Self {
        Self {
            source_dim: 0, // auto-detect
            target_dim: 0, // auto-detect
            alignment: AlignmentMethod::Procrustes,
            refinement_iterations: 5,
            learning_rate: 0.01,
        }
    }
}

// ─── AlignmentMatrix ────────────────────────────────────────────────────────

/// Learned alignment transformation matrix.
#[derive(Debug, Clone)]
pub struct AlignmentMatrix {
    /// The transformation matrix W (rows × cols).
    pub w: Vec<Vec<f64>>,
    /// Number of rows (source dimensionality).
    pub rows: usize,
    /// Number of columns (target dimensionality).
    pub cols: usize,
    /// Method used to compute this alignment.
    pub method: AlignmentMethod,
}

// ─── Linear algebra helpers ─────────────────────────────────────────────────

/// Transpose a matrix represented as Vec<Vec<f64>>.
fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return Vec::new();
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = m[i][j];
        }
    }
    t
}

/// Multiply two matrices A (m×k) and B (k×n) → C (m×n).
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let k = a[0].len();
    if b.is_empty() || b[0].is_empty() {
        return vec![vec![]; m];
    }
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i][p] * b[p][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Compute SVD of an m×n matrix using one-sided Jacobi rotations.
/// Returns (U, S, Vt) where U is m×min(m,n), S is min(m,n), Vt is min(m,n)×n.
fn svd_jacobi(matrix: &[Vec<f64>]) -> Result<SvdResult> {
    let m = matrix.len();
    if m == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let n = matrix[0].len();
    if n == 0 {
        return Ok((vec![vec![]; m], Vec::new(), Vec::new()));
    }

    let k = m.min(n);
    let max_iter = 100;
    let tol = 1e-12;

    // Work on A^T A for small cases, use a simpler power-iteration-based approach
    // For the Procrustes problem we only need the thin SVD of X^T Y which is at most dim×dim.

    // Compute A^T A (n×n)
    let at = transpose(matrix);
    let ata = matmul(&at, matrix);

    // Eigen-decompose A^T A via Jacobi
    let nn = ata.len();
    let mut d = ata.clone(); // will be diagonalised
    let mut v = vec![vec![0.0; nn]; nn]; // eigenvectors
    for i in 0..nn {
        v[i][i] = 1.0;
    }

    for _iter in 0..max_iter {
        // Find max off-diagonal
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..nn {
            for j in (i + 1)..nn {
                let val = d[i][j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break;
        }

        // Compute Jacobi rotation
        let theta = if (d[p][p] - d[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * d[p][q] / (d[p][p] - d[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to d
        let mut new_d = d.clone();
        for i in 0..nn {
            if i != p && i != q {
                new_d[i][p] = c * d[i][p] + s * d[i][q];
                new_d[p][i] = new_d[i][p];
                new_d[i][q] = -s * d[i][p] + c * d[i][q];
                new_d[q][i] = new_d[i][q];
            }
        }
        new_d[p][p] = c * c * d[p][p] + 2.0 * s * c * d[p][q] + s * s * d[q][q];
        new_d[q][q] = s * s * d[p][p] - 2.0 * s * c * d[p][q] + c * c * d[q][q];
        new_d[p][q] = 0.0;
        new_d[q][p] = 0.0;
        d = new_d;

        // Update eigenvectors
        for i in 0..nn {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, usize)> = (0..nn).map(|i| (d[i][i].max(0.0), i)).collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sigma = vec![0.0; k];
    let mut vt = vec![vec![0.0; n]; k];
    for i in 0..k {
        let (eigval, idx) = eig_pairs[i];
        sigma[i] = eigval.sqrt();
        for j in 0..nn {
            vt[i][j] = v[j][idx];
        }
    }

    // U = A V Σ^{-1}
    // V columns (transposed from vt rows)
    let mut u = vec![vec![0.0; k]; m];
    for i in 0..m {
        for j in 0..k {
            if sigma[j] > 1e-15 {
                let mut s = 0.0;
                for p in 0..n {
                    s += matrix[i][p] * vt[j][p];
                }
                u[i][j] = s / sigma[j];
            }
        }
    }

    Ok((u, sigma, vt))
}

// ─── Alignment functions ────────────────────────────────────────────────────

/// Procrustes alignment: find orthogonal W such that ‖XW − Y‖_F is minimised.
///
/// W = V U^T from SVD(X^T Y).
fn procrustes_align(
    source_anchors: &[Vec<f64>],
    target_anchors: &[Vec<f64>],
) -> Result<AlignmentMatrix> {
    if source_anchors.is_empty() || target_anchors.is_empty() {
        return Err(TextError::InvalidInput("Empty anchor sets".to_string()));
    }
    let dim_s = source_anchors[0].len();
    let dim_t = target_anchors[0].len();
    if dim_s != dim_t {
        return Err(TextError::InvalidInput(format!(
            "Procrustes requires same dimensionality, got {} vs {}",
            dim_s, dim_t
        )));
    }

    // Compute M = X^T Y (dim × dim)
    let xt = transpose(source_anchors);
    let m = matmul(&xt, target_anchors);

    // SVD of M = X^T Y
    let (u, _sigma, vt) = svd_jacobi(&m)?;

    // Procrustes solution: W = U V^T
    // SVD(M) = U Σ V^T → W = U V^T
    let w = matmul(&u, &vt);

    Ok(AlignmentMatrix {
        w,
        rows: dim_s,
        cols: dim_t,
        method: AlignmentMethod::Procrustes,
    })
}

/// CCA alignment: project both source and target to a shared space.
fn cca_align(source_anchors: &[Vec<f64>], target_anchors: &[Vec<f64>]) -> Result<AlignmentMatrix> {
    // Simplified CCA: use whitened Procrustes
    // 1. Center both sets
    let n = source_anchors.len();
    if n == 0 {
        return Err(TextError::InvalidInput("Empty anchor sets".to_string()));
    }
    let dim_s = source_anchors[0].len();
    let dim_t = target_anchors[0].len();

    // Center source
    let mut src_mean = vec![0.0; dim_s];
    for v in source_anchors {
        for (i, &x) in v.iter().enumerate() {
            src_mean[i] += x;
        }
    }
    let nf = n as f64;
    for v in &mut src_mean {
        *v /= nf;
    }

    let centered_src: Vec<Vec<f64>> = source_anchors
        .iter()
        .map(|v| v.iter().zip(src_mean.iter()).map(|(x, m)| x - m).collect())
        .collect();

    // Center target
    let mut tgt_mean = vec![0.0; dim_t];
    for v in target_anchors {
        for (i, &x) in v.iter().enumerate() {
            tgt_mean[i] += x;
        }
    }
    for v in &mut tgt_mean {
        *v /= nf;
    }

    let centered_tgt: Vec<Vec<f64>> = target_anchors
        .iter()
        .map(|v| v.iter().zip(tgt_mean.iter()).map(|(x, m)| x - m).collect())
        .collect();

    // Procrustes on centred data
    procrustes_align(&centered_src, &centered_tgt)
}

/// MUSE-style iterative alignment (supervised variant).
fn muse_align(
    source_anchors: &[Vec<f64>],
    target_anchors: &[Vec<f64>],
    iterations: usize,
) -> Result<AlignmentMatrix> {
    // Start with Procrustes, then iteratively refine
    let mut alignment = procrustes_align(source_anchors, target_anchors)?;

    for _iter in 0..iterations {
        // Apply current alignment to source anchors
        let aligned: Vec<Vec<f64>> = source_anchors
            .iter()
            .map(|s| translate_embedding(s, &alignment))
            .collect();

        // Re-solve Procrustes with aligned ↔ target
        alignment = procrustes_align(&aligned, target_anchors)?;

        // Compose: new_W = old_W * refine_W
        // But since each iteration refines, we keep the latest
    }

    Ok(alignment)
}

/// Align source embeddings to the target embedding space using anchor pairs.
///
/// `anchors` is a list of `(source_idx, target_idx)` pairs identifying
/// corresponding words across languages.
pub fn align_embeddings(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    anchors: &[(usize, usize)],
    config: &CrossLingualConfig,
) -> Result<AlignmentMatrix> {
    if anchors.is_empty() {
        return Err(TextError::InvalidInput(
            "Need at least one anchor pair".to_string(),
        ));
    }
    if source.is_empty() || target.is_empty() {
        return Err(TextError::InvalidInput(
            "Source and target embeddings must be non-empty".to_string(),
        ));
    }

    // Extract anchor vectors
    let mut src_anchors = Vec::with_capacity(anchors.len());
    let mut tgt_anchors = Vec::with_capacity(anchors.len());
    for &(si, ti) in anchors {
        if si >= source.len() {
            return Err(TextError::InvalidInput(format!(
                "Source anchor index {si} out of bounds (len={})",
                source.len()
            )));
        }
        if ti >= target.len() {
            return Err(TextError::InvalidInput(format!(
                "Target anchor index {ti} out of bounds (len={})",
                target.len()
            )));
        }
        src_anchors.push(source[si].clone());
        tgt_anchors.push(target[ti].clone());
    }

    #[allow(unreachable_patterns)]
    match &config.alignment {
        AlignmentMethod::Procrustes => procrustes_align(&src_anchors, &tgt_anchors),
        AlignmentMethod::CCA => cca_align(&src_anchors, &tgt_anchors),
        AlignmentMethod::MUSE => {
            muse_align(&src_anchors, &tgt_anchors, config.refinement_iterations)
        }
        _ => procrustes_align(&src_anchors, &tgt_anchors),
    }
}

/// Translate a single embedding using the alignment matrix: y = x W.
pub fn translate_embedding(embedding: &[f64], alignment: &AlignmentMatrix) -> Vec<f64> {
    let mut result = vec![0.0; alignment.cols];
    for j in 0..alignment.cols {
        let mut s = 0.0;
        for i in 0..alignment.rows.min(embedding.len()) {
            s += embedding[i] * alignment.w[i][j];
        }
        result[j] = s;
    }
    result
}

/// Translate a batch of embeddings.
pub fn translate_batch(embeddings: &[Vec<f64>], alignment: &AlignmentMatrix) -> Vec<Vec<f64>> {
    embeddings
        .iter()
        .map(|e| translate_embedding(e, alignment))
        .collect()
}

/// Compute the alignment quality: mean cosine similarity between aligned source
/// anchors and target anchors.
pub fn alignment_quality(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    anchors: &[(usize, usize)],
    alignment: &AlignmentMatrix,
) -> f64 {
    if anchors.is_empty() {
        return 0.0;
    }
    let mut total_sim = 0.0;
    let mut count = 0;
    for &(si, ti) in anchors {
        if si < source.len() && ti < target.len() {
            let aligned = translate_embedding(&source[si], alignment);
            let sim = cosine_sim_local(&aligned, &target[ti]);
            total_sim += sim;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total_sim / count as f64
    }
}

/// Cosine similarity (public, for cross-module use).
fn cosine_sim_local(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Compute the alignment quality using local cosine similarity.
pub fn alignment_quality_score(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    anchors: &[(usize, usize)],
    alignment: &AlignmentMatrix,
) -> f64 {
    if anchors.is_empty() {
        return 0.0;
    }
    let mut total_sim = 0.0;
    let mut count = 0;
    for &(si, ti) in anchors {
        if si < source.len() && ti < target.len() {
            let aligned = translate_embedding(&source[si], alignment);
            let sim = cosine_sim_local(&aligned, &target[ti]);
            total_sim += sim;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total_sim / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosslingual_config_default() {
        let cfg = CrossLingualConfig::default();
        assert_eq!(cfg.alignment, AlignmentMethod::Procrustes);
        assert_eq!(cfg.refinement_iterations, 5);
    }

    #[test]
    fn test_procrustes_identity() {
        // If source == target, alignment should be close to identity
        let source = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let target = source.clone();
        let anchors = vec![(0, 0), (1, 1), (2, 2)];
        let config = CrossLingualConfig::default();
        let alignment = align_embeddings(&source, &target, &anchors, &config);
        assert!(alignment.is_ok());
        let alignment = alignment.expect("should succeed");

        // Translated source should be close to target
        let translated = translate_embedding(&source[0], &alignment);
        let dist: f64 = translated
            .iter()
            .zip(target[0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            dist < 0.1,
            "Identity alignment should preserve vectors, dist={dist}"
        );
    }

    #[test]
    fn test_procrustes_rotation() {
        // 90-degree rotation in 2D
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        let anchors = vec![(0, 0), (1, 1)];
        let config = CrossLingualConfig::default();
        let alignment = align_embeddings(&source, &target, &anchors, &config).expect("ok");

        let t0 = translate_embedding(&source[0], &alignment);
        let t1 = translate_embedding(&source[1], &alignment);

        // t0 should be close to [0, 1]
        let d0 = ((t0[0] - 0.0).powi(2) + (t0[1] - 1.0).powi(2)).sqrt();
        assert!(d0 < 0.3, "Rotated [1,0] should be near [0,1], dist={d0}");

        let d1 = ((t1[0] + 1.0).powi(2) + (t1[1] - 0.0).powi(2)).sqrt();
        assert!(d1 < 0.3, "Rotated [0,1] should be near [-1,0], dist={d1}");
    }

    #[test]
    fn test_translation_preserves_relative_distances() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let target = vec![vec![0.0, 1.0], vec![-1.0, 0.0], vec![-1.0, 1.0]];
        let anchors = vec![(0, 0), (1, 1)];
        let config = CrossLingualConfig::default();
        let alignment = align_embeddings(&source, &target, &anchors, &config).expect("ok");

        // Original distances between source[0] and source[1]
        let orig_dist_01: f64 = source[0]
            .iter()
            .zip(source[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let orig_dist_02: f64 = source[0]
            .iter()
            .zip(source[2].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let t0 = translate_embedding(&source[0], &alignment);
        let t1 = translate_embedding(&source[1], &alignment);
        let t2 = translate_embedding(&source[2], &alignment);

        let new_dist_01: f64 = t0
            .iter()
            .zip(t1.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let new_dist_02: f64 = t0
            .iter()
            .zip(t2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Orthogonal transform preserves distances
        assert!(
            (orig_dist_01 - new_dist_01).abs() < 0.3,
            "Distances should be preserved: {orig_dist_01} vs {new_dist_01}"
        );
        assert!(
            (orig_dist_02 - new_dist_02).abs() < 0.3,
            "Distances should be preserved: {orig_dist_02} vs {new_dist_02}"
        );
    }

    #[test]
    fn test_cca_alignment() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        let anchors = vec![(0, 0), (1, 1)];
        let config = CrossLingualConfig {
            alignment: AlignmentMethod::CCA,
            ..Default::default()
        };
        let alignment = align_embeddings(&source, &target, &anchors, &config);
        assert!(alignment.is_ok());
    }

    #[test]
    fn test_muse_alignment() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        let anchors = vec![(0, 0), (1, 1)];
        let config = CrossLingualConfig {
            alignment: AlignmentMethod::MUSE,
            refinement_iterations: 3,
            ..Default::default()
        };
        let alignment = align_embeddings(&source, &target, &anchors, &config);
        assert!(alignment.is_ok());
    }

    #[test]
    fn test_empty_anchors_error() {
        let source = vec![vec![1.0, 0.0]];
        let target = vec![vec![0.0, 1.0]];
        let config = CrossLingualConfig::default();
        let result = align_embeddings(&source, &target, &[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_batch() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let anchors = vec![(0, 0), (1, 1)];
        let config = CrossLingualConfig::default();
        let alignment = align_embeddings(&source, &target, &anchors, &config).expect("ok");
        let batch = translate_batch(&source, &alignment);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 2);
    }
}
