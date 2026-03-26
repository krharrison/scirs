//! Sparse sketching transforms
//!
//! Provides efficient sketch matrices that can be applied in O(nnz) time:
//!
//! - **CountSketch**: each column of the sketch has exactly one non-zero entry (+/-1),
//!   mapping each row of the data to exactly one bucket.
//! - **OSNAP** (Oblivious Subspace Embedding): a block-diagonal sparse sketch where
//!   each block is an independent CountSketch. Provides stronger subspace embedding
//!   guarantees than plain CountSketch.
//! - **SparseJL**: sparse Johnson-Lindenstrauss transform with `s` non-zero entries
//!   per column, where s = O(log(n) / epsilon^2).
//!
//! ## References
//!
//! - Clarkson, K.L. & Woodruff, D.P. (2017). "Low-rank approximation and regression
//!   in input sparsity time"
//! - Nelson, J. & Nguyen, H.L. (2013). "OSNAP: Faster Numerical Linear Algebra
//!   Algorithms via Sparser Subspace Embeddings"
//! - Achlioptas, D. (2003). "Database-friendly random projections:
//!   Johnson-Lindenstrauss with binary coins"

use super::types::{SketchConfig, SketchTypeExt};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Sketch construction (returns flat row-major array of size s * m)
// ---------------------------------------------------------------------------

/// Build a sketch matrix (flat row-major, s rows x m columns) based on the
/// configured sketch type.
pub fn build_sketch(
    sketch_type: &SketchTypeExt,
    s: usize,
    m: usize,
    seed: u64,
    config: &SketchConfig,
) -> OptimizeResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    match sketch_type {
        SketchTypeExt::Gaussian => Ok(build_gaussian(s, m, &mut rng)),
        SketchTypeExt::SRHT => Ok(build_srht(s, m, &mut rng)),
        SketchTypeExt::CountSketch => Ok(build_count_sketch(s, m, &mut rng)),
        SketchTypeExt::OSNAP => {
            let blocks = if config.osnap_blocks == 0 {
                (s / 4).max(1)
            } else {
                config.osnap_blocks
            };
            Ok(build_osnap(s, m, blocks, &mut rng))
        }
        SketchTypeExt::SparseJL => {
            let sparsity = if config.sparse_jl_sparsity == 0 {
                // Default: O(log m)
                ((m as f64).ln().ceil() as usize).max(1)
            } else {
                config.sparse_jl_sparsity
            };
            Ok(build_sparse_jl(s, m, sparsity, &mut rng))
        }
        _ => {
            // Fallback: Gaussian
            Ok(build_gaussian(s, m, &mut rng))
        }
    }
}

/// Apply a flat sketch (s x m) to a matrix A (m rows, n cols each).
/// Returns the sketched rows as `Vec<Vec<f64>>` of length s.
pub fn apply_sketch(sketch: &[f64], s: usize, a: &[Vec<f64>]) -> OptimizeResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m == 0 {
        return Err(OptimizeError::InvalidInput("Empty matrix".into()));
    }
    let n = a[0].len();
    let mut result = vec![vec![0.0; n]; s];

    for k in 0..s {
        for i in 0..m {
            let s_ki = sketch[k * m + i];
            if s_ki.abs() > f64::EPSILON {
                for j in 0..n {
                    result[k][j] += s_ki * a[i][j];
                }
            }
        }
    }
    Ok(result)
}

/// Apply sketch to a single vector: Sv. Returns vector of length s.
pub fn apply_sketch_to_vec(sketch: &[f64], s: usize, v: &[f64]) -> Vec<f64> {
    let m = v.len();
    let mut result = vec![0.0; s];
    for k in 0..s {
        for i in 0..m {
            result[k] += sketch[k * m + i] * v[i];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Gaussian sketch
// ---------------------------------------------------------------------------

fn build_gaussian(s: usize, m: usize, rng: &mut StdRng) -> Vec<f64> {
    let scale = 1.0 / (s as f64).sqrt();
    let mut data = Vec::with_capacity(s * m);
    let mut spare: Option<f64> = None;
    for _ in 0..(s * m) {
        let v = match spare.take() {
            Some(z) => z,
            None => loop {
                let u: f64 = rng.random::<f64>();
                let w: f64 = rng.random::<f64>();
                if u > 0.0 {
                    let mag = (-2.0 * u.ln()).sqrt();
                    let angle = std::f64::consts::TAU * w;
                    spare = Some(mag * angle.sin());
                    break mag * angle.cos();
                }
            },
        };
        data.push(v * scale);
    }
    data
}

// ---------------------------------------------------------------------------
// CountSketch
// ---------------------------------------------------------------------------

fn build_count_sketch(s: usize, m: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut data = vec![0.0; s * m];
    for j in 0..m {
        let row = rng.random_range(0..s);
        let sign: f64 = if rng.random::<bool>() { 1.0 } else { -1.0 };
        data[row * m + j] = sign;
    }
    data
}

// ---------------------------------------------------------------------------
// SRHT (Subsampled Randomized Hadamard Transform)
// ---------------------------------------------------------------------------

fn build_srht(s: usize, m: usize, rng: &mut StdRng) -> Vec<f64> {
    let m_pad = m.next_power_of_two();
    let scale = (m_pad as f64 / s as f64).sqrt() / (m_pad as f64).sqrt();

    // Random signs
    let signs: Vec<f64> = (0..m_pad)
        .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
        .collect();

    // Sample s rows
    let mut perm: Vec<usize> = (0..m_pad).collect();
    for i in 0..s.min(m_pad) {
        let j = i + rng.random_range(0..(m_pad - i));
        perm.swap(i, j);
    }
    let selected: Vec<usize> = perm[..s.min(m_pad)].to_vec();

    // Build sketch: for each column j of A (0..m), apply D then H, then pick rows
    let mut data = vec![0.0; s * m];
    for j in 0..m {
        let mut col = vec![0.0; m_pad];
        col[j] = signs[j];
        walsh_hadamard(&mut col);
        for (k, &row_idx) in selected.iter().enumerate() {
            data[k * m + j] = scale * col[row_idx];
        }
    }
    data
}

fn walsh_hadamard(x: &mut [f64]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..(i + h) {
                let u = x[j];
                let v = x[j + h];
                x[j] = u + v;
                x[j + h] = u - v;
            }
        }
        h <<= 1;
    }
    let inv = 1.0 / (n as f64).sqrt();
    for xi in x.iter_mut() {
        *xi *= inv;
    }
}

// ---------------------------------------------------------------------------
// OSNAP (Oblivious Subspace Embedding)
// ---------------------------------------------------------------------------

/// Build an OSNAP sketch: block-diagonal structure where the m columns are
/// partitioned into `num_blocks` groups, and each group gets an independent
/// CountSketch-like mapping.
fn build_osnap(s: usize, m: usize, num_blocks: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut data = vec![0.0; s * m];
    let block_cols = (m + num_blocks - 1) / num_blocks;
    let block_rows = (s + num_blocks - 1) / num_blocks;

    for block in 0..num_blocks {
        let col_start = block * block_cols;
        let col_end = ((block + 1) * block_cols).min(m);
        let row_start = block * block_rows;
        let row_end = ((block + 1) * block_rows).min(s);
        let bk_rows = row_end - row_start;

        if bk_rows == 0 {
            continue;
        }

        for j in col_start..col_end {
            let local_row = rng.random_range(0..bk_rows);
            let global_row = row_start + local_row;
            let sign: f64 = if rng.random::<bool>() { 1.0 } else { -1.0 };
            // Scale by sqrt(num_blocks) to maintain expected norm
            let scale = (num_blocks as f64).sqrt();
            data[global_row * m + j] = sign * scale / (s as f64).sqrt();
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Sparse Johnson-Lindenstrauss
// ---------------------------------------------------------------------------

/// Build a sparse JL matrix: each column has exactly `sparsity` non-zero entries,
/// each drawn as +/-1 / sqrt(sparsity).
fn build_sparse_jl(s: usize, m: usize, sparsity: usize, rng: &mut StdRng) -> Vec<f64> {
    let nnz = sparsity.min(s);
    let scale = 1.0 / (nnz as f64).sqrt();
    let mut data = vec![0.0; s * m];

    for j in 0..m {
        // Pick `nnz` distinct rows for column j
        let mut rows: Vec<usize> = (0..s).collect();
        // Fisher-Yates partial shuffle
        for i in 0..nnz.min(s) {
            let k = i + rng.random_range(0..(s - i));
            rows.swap(i, k);
        }
        for i in 0..nnz {
            let row = rows[i];
            let sign: f64 = if rng.random::<bool>() { scale } else { -scale };
            data[row * m + j] = sign;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_sketch_preserves_norms_jl_property() {
        // JL property: for a fixed vector x, ||Sx|| should be close to ||x||
        // with high probability when s is large enough.
        let m = 20;
        let s = 50; // generous sketch dimension
        let x: Vec<f64> = (0..m).map(|i| (i as f64 + 1.0) / m as f64).collect();
        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Average over multiple trials
        let trials = 20;
        let mut total_ratio = 0.0;
        for trial in 0..trials {
            let mut rng = StdRng::seed_from_u64(100 + trial);
            let sketch = build_count_sketch(s, m, &mut rng);
            let sx = apply_sketch_to_vec(&sketch, s, &x);
            let sx_norm: f64 = sx.iter().map(|v| v * v).sum::<f64>().sqrt();
            total_ratio += sx_norm / x_norm;
        }
        let avg_ratio = total_ratio / trials as f64;
        // On average, the ratio should be close to 1.0
        assert!((avg_ratio - 1.0).abs() < 0.5, "avg ratio = {}", avg_ratio);
    }

    #[test]
    fn test_srht_output_dimensions() {
        let m = 10;
        let s = 4;
        let mut rng = StdRng::seed_from_u64(42);
        let sketch = build_srht(s, m, &mut rng);
        assert_eq!(sketch.len(), s * m);

        // Apply to identity-like rows
        let a: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut row = vec![0.0; 3];
                row[i % 3] = 1.0;
                row
            })
            .collect();
        let sa = apply_sketch(&sketch, s, &a).expect("should work");
        assert_eq!(sa.len(), s);
        assert_eq!(sa[0].len(), 3);
    }

    #[test]
    fn test_sparse_jl_norm_preservation() {
        let m = 30;
        let s = 40;
        let sparsity = 5;
        let x: Vec<f64> = (0..m).map(|i| i as f64 + 1.0).collect();
        let x_norm_sq: f64 = x.iter().map(|v| v * v).sum();

        let mut rng = StdRng::seed_from_u64(42);
        let sketch = build_sparse_jl(s, m, sparsity, &mut rng);
        let sx = apply_sketch_to_vec(&sketch, s, &x);
        let sx_norm_sq: f64 = sx.iter().map(|v| v * v).sum();

        // Check (1-epsilon) <= ||Sx||^2 / ||x||^2 <= (1+epsilon) with epsilon ~0.5
        let ratio = sx_norm_sq / x_norm_sq;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "sparse JL ratio = {} (should be near 1.0)",
            ratio
        );
    }

    #[test]
    fn test_osnap_dimensions() {
        let m = 12;
        let s = 8;
        let blocks = 3;
        let mut rng = StdRng::seed_from_u64(42);
        let sketch = build_osnap(s, m, blocks, &mut rng);
        assert_eq!(sketch.len(), s * m);
    }

    #[test]
    fn test_build_sketch_dispatch() {
        let config = SketchConfig {
            sketch_type: SketchTypeExt::Gaussian,
            sketch_size: 5,
            seed: 42,
            ..Default::default()
        };
        let s = build_sketch(&config.sketch_type, 5, 10, 42, &config);
        assert!(s.is_ok());
        assert_eq!(s.expect("ok").len(), 5 * 10);
    }

    #[test]
    fn test_apply_sketch_to_matrix() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let s = 2;
        let m = 3;
        // Simple sketch: identity-like (first two rows)
        let mut sketch = vec![0.0; s * m];
        sketch[0 * m + 0] = 1.0; // row 0 maps to row 0
        sketch[1 * m + 1] = 1.0; // row 1 maps to row 1
        let sa = apply_sketch(&sketch, s, &a).expect("should work");
        assert_eq!(sa.len(), 2);
        assert!((sa[0][0] - 1.0).abs() < 1e-10);
        assert!((sa[0][1] - 0.0).abs() < 1e-10);
        assert!((sa[1][0] - 0.0).abs() < 1e-10);
        assert!((sa[1][1] - 1.0).abs() < 1e-10);
    }
}
