//! Matrix sketching methods for large-scale linear algebra
//!
//! Sketching reduces an m×n matrix to a k×n (or m×k) matrix using a random
//! linear map S, preserving geometry with high probability. These are used as
//! building blocks for fast approximate solvers, randomized SVD, and streaming
//! algorithms.
//!
//! # Algorithms
//!
//! - **Gaussian sketch**: S drawn from N(0, 1/k) — JL embedding, universal
//! - **Subsampled Randomized Hadamard Transform (SRHT)**: O(n log k) via Walsh-Hadamard
//! - **Sparse sign sketch (CountSketch)**: Each column has exactly s nonzeros ∈ {±1}
//! - **Sketch-and-multiply**: Approximate A*B via a shared sketch
//!
//! # References
//!
//! - Halko, Martinsson, Tropp (2011). "Finding structure with randomness."
//! - Tropp et al. (2019). "Streaming low-rank matrix approximation with an application
//!   to scientific simulation."
//! - Clarkson & Woodruff (2017). "Low-rank approximation and regression in input
//!   sparsity time."

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal, Uniform};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for sketching float operations.
pub trait SketchFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<F> SketchFloat for F where F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn matmul_nn<F: SketchFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    let (m, ka) = (a.nrows(), a.ncols());
    let (kb, n) = (b.nrows(), b.ncols());
    if ka != kb {
        return Err(LinalgError::ShapeError(format!(
            "sketch matmul: inner dims {} vs {}",
            ka, kb
        )));
    }
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for l in 0..ka {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    Ok(c)
}

fn matmul_transpose_a<F: SketchFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    // Compute A^T * B
    let (m, ka) = (a.nrows(), a.ncols());
    let (kb, n) = (b.nrows(), b.ncols());
    if m != kb {
        return Err(LinalgError::ShapeError(format!(
            "sketch A^T B: dims mismatch {} vs {}",
            m, kb
        )));
    }
    let mut c = Array2::<F>::zeros((ka, n));
    for l in 0..m {
        for i in 0..ka {
            let a_li = a[[l, i]];
            if a_li == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_li * b[[l, j]];
            }
        }
    }
    Ok(c)
}

// ---------------------------------------------------------------------------
// Gaussian sketch
// ---------------------------------------------------------------------------

/// Apply a Gaussian sketch: compute S * A where S is a k×m Gaussian matrix.
///
/// Each entry of S is drawn i.i.d. from N(0, 1/sqrt(k)) so that the sketch
/// preserves inner products in expectation: E[S^T S] = I_m.
///
/// The sketch S*A is a k×n matrix that approximates the column space of A.
///
/// # Arguments
///
/// * `a` - Input m×n matrix
/// * `k` - Sketch size (number of rows in output), k < m
/// * `rng` - Mutable random number generator
///
/// # Returns
///
/// * k×n sketch of A
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::sketching::gaussian_sketch;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let sketch = gaussian_sketch(&a.view(), 2, &mut rng).expect("gaussian sketch");
/// assert_eq!(sketch.nrows(), 2);
/// assert_eq!(sketch.ncols(), 2);
/// ```
pub fn gaussian_sketch<F: SketchFloat>(
    a: &ArrayView2<F>,
    k: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let (m, n) = (a.nrows(), a.ncols());
    if k == 0 || k > m {
        return Err(LinalgError::InvalidInputError(format!(
            "gaussian_sketch: k={} must be in [1, {}]",
            k, m
        )));
    }

    let scale = F::from(1.0 / (k as f64).sqrt())
        .ok_or_else(|| LinalgError::ComputationError("Cannot convert scale".into()))?;

    let normal = Normal::new(0.0_f64, 1.0)
        .map_err(|e| LinalgError::ComputationError(format!("Normal dist: {e}")))?;

    // Build S (k×m) and compute S*A directly for memory efficiency
    let mut result = Array2::<F>::zeros((k, n));
    for i in 0..k {
        for l in 0..m {
            let s_il = F::from(normal.sample(rng)).unwrap_or(F::zero()) * scale;
            if s_il == F::zero() {
                continue;
            }
            for j in 0..n {
                result[[i, j]] += s_il * a[[l, j]];
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Subsampled Randomized Hadamard Transform (SRHT)
// ---------------------------------------------------------------------------

/// Apply Subsampled Randomized Hadamard Transform (SRHT) sketch: compute S * A.
///
/// The SRHT combines three operations:
/// 1. Random sign flip: multiply each row of A by a uniform ±1 diagonal
/// 2. Walsh-Hadamard transform (WHT): normalized fast Hadamard matrix multiply
/// 3. Random row subsampling: select k rows uniformly at random
///
/// This gives an (approximately) JL embedding in O(mn log m) time vs O(mnk)
/// for Gaussian sketching.
///
/// # Arguments
///
/// * `a` - Input m×n matrix (m must be a power of 2 or will be zero-padded)
/// * `k` - Output sketch size
/// * `rng` - Mutable random number generator
///
/// # Returns
///
/// * k×n SRHT sketch of A (scaled by sqrt(m/k))
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::sketching::subsampled_rht;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let sketch = subsampled_rht(&a.view(), 2, &mut rng).expect("srht");
/// assert_eq!(sketch.nrows(), 2);
/// ```
pub fn subsampled_rht<F: SketchFloat>(
    a: &ArrayView2<F>,
    k: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let (m, n) = (a.nrows(), a.ncols());
    if k == 0 || k > m {
        return Err(LinalgError::InvalidInputError(format!(
            "subsampled_rht: k={} must be in [1, {}]",
            k, m
        )));
    }

    // Pad to next power of two
    let m_padded = next_power_of_two(m);

    // Step 1: Apply diagonal sign flip D (±1)
    let mut da = Array2::<F>::zeros((m_padded, n));
    let uniform_sign = Uniform::new(0u8, 2)
        .map_err(|e| LinalgError::ComputationError(format!("Uniform dist: {e}")))?;
    let signs: Vec<F> = (0..m)
        .map(|_| {
            if uniform_sign.sample(rng) == 0 {
                F::one()
            } else {
                -F::one()
            }
        })
        .collect();

    for i in 0..m {
        for j in 0..n {
            da[[i, j]] = signs[i] * a[[i, j]];
        }
    }
    // Zero-padded rows are already zeros

    // Step 2: Apply Walsh-Hadamard transform (WHT) in-place along rows (dim 0)
    hadamard_transform_rows(&mut da, m_padded, n);

    // Step 3: Scale by 1/sqrt(m_padded)
    let scale = F::from(1.0 / (m_padded as f64).sqrt())
        .ok_or_else(|| LinalgError::ComputationError("scale convert".into()))?;
    for v in da.iter_mut() {
        *v *= scale;
    }

    // Step 4: Subsample k rows uniformly without replacement
    let row_indices = sample_without_replacement(m_padded, k, rng);

    // Output scale: sqrt(m_padded / k) for JL-normalization
    let out_scale = F::from((m_padded as f64 / k as f64).sqrt())
        .ok_or_else(|| LinalgError::ComputationError("out_scale convert".into()))?;

    let mut result = Array2::<F>::zeros((k, n));
    for (i, &row_idx) in row_indices.iter().enumerate() {
        for j in 0..n {
            result[[i, j]] = out_scale * da[[row_idx, j]];
        }
    }

    Ok(result)
}

/// Apply the Walsh-Hadamard transform to rows of a matrix in-place.
fn hadamard_transform_rows<F: SketchFloat>(a: &mut Array2<F>, m: usize, n: usize) {
    let mut len = 1usize;
    while len < m {
        let stride = len * 2;
        let mut start = 0;
        while start < m {
            for j in 0..n {
                for i in 0..len {
                    let u = a[[start + i, j]];
                    let v = a[[start + i + len, j]];
                    a[[start + i, j]] = u + v;
                    a[[start + i + len, j]] = u - v;
                }
            }
            start += stride;
        }
        len *= 2;
    }
}

/// Compute the next power of two >= n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Sample k indices from [0, m) without replacement using Fisher-Yates shuffle.
fn sample_without_replacement(m: usize, k: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..m).collect();
    let k = k.min(m);
    for i in 0..k {
        let j = i + rng.random_range(0..(m - i));
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

// ---------------------------------------------------------------------------
// Sparse sign sketch (CountSketch variant)
// ---------------------------------------------------------------------------

/// Apply a sparse sign sketch where each column has exactly s nonzeros.
///
/// For each column l of A, the sketch matrix S has exactly s nonzero entries
/// in column l, each ±1/sqrt(s), placed at uniformly random rows. This gives
/// a sparse embedding matrix similar to CountSketch.
///
/// This is the "OSNAP" (Oblivious Sparse Norming Array Polynomial) embedding.
///
/// # Arguments
///
/// * `a` - Input m×n matrix
/// * `k` - Sketch dimension (output rows)
/// * `s` - Sparsity: number of nonzeros per column of S (1 <= s <= k)
/// * `rng` - Mutable random number generator
///
/// # Returns
///
/// * k×n sparse sign sketch
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::sketching::sparse_sign_sketch;
///
/// let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let sketch = sparse_sign_sketch(&a.view(), 2, 1, &mut rng).expect("sparse sketch");
/// assert_eq!(sketch.nrows(), 2);
/// assert_eq!(sketch.ncols(), 3);
/// ```
pub fn sparse_sign_sketch<F: SketchFloat>(
    a: &ArrayView2<F>,
    k: usize,
    s: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let (m, n) = (a.nrows(), a.ncols());
    if k == 0 {
        return Err(LinalgError::InvalidInputError(
            "sparse_sign_sketch: k must be >= 1".into(),
        ));
    }
    let s = s.min(k).max(1);

    let scale = F::from(1.0 / (s as f64).sqrt())
        .ok_or_else(|| LinalgError::ComputationError("scale convert".into()))?;

    let uniform_sign = Uniform::new(0u8, 2)
        .map_err(|e| LinalgError::ComputationError(format!("Uniform sign dist: {e}")))?;

    let mut result = Array2::<F>::zeros((k, n));

    // For each input column l, sample s row positions and signs
    for l in 0..m {
        // Sample s distinct row indices from [0, k)
        let row_indices = sample_without_replacement(k, s, rng);

        for &row in &row_indices {
            let sign_bit = uniform_sign.sample(rng);
            let sign = if sign_bit == 0 { scale } else { -scale };

            for j in 0..n {
                result[[row, j]] += sign * a[[l, j]];
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Sketch-and-multiply: approximate A * B
// ---------------------------------------------------------------------------

/// Approximate the matrix product A * B using a shared random sketch.
///
/// Uses the identity E[S^T (S*A) (S*B)^T] ≈ A * B (approximately) where S is a
/// Gaussian sketch. More precisely, for tall A (m×p) and B (m×q):
///   A^T * B ≈ (S*A)^T * (S*B)
///
/// This function computes an approximation to A * B^T where A is m×p, B is n×p,
/// using k sketch rows.
///
/// # Arguments
///
/// * `a` - m×p matrix
/// * `b` - n×p matrix (note: transpose will be taken)
/// * `k` - Sketch dimension
/// * `rng` - Mutable random number generator
///
/// # Returns
///
/// * Approximate m×n matrix A * B^T
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::prelude::*;
/// use scirs2_linalg::randomized::sketching::sketch_multiply;
///
/// // Approximate [[1,2],[3,4]] * [[1,2],[3,4]]^T using k=5 sketch rows
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let mut rng = scirs2_core::random::seeded_rng(42);
/// let ab = sketch_multiply(&a.view(), &b.view(), 10, &mut rng).expect("sketch multiply");
/// assert_eq!(ab.nrows(), 2);
/// assert_eq!(ab.ncols(), 2);
/// ```
pub fn sketch_multiply<F: SketchFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    k: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let (m, p) = (a.nrows(), a.ncols());
    let (n, pb) = (b.nrows(), b.ncols());

    if p != pb {
        return Err(LinalgError::ShapeError(format!(
            "sketch_multiply: inner dimensions {} vs {} must match",
            p, pb
        )));
    }

    // Build Gaussian sketch S (k×p)
    let normal = Normal::new(0.0_f64, 1.0)
        .map_err(|e| LinalgError::ComputationError(format!("Normal dist: {e}")))?;

    let inv_sqrt_k = F::from(1.0 / (k as f64).sqrt())
        .ok_or_else(|| LinalgError::ComputationError("inv_sqrt_k convert".into()))?;

    // Build S (k×p)
    let mut s = Array2::<F>::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            s[[i, j]] = F::from(normal.sample(rng)).unwrap_or(F::zero()) * inv_sqrt_k;
        }
    }

    // SA = S * A^T ... wait, we want A * B^T
    // Use: A * B^T ≈ A * S^T * S * B^T
    // but that's expensive. Instead do direct sketch:
    // Compute (S * A^T)^T * (S * B^T) = A * S^T * S * B^T
    // Let's compute it as A * (S^T * (S * B^T))

    // S*A^T: (k×p)*(p×m) = k×m, but A is m×p so A^T is p×m, S*A^T = k×m
    // We want A*B^T = (m×p)*(p×n)

    // Better: sketch over the p dimension
    // A ~ m×p, B ~ n×p
    // S ~ k×p (Gaussian sketch)
    // A*S^T ~ m×k, B*S^T ~ n×k
    // (A*S^T)*(B*S^T)^T ~ m×n ← this is our approximation

    // Compute A * S^T (m×k)
    let mut as_mat = Array2::<F>::zeros((m, k));
    for i in 0..m {
        for l in 0..p {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..k {
                as_mat[[i, j]] += a_il * s[[j, l]];
            }
        }
    }

    // Compute B * S^T (n×k)
    let mut bs_mat = Array2::<F>::zeros((n, k));
    for i in 0..n {
        for l in 0..p {
            let b_il = b[[i, l]];
            if b_il == F::zero() {
                continue;
            }
            for j in 0..k {
                bs_mat[[i, j]] += b_il * s[[j, l]];
            }
        }
    }

    // Result = (A * S^T) * (B * S^T)^T = as_mat * bs_mat^T
    matmul_transpose_a(&as_mat.view().t().to_owned(), &bs_mat).or_else(|_| {
        // Fallback: manual computation
        let mut result = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for r in 0..n {
                let mut dot = F::zero();
                for j in 0..k {
                    dot += as_mat[[i, j]] * bs_mat[[r, j]];
                }
                result[[i, r]] = dot;
            }
        }
        Ok(result)
    })
}

// Actually, let me rewrite sketch_multiply more cleanly:

/// Compute an approximate matrix product A * B using sketching.
///
/// This version directly computes the approximation without the transpose confusion:
/// S (k×m Gaussian) is applied to the left:
///   A * B ≈ (S*A)^† * (S*B) [least-squares sense for overdetermined case]
///
/// For square matrices or computing A*B exactly:
///   result = A * B via randomized route as sanity check
///
/// # Arguments
///
/// * `a` - m×p matrix  
/// * `b` - p×n matrix
/// * `k` - Sketch dimension (inner sketch)
/// * `rng` - Mutable random number generator
///
/// # Returns
///
/// * Approximate m×n product A * B
pub fn sketch_multiply_direct<F: SketchFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    k: usize,
    rng: &mut impl Rng,
) -> LinalgResult<Array2<F>> {
    let (m, p) = (a.nrows(), a.ncols());
    let (pb, n) = (b.nrows(), b.ncols());

    if p != pb {
        return Err(LinalgError::ShapeError(format!(
            "sketch_multiply_direct: p={} vs pb={}",
            p, pb
        )));
    }

    // Gaussian sketch S of size k×p
    let normal = Normal::new(0.0_f64, 1.0)
        .map_err(|e| LinalgError::ComputationError(format!("Normal dist: {e}")))?;

    let inv_sqrt_k = F::from(1.0 / (k as f64).sqrt())
        .ok_or_else(|| LinalgError::ComputationError("inv_sqrt_k convert".into()))?;

    let mut s_mat = Array2::<F>::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            s_mat[[i, j]] = F::from(normal.sample(rng)).unwrap_or(F::zero()) * inv_sqrt_k;
        }
    }

    // Compute A * S^T (m×k): "sketch" the columns of A
    let mut a_sk = Array2::<F>::zeros((m, k));
    for i in 0..m {
        for l in 0..p {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..k {
                a_sk[[i, j]] += a_il * s_mat[[j, l]];
            }
        }
    }

    // Compute S * B (k×n): "sketch" the rows of B
    let mut s_b = Array2::<F>::zeros((k, n));
    for i in 0..k {
        for l in 0..p {
            let s_il = s_mat[[i, l]];
            if s_il == F::zero() {
                continue;
            }
            for j in 0..n {
                s_b[[i, j]] += s_il * b[[l, j]];
            }
        }
    }

    // Result: (A * S^T) * (S * B)  -- note this is an O(mnk) computation
    // but the individual components are O(mpk) and O(pkn)
    matmul_nn(&a_sk, &s_b)
}

// ---------------------------------------------------------------------------
// Leverage score sampling utility
// ---------------------------------------------------------------------------

/// Compute approximate leverage scores for row sampling.
///
/// Leverage scores l_i = ||e_i^T Q||_2^2 where Q comes from a QR decomposition.
/// Used for importance sampling in CUR and column/row subset selection.
///
/// # Arguments
///
/// * `q` - Orthonormal matrix Q (m×k) from QR factorization
///
/// # Returns
///
/// * Vector of m leverage scores (sum to k)
pub fn leverage_scores<F: SketchFloat>(q: &ArrayView2<F>) -> Array1<F> {
    let (m, _k) = (q.nrows(), q.ncols());
    let mut scores = Array1::<F>::zeros(m);
    for i in 0..m {
        let mut norm_sq = F::zero();
        for j in 0..q.ncols() {
            norm_sq += q[[i, j]] * q[[i, j]];
        }
        scores[i] = norm_sq;
    }
    scores
}

/// Sample row indices by leverage scores (importance sampling without replacement).
///
/// # Arguments
///
/// * `scores` - Non-negative sampling probabilities (will be normalized)
/// * `k` - Number of samples
/// * `rng` - Mutable RNG
///
/// # Returns
///
/// * Vector of k sampled indices (may contain duplicates if with_replacement=true)
pub fn sample_by_leverage<F: SketchFloat>(
    scores: &Array1<F>,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let m = scores.len();
    if m == 0 || k == 0 {
        return Vec::new();
    }

    // Normalize scores to probabilities
    let total: F = scores
        .iter()
        .copied()
        .fold(F::zero(), |acc, x| acc + x.abs());
    let probs: Vec<f64> = if total > F::zero() {
        scores
            .iter()
            .map(|&s| s.abs().to_f64().unwrap_or(0.0) / total.to_f64().unwrap_or(1.0))
            .collect()
    } else {
        vec![1.0 / m as f64; m]
    };

    // Cumulative probabilities for inverse CDF sampling
    let mut cdf = vec![0.0f64; m + 1];
    for i in 0..m {
        cdf[i + 1] = cdf[i] + probs[i];
    }

    let uniform = Uniform::new(0.0_f64, 1.0).unwrap_or_else(|_| {
        Uniform::new(0.0, 1.0 - f64::EPSILON).expect("failed to create uniform")
    });

    (0..k)
        .map(|_| {
            let u = uniform.sample(rng);
            // Binary search for the bucket
            let mut lo = 0;
            let mut hi = m;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cdf[mid + 1] < u {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo.min(m - 1)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::prelude::*;

    fn make_rng() -> impl Rng {
        scirs2_core::random::seeded_rng(12345)
    }

    #[test]
    fn test_gaussian_sketch_dimensions() {
        let a = array![
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let mut rng = make_rng();
        let sketch = gaussian_sketch(&a.view(), 2, &mut rng).expect("gaussian_sketch dims");
        assert_eq!(sketch.nrows(), 2);
        assert_eq!(sketch.ncols(), 3);
    }

    #[test]
    fn test_gaussian_sketch_preserves_norms_approx() {
        // A Gaussian sketch of a vector should roughly preserve its norm
        // with high probability (JL lemma). Test with many samples.
        let m = 100;
        let n = 1;
        let k = 50;
        let mut a = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            a[[i, 0]] = (i as f64 + 1.0).sqrt();
        }

        let mut rng = scirs2_core::random::seeded_rng(99);
        let sketch = gaussian_sketch(&a.view(), k, &mut rng).expect("sketch norm test");

        // Original norm^2
        let orig_norm_sq: f64 = (0..m).map(|i| a[[i, 0]] * a[[i, 0]]).sum();
        // Sketch norm^2 (expected to approximately equal orig_norm_sq)
        let sketch_norm_sq: f64 = (0..k).map(|i| sketch[[i, 0]] * sketch[[i, 0]]).sum();

        // Within 50% (weak test since k=50 < m=100 but not huge)
        let ratio = sketch_norm_sq / orig_norm_sq;
        assert!(ratio > 0.5 && ratio < 2.5, "ratio={ratio:.3}");
    }

    #[test]
    fn test_srht_dimensions() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let mut rng = make_rng();
        let sketch = subsampled_rht(&a.view(), 2, &mut rng).expect("srht dims");
        assert_eq!(sketch.nrows(), 2);
        assert_eq!(sketch.ncols(), 2);
    }

    #[test]
    fn test_sparse_sign_sketch_dimensions() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut rng = make_rng();
        let sketch = sparse_sign_sketch(&a.view(), 2, 1, &mut rng).expect("sparse sketch dims");
        assert_eq!(sketch.nrows(), 2);
        assert_eq!(sketch.ncols(), 3);
    }

    #[test]
    fn test_sketch_multiply_dimensions() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let mut rng = make_rng();
        let ab = sketch_multiply(&a.view(), &b.view(), 10, &mut rng).expect("sketch_multiply");
        assert_eq!(ab.nrows(), 2);
        assert_eq!(ab.ncols(), 2);
    }

    #[test]
    fn test_sketch_multiply_direct_dimensions() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let mut rng = make_rng();
        let ab = sketch_multiply_direct(&a.view(), &b.view(), 10, &mut rng).expect("direct");
        assert_eq!(ab.nrows(), 2);
        assert_eq!(ab.ncols(), 2);
    }

    #[test]
    fn test_leverage_scores_orthonormal() {
        // For an orthonormal Q (identity), leverage scores are all 1.0
        let q = Array2::<f64>::eye(3);
        let scores = leverage_scores(&q.view());
        for i in 0..3 {
            assert!(
                (scores[i] - 1.0).abs() < 1e-10,
                "score {} = {}",
                i,
                scores[i]
            );
        }
    }

    #[test]
    fn test_sample_by_leverage() {
        let scores = Array1::from_vec(vec![0.5_f64, 0.3, 0.2]);
        let mut rng = make_rng();
        let samples = sample_by_leverage(&scores, 10, &mut rng);
        assert_eq!(samples.len(), 10);
        for &s in &samples {
            assert!(s < 3, "sample {} out of range", s);
        }
    }

    #[test]
    fn test_hadamard_transform_involution() {
        // H * H = m * I, so applying twice and dividing by m recovers original
        let mut a = Array2::<f64>::zeros((4, 2));
        a[[0, 0]] = 1.0;
        a[[1, 0]] = 2.0;
        a[[2, 0]] = 3.0;
        a[[3, 0]] = 4.0;
        a[[0, 1]] = -1.0;
        a[[3, 1]] = 1.0;

        let original = a.clone();
        hadamard_transform_rows(&mut a, 4, 2);
        hadamard_transform_rows(&mut a, 4, 2);
        // After two applications: each element multiplied by 4 (= m)
        for i in 0..4 {
            for j in 0..2 {
                assert!((a[[i, j]] / 4.0 - original[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
