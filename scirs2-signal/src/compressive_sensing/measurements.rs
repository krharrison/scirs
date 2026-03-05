//! Measurement matrices for compressive sensing.
//!
//! Implements Gaussian, Bernoulli, partial-DCT, sparse JL, and RIP checking.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ──────────────────────────────────────────────────────────────────────────────
// Simple seeded PRNG (LCG + xorshift for better quality)
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal seeded PRNG implementing a simplified xorshift64 variant.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    /// Draw a pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Draw a value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Draw a standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64() + 1e-14).min(1.0 - 1e-14);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Draw uniformly from `{0, ..., n-1}`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle(&mut self, v: &mut Vec<usize>) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            v.swap(i, j);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gaussian measurement matrix
// ──────────────────────────────────────────────────────────────────────────────

/// Gaussian measurement matrix with normalised columns.
pub struct GaussianMatrix;

impl GaussianMatrix {
    /// Generate an `m × n` Gaussian matrix with normalised columns.
    ///
    /// Each entry is drawn from N(0, 1/m) and then each column is normalised to
    /// unit L2 norm.
    pub fn new(m: usize, n: usize) -> SignalResult<Vec<Vec<f64>>> {
        Self::with_seed(m, n, 42)
    }

    /// Generate with explicit seed.
    pub fn with_seed(m: usize, n: usize, seed: u64) -> SignalResult<Vec<Vec<f64>>> {
        if m == 0 || n == 0 {
            return Err(SignalError::InvalidArgument(
                "GaussianMatrix: dimensions must be > 0".into(),
            ));
        }
        let mut rng = Rng::new(seed);
        let scale = (m as f64).sqrt().recip();
        let mut mat = vec![vec![0.0_f64; n]; m];
        for j in 0..n {
            let mut col_norm_sq = 0.0_f64;
            for i in 0..m {
                let v = rng.next_normal() * scale;
                mat[i][j] = v;
                col_norm_sq += v * v;
            }
            let col_norm = col_norm_sq.sqrt().max(1e-14);
            for i in 0..m {
                mat[i][j] /= col_norm;
            }
        }
        Ok(mat)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bernoulli measurement matrix  (±1/sqrt(m))
// ──────────────────────────────────────────────────────────────────────────────

/// Bernoulli measurement matrix with entries ±1/√m.
pub struct BernoulliMatrix;

impl BernoulliMatrix {
    /// Generate an `m × n` Bernoulli matrix.
    pub fn new(m: usize, n: usize, rng: &mut Rng) -> SignalResult<Vec<Vec<f64>>> {
        if m == 0 || n == 0 {
            return Err(SignalError::InvalidArgument(
                "BernoulliMatrix: dimensions must be > 0".into(),
            ));
        }
        let scale = (m as f64).sqrt().recip();
        let mut mat = vec![vec![0.0_f64; n]; m];
        for i in 0..m {
            for j in 0..n {
                mat[i][j] = if rng.next_u64() & 1 == 0 { scale } else { -scale };
            }
        }
        Ok(mat)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DCT helper
// ──────────────────────────────────────────────────────────────────────────────

/// Compute one row of the DCT-II matrix: row `k` element `n` = sqrt(2/N) * cos(π(2n+1)k/(2N)).
fn dct2_entry(k: usize, n_idx: usize, n_total: usize) -> f64 {
    let scale = if k == 0 {
        (1.0 / n_total as f64).sqrt()
    } else {
        (2.0 / n_total as f64).sqrt()
    };
    scale * (PI * (2 * n_idx + 1) as f64 * k as f64 / (2.0 * n_total as f64)).cos()
}

// ──────────────────────────────────────────────────────────────────────────────
// Partial DCT measurement matrix
// ──────────────────────────────────────────────────────────────────────────────

/// Partial DCT measurement matrix: randomly selected rows of the DCT-II matrix.
pub struct PartialDCT;

impl PartialDCT {
    /// Generate an `m × n` partial DCT matrix by picking `m` random rows from
    /// the `n`-point DCT-II matrix.
    pub fn new(m: usize, n: usize, rng: &mut Rng) -> SignalResult<Vec<Vec<f64>>> {
        if m == 0 || n == 0 {
            return Err(SignalError::InvalidArgument(
                "PartialDCT: dimensions must be > 0".into(),
            ));
        }
        if m > n {
            return Err(SignalError::InvalidArgument(format!(
                "PartialDCT: m={m} cannot exceed n={n}"
            )));
        }
        let mut row_indices: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut row_indices);
        let selected = &row_indices[..m];

        let mat: Vec<Vec<f64>> = selected
            .iter()
            .map(|&k| (0..n).map(|j| dct2_entry(k, j, n)).collect())
            .collect();
        Ok(mat)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sparse JL transform
// ──────────────────────────────────────────────────────────────────────────────

/// Sparse Johnson-Lindenstrauss transform.
pub struct SparseJL;

impl SparseJL {
    /// Generate an `m × n` sparse JL matrix where each column has exactly `s` non-zeros
    /// drawn uniformly from ±1/√s.
    ///
    /// If `s >= m` the matrix degenerates to a scaled Bernoulli matrix.
    pub fn new(m: usize, n: usize, s: usize, rng: &mut Rng) -> SignalResult<Vec<Vec<f64>>> {
        if m == 0 || n == 0 {
            return Err(SignalError::InvalidArgument(
                "SparseJL: dimensions must be > 0".into(),
            ));
        }
        let s_eff = s.min(m).max(1);
        let scale = (s_eff as f64).sqrt().recip();
        let mut mat = vec![vec![0.0_f64; n]; m];

        for j in 0..n {
            // Choose s_eff distinct rows
            let mut row_indices: Vec<usize> = (0..m).collect();
            rng.shuffle(&mut row_indices);
            for &row in &row_indices[..s_eff] {
                mat[row][j] = if rng.next_u64() & 1 == 0 { scale } else { -scale };
            }
        }
        Ok(mat)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RIP check (approximate, Monte Carlo)
// ──────────────────────────────────────────────────────────────────────────────

/// Approximate RIP check: test the Restricted Isometry Property on random
/// `s`-sparse vectors.
///
/// Estimates the RIP constant δ_s by sampling `n_trials` random s-sparse unit
/// vectors x and computing `||(Ax)^2 - 1||_inf`.
///
/// Returns `true` if the estimated δ_s ≤ `delta`.
pub fn is_rip(a: &[Vec<f64>], s: usize, delta: f64) -> bool {
    let m = a.len();
    if m == 0 {
        return false;
    }
    let n = a[0].len();
    if n == 0 || s == 0 || s > n {
        return false;
    }

    let mut rng = Rng::new(1234);
    let n_trials = 200;
    let mut max_dev = 0.0_f64;

    for _ in 0..n_trials {
        // Build a random s-sparse unit vector
        let mut indices: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut indices);
        let support = &indices[..s];

        let mut x = vec![0.0_f64; n];
        let mut norm_sq = 0.0_f64;
        for &idx in support {
            let v = rng.next_normal();
            x[idx] = v;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt().max(1e-14);
        for v in &mut x { *v /= norm; }

        // Compute Ax
        let mut ax = vec![0.0_f64; m];
        for (i, row) in a.iter().enumerate() {
            ax[i] = row.iter().zip(x.iter()).map(|(&aij, &xj)| aij * xj).sum();
        }

        // ||Ax||^2
        let ax_norm_sq: f64 = ax.iter().map(|&v| v * v).sum();
        let dev = (ax_norm_sq - 1.0).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }

    max_dev <= delta
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_matrix_shape() {
        let mat = GaussianMatrix::new(10, 30).expect("gaussian");
        assert_eq!(mat.len(), 10);
        assert_eq!(mat[0].len(), 30);
    }

    #[test]
    fn test_gaussian_columns_normalised() {
        let mat = GaussianMatrix::new(20, 15).expect("gaussian");
        for j in 0..mat[0].len() {
            let norm_sq: f64 = mat.iter().map(|row| row[j] * row[j]).sum();
            assert!((norm_sq - 1.0).abs() < 1e-10, "col {j} norm_sq={norm_sq}");
        }
    }

    #[test]
    fn test_bernoulli_matrix_shape() {
        let mut rng = Rng::new(42);
        let mat = BernoulliMatrix::new(8, 20, &mut rng).expect("bernoulli");
        assert_eq!(mat.len(), 8);
        assert_eq!(mat[0].len(), 20);
    }

    #[test]
    fn test_bernoulli_entries() {
        let mut rng = Rng::new(7);
        let m = 5;
        let mat = BernoulliMatrix::new(m, 10, &mut rng).expect("bernoulli");
        let scale = (m as f64).sqrt().recip();
        for row in &mat {
            for &v in row {
                assert!((v.abs() - scale).abs() < 1e-12, "entry {v} != ±{scale}");
            }
        }
    }

    #[test]
    fn test_partial_dct_shape() {
        let mut rng = Rng::new(99);
        let mat = PartialDCT::new(8, 16, &mut rng).expect("partial_dct");
        assert_eq!(mat.len(), 8);
        assert_eq!(mat[0].len(), 16);
    }

    #[test]
    fn test_partial_dct_m_exceeds_n_error() {
        let mut rng = Rng::new(1);
        assert!(PartialDCT::new(20, 10, &mut rng).is_err());
    }

    #[test]
    fn test_sparse_jl_shape() {
        let mut rng = Rng::new(11);
        let mat = SparseJL::new(10, 30, 3, &mut rng).expect("sparse_jl");
        assert_eq!(mat.len(), 10);
        assert_eq!(mat[0].len(), 30);
    }

    #[test]
    fn test_sparse_jl_sparsity() {
        let mut rng = Rng::new(22);
        let s = 4;
        let mat = SparseJL::new(20, 10, s, &mut rng).expect("sparse_jl");
        for j in 0..10 {
            let nnz = mat.iter().filter(|row| row[j].abs() > 1e-12).count();
            assert_eq!(nnz, s, "col {j} has {nnz} nonzeros, expected {s}");
        }
    }

    #[test]
    fn test_rip_identity() {
        // n×n identity: RIP constant should be 0
        let n = 16_usize;
        let identity: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        assert!(is_rip(&identity, 1, 0.01), "identity should satisfy RIP with delta=0.01");
    }

    #[test]
    fn test_rip_gaussian() {
        // Gaussian matrix with m=40, n=64 should satisfy RIP with delta=0.5 for s=3
        let mat = GaussianMatrix::new(40, 64).expect("gaussian");
        assert!(is_rip(&mat, 3, 0.5), "Gaussian matrix should satisfy RIP");
    }

    #[test]
    fn test_zero_dimensions_error() {
        assert!(GaussianMatrix::new(0, 10).is_err());
        assert!(GaussianMatrix::new(10, 0).is_err());
        let mut rng = Rng::new(1);
        assert!(BernoulliMatrix::new(0, 5, &mut rng).is_err());
        assert!(SparseJL::new(0, 5, 2, &mut rng).is_err());
    }
}
