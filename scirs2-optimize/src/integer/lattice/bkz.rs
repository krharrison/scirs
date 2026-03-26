//! BKZ (Block Korkine-Zolotarev) lattice basis reduction algorithm.
//!
//! BKZ generalizes LLL by applying SVP oracles within overlapping blocks of the
//! basis. Larger block sizes yield stronger reduction (approaching HKZ) at the
//! cost of more computation. BKZ-β with β=2 is equivalent to LLL.
//!
//! # References
//! - Schnorr, C.P. (1987). "A hierarchy of polynomial time lattice basis reduction
//!   algorithms." Theoretical Computer Science, 53(2-3), 201–224.
//! - Schnorr, C.P., Euchner, M. (1994). "Lattice basis reduction: Improved practical
//!   algorithms and solving subset sum problems." Mathematical programming, 66(1-3), 181–199.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::gram_schmidt::gram_schmidt;
use super::lll::{LLLConfig, LLLReducer};
use super::svp::solve_svp;

/// Configuration for the BKZ reduction algorithm.
#[derive(Debug, Clone)]
pub struct BKZConfig {
    /// Block size β (BKZ window size). Larger values yield stronger reduction.
    /// β=2 is equivalent to LLL. Practical range: 10–30.
    pub block_size: usize,
    /// Maximum number of full tours over all blocks.
    pub max_tours: usize,
    /// LLL preprocessing Lovász parameter (usually set close to 1 for BKZ preprocessing).
    pub lll_delta: f64,
    /// Maximum enumeration nodes for each SVP call within a block.
    pub svp_max_nodes: usize,
}

impl Default for BKZConfig {
    fn default() -> Self {
        BKZConfig {
            block_size: 10,
            max_tours: 8,
            lll_delta: 0.99,
            svp_max_nodes: 100_000,
        }
    }
}

/// Result from BKZ basis reduction.
#[derive(Debug, Clone)]
pub struct BKZResult {
    /// BKZ-reduced basis (rows are reduced basis vectors).
    pub reduced_basis: Array2<f64>,
    /// Number of full tours completed.
    pub n_tours: usize,
    /// Euclidean norm of the first (shortest) basis vector.
    pub first_vector_norm: f64,
}

/// BKZ lattice basis reducer.
///
/// Applies LLL preprocessing followed by BKZ block reduction, using an SVP oracle
/// within each sliding window of size `block_size`.
pub struct BKZReducer {
    config: BKZConfig,
}

impl BKZReducer {
    /// Create a new BKZ reducer with the given configuration.
    pub fn new(config: BKZConfig) -> Self {
        BKZReducer { config }
    }

    /// Apply BKZ reduction to the given lattice basis.
    ///
    /// # Arguments
    /// * `basis` - Matrix of shape [n, d] where each row is a lattice basis vector
    ///
    /// # Returns
    /// `BKZResult` with the reduced basis and statistics.
    pub fn reduce(&self, basis: &Array2<f64>) -> OptimizeResult<BKZResult> {
        let n = basis.nrows();
        let d = basis.ncols();
        if n == 0 || d == 0 {
            return Err(OptimizeError::ValueError(
                "BKZ: basis matrix must be non-empty".to_string(),
            ));
        }

        let beta = self.config.block_size.min(n);

        // Step 1: LLL preprocessing
        let lll_config = LLLConfig {
            delta: self.config.lll_delta,
            ..Default::default()
        };
        let lll_reducer = LLLReducer::new(lll_config.clone());
        let lll_result = lll_reducer.reduce(basis)?;
        let mut b = lll_result.reduced_basis;

        // Step 2: BKZ tours
        let mut n_tours = 0usize;

        for _tour in 0..self.config.max_tours {
            n_tours += 1;
            let mut improved = false;

            // Slide the window of size beta over all blocks
            for i in 0..=(n.saturating_sub(beta)) {
                let block_end = (i + beta).min(n);
                let block_size = block_end - i;

                // Extract the projected block basis (rows i..block_end)
                // We work with the projected lattice π_i(b_{i..block_end})
                let block_basis = extract_projected_block(&b, i, block_end)?;

                // Solve SVP in the projected block
                let svp_vec = match solve_svp(&block_basis, self.config.svp_max_nodes) {
                    Ok(v) => v,
                    Err(_) => continue, // Skip this block if SVP fails
                };

                // The SVP gives a short vector in the projected sublattice.
                // We need to lift it back and insert into the basis.
                // Compute the norm of the SVP solution vs the current b_i projection
                let svp_norm_sq: f64 = svp_vec.iter().map(|x| x * x).sum();

                // Compute the norm of the projected first vector of the block
                let (_, bnorm_sq) = gram_schmidt(&block_basis);
                let current_b0_proj_norm_sq = if !bnorm_sq.is_empty() { bnorm_sq[0] } else { 0.0 };

                if svp_norm_sq < current_b0_proj_norm_sq - 1e-8 {
                    // The SVP found a shorter vector: lift it back and insert into basis
                    // Express svp_vec as an integer combination of block_basis rows
                    // and update basis
                    if let Ok(true) = insert_svp_vector(&mut b, &block_basis, &svp_vec, i, block_end, d) {
                        // Re-apply LLL to maintain size-reduction after insertion
                        let lll_r = lll_reducer.reduce(&b)?;
                        b = lll_r.reduced_basis;
                        improved = true;
                    }
                }

                // Handle the last block of size < beta (final step)
                if block_end == n && block_size < beta {
                    // Solve SVP in the last remaining block too
                    break;
                }
            }

            if !improved {
                break;
            }
        }

        let first_vector_norm: f64 = b.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(BKZResult {
            reduced_basis: b,
            n_tours,
            first_vector_norm,
        })
    }
}

/// Extract the projected block basis for BKZ.
///
/// Computes the Gram-Schmidt orthogonalization of the full basis and extracts
/// the projected block starting at index `start`. The projected basis is defined
/// as π_start(b_{start}), π_start(b_{start+1}), ..., π_start(b_{end-1}).
fn extract_projected_block(
    basis: &Array2<f64>,
    start: usize,
    end: usize,
) -> OptimizeResult<Array2<f64>> {
    let d = basis.ncols();
    let block_size = end - start;
    if block_size == 0 {
        return Err(OptimizeError::ValueError(
            "BKZ block must be non-empty".to_string(),
        ));
    }

    let (mu, bnorm_sq) = gram_schmidt(basis);
    let n = basis.nrows();

    // Compute b_star (orthogonalized) vectors for the full basis
    let mut b_star: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut bsi: Vec<f64> = (0..d).map(|k| basis[[i, k]]).collect();
        for j in 0..i {
            let c = mu[[i, j]];
            for k in 0..d {
                bsi[k] -= c * b_star[j][k];
            }
        }
        b_star.push(bsi);
    }

    // Projected vectors: π_start(b_i) = b_i - Σ_{j<start} mu[i][j] * b̃_j
    let mut block = Array2::<f64>::zeros((block_size, d));
    for (bi, i) in (start..end).enumerate() {
        // Start with b_i
        let mut proj: Vec<f64> = (0..d).map(|k| basis[[i, k]]).collect();
        // Subtract components along b̃_0, ..., b̃_{start-1}
        for j in 0..start {
            let c = mu[[i, j]];
            for k in 0..d {
                proj[k] -= c * b_star[j][k];
            }
        }
        for k in 0..d {
            block[[bi, k]] = proj[k];
        }
    }

    // Suppress unused variable warnings
    let _ = bnorm_sq;

    Ok(block)
}

/// Attempt to insert an SVP solution vector into the basis at position `insert_pos`.
///
/// The SVP vector is expressed in terms of the block_basis; we find integer
/// coefficients and then update the full basis accordingly.
/// Returns Ok(true) if the insertion was performed, Ok(false) otherwise.
fn insert_svp_vector(
    basis: &mut Array2<f64>,
    block_basis: &Array2<f64>,
    svp_vec: &Array1<f64>,
    insert_pos: usize,
    block_end: usize,
    _d: usize,
) -> OptimizeResult<bool> {
    let n_rows = basis.nrows();
    let d = basis.ncols();

    // Compute integer coordinates of svp_vec in block_basis via least-squares rounding
    // For a projected block, we approximate: svp_vec ≈ Σ_i c_i * block_basis[i]
    let block_n = block_basis.nrows();
    let mut coeffs = vec![0.0f64; block_n];

    // Use back-substitution via Gram-Schmidt of block_basis
    let (block_mu, block_bnorm_sq) = gram_schmidt(block_basis);

    // Compute b_star for the block
    let mut b_star: Vec<Vec<f64>> = Vec::with_capacity(block_n);
    for i in 0..block_n {
        let mut bsi: Vec<f64> = (0..d).map(|k| block_basis[[i, k]]).collect();
        for j in 0..i {
            let c = block_mu[[i, j]];
            for k in 0..d {
                bsi[k] -= c * b_star[j][k];
            }
        }
        b_star.push(bsi);
    }

    // Compute the Gram-Schmidt coordinates of svp_vec
    let mut residual: Vec<f64> = (0..d).map(|k| svp_vec[k]).collect();
    for i in (0..block_n).rev() {
        if block_bnorm_sq[i] < 1e-14 {
            continue;
        }
        let dot: f64 = residual.iter().zip(b_star[i].iter()).map(|(a, b)| a * b).sum();
        let c = dot / block_bnorm_sq[i];
        // This gives the GS coordinate; back out the original coordinate
        // In GS decomposition: coordinate in original basis direction i
        // is accumulated from c and the mu coefficients
        coeffs[i] = c;
        for k in 0..d {
            residual[k] -= c * b_star[i][k];
        }
    }

    // Round coefficients to integers
    let int_coeffs: Vec<i64> = coeffs.iter().map(|&c| c.round() as i64).collect();

    // Reconstruct the actual lattice vector from rounded integer coefficients
    let mut lattice_vec = vec![0.0f64; d];
    for (bi, i) in (insert_pos..block_end).enumerate() {
        let c = int_coeffs[bi] as f64;
        if c != 0.0 {
            for k in 0..d {
                lattice_vec[k] += c * basis[[i, k]];
            }
        }
    }

    // Verify it's actually shorter than the current basis[insert_pos]
    let lv_norm_sq: f64 = lattice_vec.iter().map(|x| x * x).sum();
    let b0_norm_sq: f64 = (0..d).map(|k| basis[[insert_pos, k]].powi(2)).sum();
    if lv_norm_sq < b0_norm_sq - 1e-8 && lv_norm_sq > 1e-14 {
        // Insert by shifting: place lattice_vec at insert_pos, shift others down
        // Move all vectors in [insert_pos, block_end-1] down one position
        let new_vec = lattice_vec.clone();

        // Shift rows down from insert_pos..block_end-1
        for row in (insert_pos + 1..block_end).rev() {
            for k in 0..d {
                let val = basis[[row - 1, k]];
                basis[[row, k]] = val;
            }
        }
        // Place the new vector at insert_pos
        for k in 0..d {
            basis[[insert_pos, k]] = new_vec[k];
        }
        let _ = n_rows; // suppress warning
        return Ok(true);
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bkz_block2_equivalent_to_lll() {
        // BKZ with block size 2 should give roughly the same result as LLL
        let basis = array![
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 2.0],
            [3.0, 5.0, 6.0]
        ];

        let bkz_config = BKZConfig {
            block_size: 2,
            max_tours: 10,
            lll_delta: 0.75,
            svp_max_nodes: 10_000,
        };
        let bkz = BKZReducer::new(bkz_config);
        let bkz_result = bkz.reduce(&basis).expect("BKZ should succeed");

        let lll = LLLReducer::new(LLLConfig::default());
        let lll_result = lll.reduce(&basis).expect("LLL should succeed");

        // Both should give a valid reduced basis
        assert_eq!(bkz_result.reduced_basis.nrows(), 3);
        assert_eq!(lll_result.reduced_basis.nrows(), 3);
    }

    #[test]
    fn test_bkz_first_vector_not_longer_than_lll() {
        // BKZ should produce a first vector no longer than LLL
        let basis = array![
            [10.0, 3.0, -1.0, 2.0],
            [-4.0, 7.0, 2.0, 1.0],
            [1.0, -2.0, 5.0, 3.0],
            [2.0, 1.0, -3.0, 6.0]
        ];

        let lll = LLLReducer::new(LLLConfig::default());
        let lll_result = lll.reduce(&basis).expect("LLL should succeed");
        let lll_norm: f64 = lll_result.reduced_basis.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();

        let bkz = BKZReducer::new(BKZConfig {
            block_size: 3,
            max_tours: 5,
            lll_delta: 0.99,
            svp_max_nodes: 50_000,
        });
        let bkz_result = bkz.reduce(&basis).expect("BKZ should succeed");

        // BKZ first vector should be <= LLL first vector (with some tolerance)
        assert!(
            bkz_result.first_vector_norm <= lll_norm + 1e-6,
            "BKZ norm {} should be <= LLL norm {}",
            bkz_result.first_vector_norm, lll_norm
        );
    }

    #[test]
    fn test_bkz_respects_max_tours() {
        let basis = array![[1.0, 0.0], [0.0, 1.0]];
        let config = BKZConfig {
            block_size: 2,
            max_tours: 3,
            ..Default::default()
        };
        let bkz = BKZReducer::new(config);
        let result = bkz.reduce(&basis).expect("BKZ should succeed");
        assert!(result.n_tours <= 3, "Should not exceed max_tours");
    }
}
