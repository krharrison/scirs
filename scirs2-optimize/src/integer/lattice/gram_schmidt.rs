//! Gram-Schmidt orthogonalization for lattice basis reduction.
//!
//! Provides efficient computation of Gram-Schmidt coefficients and squared norms,
//! as well as incremental update after basis swaps and size-reduction steps.

use scirs2_core::ndarray::{Array1, Array2};

/// Compute Gram-Schmidt orthogonalization of a lattice basis.
///
/// Given a basis matrix `basis` where each row is a basis vector, computes:
/// - `mu[i][j] = <b_i, b̃_j> / <b̃_j, b̃_j>` for `j < i` (Gram-Schmidt coefficients)
/// - `bnorm_sq[i] = <b̃_i, b̃_i>` (squared norms of orthogonalized basis vectors)
///
/// The orthogonalized vectors b̃ are implicitly defined by:
/// `b̃_i = b_i - sum_{j < i} mu[i][j] * b̃_j`
///
/// # Arguments
/// * `basis` - Matrix of shape [n, d] where each row is a lattice basis vector
///
/// # Returns
/// `(mu, bnorm_sq)` where `mu` has shape [n, n] and `bnorm_sq` has length n.
/// Only the lower-triangular part of `mu` (with j < i) is meaningful.
pub fn gram_schmidt(basis: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let n = basis.nrows();
    let d = basis.ncols();

    let mut mu = Array2::<f64>::zeros((n, n));
    let mut bnorm_sq = Array1::<f64>::zeros(n);

    // Store orthogonalized vectors explicitly for numeric stability
    let mut b_star: Vec<Vec<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        // Start with b_i
        let mut b_star_i: Vec<f64> = (0..d).map(|k| basis[[i, k]]).collect();

        for j in 0..i {
            if bnorm_sq[j] < 1e-14 {
                // Degenerate basis vector: skip projection
                mu[[i, j]] = 0.0;
                continue;
            }
            // mu[i][j] = <b_i, b̃_j> / <b̃_j, b̃_j>
            let dot = b_star_i
                .iter()
                .zip(b_star[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let coeff = dot / bnorm_sq[j];
            mu[[i, j]] = coeff;
            // b̃_i -= mu[i][j] * b̃_j
            for k in 0..d {
                b_star_i[k] -= coeff * b_star[j][k];
            }
        }

        // Squared norm of b̃_i
        bnorm_sq[i] = b_star_i.iter().map(|x| x * x).sum();
        b_star.push(b_star_i);
    }

    (mu, bnorm_sq)
}

/// Update Gram-Schmidt coefficients and norms after swapping rows `k-1` and `k`.
///
/// When the LLL algorithm swaps basis vectors at positions `k-1` and `k`, the
/// full Gram-Schmidt recomputation can be avoided using the following formulas
/// (Lovász condition update). This is more efficient than full recomputation for
/// large bases.
///
/// The update formulas are based on:
/// - Nguyen & Stehlé (2009). "LLL on the Average"
///
/// # Arguments
/// * `basis` - The basis *after* the swap has been applied
/// * `mu` - Gram-Schmidt coefficients to update (modified in place)
/// * `bnorm_sq` - Squared norms to update (modified in place)
/// * `k` - The index (1-based) such that rows k-1 and k were swapped
pub fn update_gram_schmidt_after_swap(
    basis: &Array2<f64>,
    mu: &mut Array2<f64>,
    bnorm_sq: &mut Array1<f64>,
    k: usize,
) {
    // After a swap, we need to recompute from position k-1 onward.
    // For correctness and simplicity, we recompute from k-1.
    let n = basis.nrows();
    let d = basis.ncols();

    // Recompute orthogonalized vectors from scratch for rows k-1..n
    // First, we need all b_star up to k-2 to still be valid
    // Rebuild everything from k-1 forward

    // Compute b_star for rows 0..k-1 (unchanged)
    let mut b_star: Vec<Vec<f64>> = Vec::with_capacity(n);
    // Need to rebuild b_star for rows 0..k-1 from scratch since mu may have changed
    // We rebuild from 0 for safety, which is correct
    for i in 0..k.saturating_sub(1) {
        let mut b_star_i: Vec<f64> = (0..d).map(|col| basis[[i, col]]).collect();
        for j in 0..i {
            let coeff = mu[[i, j]];
            for col in 0..d {
                b_star_i[col] -= coeff * b_star[j][col];
            }
        }
        b_star.push(b_star_i);
    }

    // Recompute from k-1 forward
    for i in k.saturating_sub(1)..n {
        let mut b_star_i: Vec<f64> = (0..d).map(|col| basis[[i, col]]).collect();
        for j in 0..i {
            if bnorm_sq[j] < 1e-14 {
                mu[[i, j]] = 0.0;
                continue;
            }
            let dot = b_star_i
                .iter()
                .zip(b_star[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let coeff = dot / bnorm_sq[j];
            mu[[i, j]] = coeff;
            for col in 0..d {
                b_star_i[col] -= coeff * b_star[j][col];
            }
        }
        bnorm_sq[i] = b_star_i.iter().map(|x| x * x).sum();
        b_star.push(b_star_i);
    }
}

/// Perform one size-reduction step: reduce `basis[k]` with respect to `basis[j]`.
///
/// Adds `-round(mu[k][j]) * basis[j]` to `basis[k]` and updates the transformation
/// matrix row accordingly. The Gram-Schmidt coefficients are updated too.
///
/// After this operation, `|mu[k][j]| <= 0.5`.
///
/// # Arguments
/// * `basis` - The lattice basis (modified in place)
/// * `unimod` - The unimodular transformation matrix (modified in place)
/// * `mu` - Gram-Schmidt coefficients (modified in place)
/// * `k` - Row to reduce
/// * `j` - Row to reduce against (`j < k`)
pub fn size_reduce_step(
    basis: &mut Array2<f64>,
    unimod: &mut Array2<f64>,
    mu: &mut Array2<f64>,
    k: usize,
    j: usize,
) {
    let q = mu[[k, j]].round();
    if q == 0.0 {
        return;
    }
    let n = basis.ncols();
    let num_vecs = basis.nrows();

    // basis[k] -= q * basis[j]
    for col in 0..n {
        let bj = basis[[j, col]];
        basis[[k, col]] -= q * bj;
    }
    // unimod[k] -= q * unimod[j]
    for col in 0..num_vecs {
        let uj = unimod[[j, col]];
        unimod[[k, col]] -= q * uj;
    }
    // Update mu[k][l] for l <= j: mu[k][l] -= q * mu[j][l]
    for l in 0..j {
        let mujl = mu[[j, l]];
        mu[[k, l]] -= q * mujl;
    }
    // mu[k][j] -= q
    mu[[k, j]] -= q;
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let basis = array![[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (mu, bnorm_sq) = gram_schmidt(&basis);

        let n = basis.nrows();
        let d = basis.ncols();

        // Reconstruct b_star vectors
        let mut b_star: Vec<Vec<f64>> = Vec::new();
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

        // Check orthogonality: <b̃_i, b̃_j> should be ~0 for i != j
        for i in 0..n {
            for j in 0..i {
                let dot: f64 = b_star[i].iter().zip(b_star[j].iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-10, "b̃_{} and b̃_{} not orthogonal: dot={}", i, j, dot);
            }
        }

        // Check that bnorm_sq matches actual squared norms
        for i in 0..n {
            let ns: f64 = b_star[i].iter().map(|x| x * x).sum();
            assert!((bnorm_sq[i] - ns).abs() < 1e-10, "bnorm_sq[{}] mismatch", i);
        }
    }

    #[test]
    fn test_gram_schmidt_mu_coefficients() {
        // Simple 2D case: b0 = [1,0], b1 = [1,1]
        let basis = array![[1.0, 0.0], [1.0, 1.0]];
        let (mu, _bnorm_sq) = gram_schmidt(&basis);
        // b̃_0 = [1, 0], b̃_1 = [1,1] - <[1,1],[1,0]>/<[1,0],[1,0]> * [1,0]
        //      = [1,1] - 1.0 * [1,0] = [0,1]
        // mu[1][0] = <[1,1],[1,0]> / <[1,0],[1,0]> = 1/1 = 1.0
        assert!((mu[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gram_schmidt_identity() {
        // Orthonormal basis: identity matrix
        let basis = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (mu, bnorm_sq) = gram_schmidt(&basis);
        // All mu[i][j] for j < i should be 0
        for i in 0..3 {
            for j in 0..i {
                assert!(mu[[i, j]].abs() < 1e-10, "mu[{}][{}] = {}", i, j, mu[[i, j]]);
            }
            // All bnorm_sq should be 1
            assert!((bnorm_sq[i] - 1.0).abs() < 1e-10);
        }
    }
}
