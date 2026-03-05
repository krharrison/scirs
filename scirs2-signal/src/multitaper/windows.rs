// DPSS (Slepian) window generation for multitaper spectral estimation.
//
// This module delegates to the enhanced DPSS implementation in dpss_enhanced.rs
// which provides a validated, numerically accurate computation.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

#[allow(unused_imports)]
/// Compute Discrete Prolate Spheroidal Sequences (DPSS), also known as Slepian sequences.
///
/// DPSS tapers are often used in multitaper spectral estimation and are designed
/// to maximize energy concentration in a specified frequency band.
///
/// # Arguments
///
/// * `n` - Length of the tapers
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers to compute (should be less than or equal to 2*nw)
/// * `return_ratios` - If true, also return the eigenvalues (concentration ratios)
///
/// # Returns
///
/// * Tuple of (tapers, `Option<eigenvalues>`)
///   - tapers: Array2 of DPSS tapers (shape: [k, n])
///   - eigenvalues: Array1 of concentration ratios if return_ratios is true
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::dpss;
///
/// // Compute 4 DPSS tapers of length 64 with time-bandwidth product of 4
/// let result = dpss(64, 4.0, 4, true).expect("Operation failed");
/// let (tapers, eigenvalues) = (result.0, result.1.expect("operation should succeed"));
///
/// // Check number of tapers
/// assert_eq!(tapers.shape()[0], 4);
/// assert_eq!(tapers.shape()[1], 64);
///
/// // Basic verification - eigenvalues should exist and be positive
/// assert!(eigenvalues.len() >= 2);
/// assert!(eigenvalues[0] > 0.0);
/// ```
#[allow(dead_code)]
pub fn dpss(
    n: usize,
    nw: f64,
    k: usize,
    return_ratios: bool,
) -> SignalResult<(Array2<f64>, Option<Array1<f64>>)> {
    if n < 2 {
        return Err(SignalError::ValueError(
            "Length of tapers must be at least 2".to_string(),
        ));
    }

    if nw <= 0.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product must be positive".to_string(),
        ));
    }

    if k < 1 {
        return Err(SignalError::ValueError(
            "Number of tapers must be at least 1".to_string(),
        ));
    }

    // Maximum number of tapers that can be well-concentrated with the given nw
    let max_tapers = (2.0 * nw).floor() as usize;

    if k > max_tapers {
        return Err(SignalError::ValueError(format!(
            "Number of tapers k ({}) must not exceed 2*nw ({})",
            k, max_tapers
        )));
    }

    // Delegate to the enhanced DPSS implementation which uses the validated
    // Jacobi eigendecomposition (for n<=256) or implicit QR algorithm (for larger n)
    super::dpss_enhanced::dpss_enhanced(n, nw, k, return_ratios)
}
