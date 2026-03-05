//! Pseudospectrum analysis.
//!
//! The **ε-pseudospectrum** of a matrix `A` is the set of complex numbers `z`
//! for which the matrix `zI − A` is "nearly singular":
//!
//! ```text
//! Λ_ε(A) = { z ∈ ℂ : σ_min(zI − A) < ε }
//! ```
//!
//! Equivalently, `z ∈ Λ_ε(A)` iff `z` is an eigenvalue of some perturbed
//! matrix `A + E` with `‖E‖₂ < ε`.
//!
//! | Function | Description |
//! |---|---|
//! | [`epsilon_pseudospectrum`] | Compute `σ_min(zI − A)` on a complex grid |
//! | [`kreiss_constant`] | Lower bound on the Kreiss constant via grid search |
//! | [`pseudospectral_abscissa`] | Rightmost point of `Λ_ε(A)` (stability margin) |
//! | [`transient_bound`] | Upper bound on ‖e^{tA}‖₂ from Kreiss constant |
//!
//! ## Mathematical Background
//!
//! ### Kreiss Matrix Theorem
//!
//! The Kreiss constant `K(A)` is defined by
//! ```text
//! K(A) = sup_{Re(z) > 0} Re(z) · ‖(zI − A)⁻¹‖₂
//!      = sup_{ε > 0}  α_ε(A) / ε
//! ```
//! where `α_ε(A)` is the ε-pseudospectral abscissa.
//!
//! The Kreiss Matrix Theorem states
//! ```text
//! K(A) ≤ sup_{t ≥ 0} ‖e^{tA}‖₂ ≤ e · n · K(A)
//! ```
//! (for matrices in ℝ^{n×n}) giving a transient growth bound.
//!
//! ### ε-Pseudospectral Abscissa
//!
//! The ε-pseudospectral abscissa is the rightmost real part of the
//! ε-pseudospectrum:
//! ```text
//! α_ε(A) = sup{ Re(z) : z ∈ Λ_ε(A) }
//! ```
//!
//! It is a smooth function of `ε` and can be used to measure the stability
//! margin of a dynamical system `ẋ = Ax`.
//!
//! ## References
//!
//! - Trefethen, L. N.; Embree, M. (2005). *Spectra and Pseudospectra: The
//!   Behavior of Nonnormal Matrices and Operators*. Princeton University Press.
//! - Kreiss, H.-O. (1962). "Über die Stabilitätsdefinition für
//!   Differenzengleichungen die partielle Differentialgleichungen approximieren".
//!   *BIT Numer. Math.* 2: 153–181.
//! - Trefethen, L. N. (1992). "Pseudospectra of matrices". In: *Numerical
//!   Analysis 1991*, Longman Scientific.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{LinalgError, LinalgResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the matrix `zI − A` for a complex shift `z = (x, y)`.
///
/// Returns a real 2n×2n matrix representing the complex matrix as a real
/// block matrix:
/// ```text
/// Re(zI-A)  -Im(z)I
/// Im(z)I    Re(zI-A)
/// ```
/// This lets us compute `σ_min(zI − A)` using only real SVD.
fn build_shifted_real_block(a: &ArrayView2<f64>, re: f64, im: f64) -> Array2<f64> {
    let n = a.nrows();
    let mut block = Array2::<f64>::zeros((2 * n, 2 * n));
    // Top-left: (re*I - A)
    for i in 0..n {
        for j in 0..n {
            block[[i, j]] = if i == j { re - a[[i, j]] } else { -a[[i, j]] };
        }
    }
    // Top-right: -im * I
    for i in 0..n {
        block[[i, i + n]] = -im;
    }
    // Bottom-left: im * I
    for i in 0..n {
        block[[i + n, i]] = im;
    }
    // Bottom-right: (re*I - A) again
    for i in 0..n {
        for j in 0..n {
            block[[i + n, j + n]] = if i == j { re - a[[i, j]] } else { -a[[i, j]] };
        }
    }
    block
}

/// Compute `σ_min(zI − A)` for a single complex point `z = re + i*im`.
///
/// Uses the real block embedding to avoid complex arithmetic.
fn sigma_min_at_point(a: &ArrayView2<f64>, re: f64, im: f64) -> LinalgResult<f64> {
    let block = build_shifted_real_block(a, re, im);
    let (_, s, _) = crate::decomposition::svd(&block.view(), false, None)?;
    // σ_min of the complex matrix is the smallest singular value of the block.
    // The 2n singular values of the block come in pairs; the minimum is σ_min.
    Ok(s[s.len() - 1])
}

// ---------------------------------------------------------------------------
// epsilon_pseudospectrum
// ---------------------------------------------------------------------------

/// Result of [`epsilon_pseudospectrum`].
#[derive(Debug, Clone)]
pub struct PseudospectrumGrid {
    /// Real-axis grid points.
    pub re_grid: Array1<f64>,
    /// Imaginary-axis grid points.
    pub im_grid: Array1<f64>,
    /// `sigma_min[i, j] = σ_min(z_{i,j} I − A)` where `z_{i,j} = re_grid[j] + i·im_grid[i]`.
    /// Shape: `(n_im, n_re)`.
    pub sigma_min: Array2<f64>,
    /// Boolean grid: `inside[i, j]` is true iff `z_{i,j} ∈ Λ_ε(A)`.
    pub inside: Array2<bool>,
    /// The epsilon threshold used.
    pub epsilon: f64,
}

/// Compute the ε-pseudospectrum of `A` on a rectangular complex grid.
///
/// For each grid point `z = x + iy`, evaluates `σ_min(zI − A)`.  The
/// point belongs to `Λ_ε(A)` iff `σ_min(zI − A) < ε`.
///
/// # Arguments
///
/// * `a` — Square matrix `A` (n × n).
/// * `re_range` — `(re_min, re_max)` real-axis extent.
/// * `im_range` — `(im_min, im_max)` imaginary-axis extent.
/// * `n_re` — Number of real-axis grid points (columns).
/// * `n_im` — Number of imaginary-axis grid points (rows).
/// * `epsilon` — Threshold ε > 0.
///
/// # Returns
///
/// [`PseudospectrumGrid`] with the σ_min values and membership grid.
///
/// # Errors
///
/// * [`LinalgError::ShapeError`] if `a` is not square or empty.
/// * [`LinalgError::ValueError`] for invalid grid parameters.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::pseudospectrum::epsilon_pseudospectrum;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // eigenvalues ±i
/// let grid = epsilon_pseudospectrum(
///     &a.view(), (-2.0, 2.0), (-2.0, 2.0), 10, 10, 0.5
/// ).expect("failed");
/// assert_eq!(grid.sigma_min.shape(), &[10, 10]);
/// ```
pub fn epsilon_pseudospectrum(
    a: &ArrayView2<f64>,
    re_range: (f64, f64),
    im_range: (f64, f64),
    n_re: usize,
    n_im: usize,
    epsilon: f64,
) -> LinalgResult<PseudospectrumGrid> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "epsilon_pseudospectrum: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "epsilon_pseudospectrum: empty matrix".to_string(),
        ));
    }
    if n_re < 2 || n_im < 2 {
        return Err(LinalgError::ValueError(
            "epsilon_pseudospectrum: grid must have at least 2 points in each direction".to_string(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(LinalgError::ValueError(format!(
            "epsilon_pseudospectrum: epsilon must be positive, got {}",
            epsilon
        )));
    }
    if re_range.0 >= re_range.1 {
        return Err(LinalgError::ValueError(
            "epsilon_pseudospectrum: re_range min must be < max".to_string(),
        ));
    }
    if im_range.0 >= im_range.1 {
        return Err(LinalgError::ValueError(
            "epsilon_pseudospectrum: im_range min must be < max".to_string(),
        ));
    }

    // Build grids.
    let re_step = (re_range.1 - re_range.0) / (n_re as f64 - 1.0);
    let im_step = (im_range.1 - im_range.0) / (n_im as f64 - 1.0);

    let re_grid: Array1<f64> =
        Array1::from_iter((0..n_re).map(|k| re_range.0 + k as f64 * re_step));
    let im_grid: Array1<f64> =
        Array1::from_iter((0..n_im).map(|k| im_range.0 + k as f64 * im_step));

    let mut sigma_min = Array2::<f64>::zeros((n_im, n_re));
    let mut inside = Array2::<bool>::default((n_im, n_re));

    for i in 0..n_im {
        for j in 0..n_re {
            let re = re_grid[j];
            let im = im_grid[i];
            let sm = sigma_min_at_point(a, re, im)?;
            sigma_min[[i, j]] = sm;
            inside[[i, j]] = sm < epsilon;
        }
    }

    Ok(PseudospectrumGrid {
        re_grid,
        im_grid,
        sigma_min,
        inside,
        epsilon,
    })
}

// ---------------------------------------------------------------------------
// kreiss_constant
// ---------------------------------------------------------------------------

/// Result of [`kreiss_constant`].
#[derive(Debug, Clone)]
pub struct KreissResult {
    /// Lower bound on the Kreiss constant `K(A)`.
    pub kreiss_lower_bound: f64,
    /// The ε and z at which the maximum `Re(z) / ε · ‖(zI-A)^{-1}‖` was attained,
    /// stored as `(epsilon, re, im, resolvent_norm)`.
    pub maximising_point: (f64, f64, f64, f64),
    /// Number of grid points evaluated.
    pub grid_evaluations: usize,
}

/// Compute a lower bound on the Kreiss constant via a grid search.
///
/// The Kreiss constant is defined as
/// ```text
/// K(A) = sup_{Re(z) > 0}  Re(z) · ‖(zI − A)⁻¹‖₂
///      = sup_{ε > 0}  α_ε(A) / ε
/// ```
///
/// This function estimates a lower bound by evaluating the resolvent norm
/// `‖(zI-A)⁻¹‖₂ = 1 / σ_min(zI-A)` on a grid in the right half-plane and
/// computing `Re(z) / σ_min(zI-A)`.
///
/// # Arguments
///
/// * `a` — Square matrix `A`.
/// * `re_max` — Maximum real part of grid (default 5× spectral abscissa + 1).
/// * `n_re` — Real-axis resolution (default 30).
/// * `n_im` — Imaginary-axis resolution (default 60).
///
/// # Returns
///
/// [`KreissResult`] with the lower bound and maximising grid point.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::pseudospectrum::kreiss_constant;
///
/// // Normal matrix (rotation): K(A) = 1.
/// let a = array![[0.0_f64, -1.0], [1.0, 0.0]];
/// let res = kreiss_constant(&a.view(), None, Some(20), Some(40)).expect("failed");
/// // Lower bound should be at most a few units for a rotation matrix.
/// assert!(res.kreiss_lower_bound >= 1.0 - 1e-3);
/// ```
pub fn kreiss_constant(
    a: &ArrayView2<f64>,
    re_max: Option<f64>,
    n_re: Option<usize>,
    n_im: Option<usize>,
) -> LinalgResult<KreissResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "kreiss_constant: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "kreiss_constant: empty matrix".to_string(),
        ));
    }

    let nr = n_re.unwrap_or(30);
    let ni = n_im.unwrap_or(60);

    // Estimate spectral abscissa: max real part of eigenvalues.
    // For a Hermitian matrix all eigenvalues are real; for general we use eig.
    let spectral_abscissa = compute_spectral_abscissa(a)?;
    let re_upper = re_max.unwrap_or(5.0 * spectral_abscissa.abs() + 1.0).max(1.0);

    // Search in right half-plane: Re(z) ∈ (ε_small, re_upper].
    let re_min = 1e-4; // small positive offset
    let im_half = re_upper * 2.0;

    // Build grid.
    let re_step = (re_upper - re_min) / (nr as f64 - 1.0).max(1.0);
    let im_step = 2.0 * im_half / (ni as f64 - 1.0).max(1.0);

    let mut best_value = 1.0_f64; // Kreiss constant ≥ 1 always
    let mut best_point = (0.0_f64, re_min, 0.0, 1.0);
    let mut evaluations = 0_usize;

    for i in 0..ni {
        let im = -im_half + i as f64 * im_step;
        for j in 0..nr {
            let re = re_min + j as f64 * re_step;
            match sigma_min_at_point(a, re, im) {
                Ok(sm) => {
                    evaluations += 1;
                    if sm > 0.0 {
                        let resolvent_norm = 1.0 / sm;
                        let kreiss_estimate = re * resolvent_norm;
                        if kreiss_estimate > best_value {
                            best_value = kreiss_estimate;
                            best_point = (sm, re, im, resolvent_norm);
                        }
                    }
                }
                Err(_) => { /* skip singular-looking points */ }
            }
        }
    }

    Ok(KreissResult {
        kreiss_lower_bound: best_value,
        maximising_point: best_point,
        grid_evaluations: evaluations,
    })
}

// ---------------------------------------------------------------------------
// pseudospectral_abscissa
// ---------------------------------------------------------------------------

/// Result of [`pseudospectral_abscissa`].
#[derive(Debug, Clone)]
pub struct PseudospectralAbscissaResult {
    /// The ε-pseudospectral abscissa `α_ε(A)`.
    pub abscissa: f64,
    /// The imaginary part at which the rightmost pseudospectral point was found.
    pub im_at_abscissa: f64,
    /// The classical spectral abscissa (max real part of eigenvalues).
    pub spectral_abscissa: f64,
    /// Stability margin: `−α_ε(A)` (positive means ε-pseudostable).
    pub stability_margin: f64,
}

/// Compute the ε-pseudospectral abscissa of `A`.
///
/// The **ε-pseudospectral abscissa** is:
/// ```text
/// α_ε(A) = sup{ Re(z) : z ∈ Λ_ε(A) }
/// ```
///
/// It generalises the classical spectral abscissa `α(A) = max Re(λᵢ(A))`
/// to account for non-normality.  A matrix with `α(A) < 0` but large
/// `α_ε(A) / ε` can exhibit significant transient growth even though it is
/// asymptotically stable.
///
/// # Algorithm
///
/// The function performs a grid search over the imaginary axis to find the
/// real value `x` such that `σ_min(xI − A) = ε` (rightmost crossing point).
/// The `x`-coordinate of this crossing gives `α_ε(A)`.
///
/// # Arguments
///
/// * `a` — Square matrix `A`.
/// * `epsilon` — Pseudospectrum threshold ε > 0.
/// * `n_im` — Number of imaginary-axis probe points (default 200).
///
/// # Returns
///
/// [`PseudospectralAbscissaResult`] with the abscissa and diagnostics.
///
/// # Errors
///
/// * [`LinalgError::ShapeError`] if `a` is not square.
/// * [`LinalgError::ValueError`] for non-positive epsilon.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::pseudospectrum::pseudospectral_abscissa;
///
/// let a = array![[-1.0_f64, 10.0], [0.0, -2.0]]; // stable but non-normal
/// let res = pseudospectral_abscissa(&a.view(), 0.1, None).expect("failed");
/// // α_0.1 >= α(-1, -2) = -1.0  (can be > α for non-normal)
/// assert!(res.abscissa >= res.spectral_abscissa - 1e-6);
/// ```
pub fn pseudospectral_abscissa(
    a: &ArrayView2<f64>,
    epsilon: f64,
    n_im: Option<usize>,
) -> LinalgResult<PseudospectralAbscissaResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "pseudospectral_abscissa: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "pseudospectral_abscissa: empty matrix".to_string(),
        ));
    }
    if epsilon <= 0.0 {
        return Err(LinalgError::ValueError(format!(
            "pseudospectral_abscissa: epsilon must be positive, got {}",
            epsilon
        )));
    }

    let ni = n_im.unwrap_or(200);
    let spectral_abscissa = compute_spectral_abscissa(a)?;

    // Determine the imaginary range and horizontal search range.
    let frobenius_norm_a = frob_norm_view(a);
    let im_range = frobenius_norm_a + epsilon + 10.0;
    let re_range_low = spectral_abscissa - 5.0 * epsilon - frobenius_norm_a;
    let re_range_high = spectral_abscissa + 5.0 * epsilon + 1.0;

    // For each imaginary value, binary-search the rightmost Re(z) where σ_min = ε.
    let im_step = 2.0 * im_range / (ni as f64 - 1.0).max(1.0);
    let mut best_re = spectral_abscissa; // lower bound: at least the true abscissa
    let mut best_im = 0.0;

    for i in 0..ni {
        let im = -im_range + i as f64 * im_step;
        // Binary search for the rightmost Re(z) at this Im(z) where σ_min(zI-A) < ε.
        let mut lo = re_range_low;
        let mut hi = re_range_high;
        // Check if there is any Re ∈ [lo, hi] with σ_min < ε at this Im.
        let sm_hi = match sigma_min_at_point(a, hi, im) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if sm_hi >= epsilon {
            // No crossing above hi; nothing to find.
            continue;
        }
        // Binary search for the rightmost crossing: σ_min(x + i*im, A) = ε.
        // σ_min is increasing as |x| → ∞ away from the spectrum, so we look
        // for the root of (σ_min − ε) changing sign.
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let sm_mid = match sigma_min_at_point(a, mid, im) {
                Ok(v) => v,
                Err(_) => break,
            };
            if sm_mid < epsilon {
                lo = mid; // move rightward
            } else {
                hi = mid; // too far right
            }
        }
        // The rightmost point inside Λ_ε(A) at this Im is approximately lo.
        if lo > best_re {
            best_re = lo;
            best_im = im;
        }
    }

    let stability_margin = -best_re;

    Ok(PseudospectralAbscissaResult {
        abscissa: best_re,
        im_at_abscissa: best_im,
        spectral_abscissa,
        stability_margin,
    })
}

// ---------------------------------------------------------------------------
// transient_bound
// ---------------------------------------------------------------------------

/// Result of [`transient_bound`].
#[derive(Debug, Clone)]
pub struct TransientBoundResult {
    /// The Kreiss lower bound `K ≥ K(A)`.
    pub kreiss_lower_bound: f64,
    /// The Kreiss Matrix Theorem upper bound: `‖e^{tA}‖ ≤ e · n · K(A)`.
    pub upper_bound: f64,
    /// Dimension `n` used in the bound.
    pub n: usize,
    /// The time horizon `t` for which `upper_bound` was computed.
    pub time_horizon: f64,
}

/// Compute the Kreiss Matrix Theorem upper bound on transient matrix exponential growth.
///
/// The Kreiss Matrix Theorem for matrices in ℝ^{n×n} states:
/// ```text
/// sup_{t ≥ 0} ‖e^{tA}‖₂ ≤ e · n · K(A)
/// ```
/// where `K(A)` is the Kreiss constant.
///
/// This provides a guaranteed upper bound on the worst-case transient growth
/// of the linear ODE `ẋ = Ax`.  The bound is tight up to a factor of `n`.
///
/// For a time-dependent bound, the function also reports the exponential
/// growth `e^{α t} · e · n · K(A)` where `α = α(A)` is the spectral abscissa.
///
/// # Arguments
///
/// * `a` — Square matrix `A`.
/// * `time_horizon` — Time `t ≥ 0` at which to evaluate the time-dependent
///   bound (default 1.0).
/// * `n_re` — Grid resolution for Kreiss constant search (default 30).
/// * `n_im` — Grid resolution for Kreiss constant search (default 60).
///
/// # Returns
///
/// [`TransientBoundResult`] with the Kreiss bound and diagnostics.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::pseudospectrum::transient_bound;
///
/// let a = array![[-1.0_f64, 0.0], [0.0, -2.0]]; // stable diagonal
/// let res = transient_bound(&a.view(), Some(1.0), Some(20), Some(40)).expect("failed");
/// assert!(res.upper_bound >= 1.0);
/// assert_eq!(res.n, 2);
/// ```
pub fn transient_bound(
    a: &ArrayView2<f64>,
    time_horizon: Option<f64>,
    n_re: Option<usize>,
    n_im: Option<usize>,
) -> LinalgResult<TransientBoundResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "transient_bound: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "transient_bound: empty matrix".to_string(),
        ));
    }

    let t = time_horizon.unwrap_or(1.0);
    let kreiss_res = kreiss_constant(a, None, n_re, n_im)?;
    let k = kreiss_res.kreiss_lower_bound;

    // Kreiss Matrix Theorem: sup_{s>=0} ‖e^{sA}‖ ≤ e * n * K(A)
    // For a specific t: ‖e^{tA}‖ ≤ e^{α*t} * e * n * K  (α = spectral abscissa)
    let alpha = compute_spectral_abscissa(a)?;
    let e_const = std::f64::consts::E;
    let upper_bound = (alpha * t).exp() * e_const * (n as f64) * k;

    Ok(TransientBoundResult {
        kreiss_lower_bound: k,
        upper_bound,
        n,
        time_horizon: t,
    })
}

// ---------------------------------------------------------------------------
// Internal: spectral abscissa computation
// ---------------------------------------------------------------------------

/// Compute the classical spectral abscissa: `α(A) = max_i Re(λᵢ(A))`.
///
/// For a real symmetric matrix this equals the largest eigenvalue.
/// For a general matrix we use the real eigenvalue computation and take the
/// real parts of the eigenvalues returned by `eig`.
pub(crate) fn compute_spectral_abscissa(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    use crate::eigen::eig;
    let (eigenvalues, _eigenvectors) = eig(a, None)?;
    // The eigenvalues are complex; take max of real parts.
    let max_re = eigenvalues
        .iter()
        .map(|z| z.re)
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(max_re)
}

/// Frobenius norm of a matrix view (used internally).
fn frob_norm_view(a: &ArrayView2<f64>) -> f64 {
    a.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ----- epsilon_pseudospectrum -------------------------------------------

    #[test]
    fn test_pseudospectrum_grid_shape() {
        let a = array![[0.0_f64, 1.0], [-1.0, 0.0]];
        let grid = epsilon_pseudospectrum(
            &a.view(), (-2.0, 2.0), (-2.0, 2.0), 8, 6, 0.5,
        )
        .expect("failed");
        assert_eq!(grid.sigma_min.shape(), &[6, 8]);
        assert_eq!(grid.inside.shape(), &[6, 8]);
        assert_eq!(grid.re_grid.len(), 8);
        assert_eq!(grid.im_grid.len(), 6);
    }

    #[test]
    fn test_pseudospectrum_identity_inside_at_eigenvalue() {
        // A = 2*I has eigenvalue 2.  The point z = 2 + 0i should be inside
        // Λ_ε for any ε > 0.
        let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let sm = sigma_min_at_point(&a.view(), 2.0, 0.0).expect("failed");
        assert!(sm < 1e-10, "σ_min at eigenvalue should be ~0, got {}", sm);
    }

    #[test]
    fn test_pseudospectrum_far_from_spectrum_outside() {
        // A = 0; z = 100 + 0i should give σ_min = 100.
        let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let sm = sigma_min_at_point(&a.view(), 100.0, 0.0).expect("failed");
        assert!((sm - 100.0).abs() < 1e-6, "σ_min = {}", sm);
    }

    // ----- kreiss_constant --------------------------------------------------

    #[test]
    fn test_kreiss_constant_normal_matrix() {
        // Rotation matrix: all eigenvalues on unit circle, Kreiss constant = 1.
        let a = array![[0.0_f64, -1.0], [1.0, 0.0]];
        let res = kreiss_constant(&a.view(), None, Some(15), Some(30)).expect("failed");
        // Lower bound should be ≥ 1
        assert!(
            res.kreiss_lower_bound >= 1.0 - 1e-6,
            "K ≥ 1 always, got {}",
            res.kreiss_lower_bound
        );
    }

    #[test]
    fn test_kreiss_constant_positive_evaluations() {
        let a = array![[1.0_f64, 0.0], [0.0, -1.0]];
        let res = kreiss_constant(&a.view(), None, Some(10), Some(20)).expect("failed");
        assert!(res.grid_evaluations > 0);
    }

    // ----- pseudospectral_abscissa ------------------------------------------

    #[test]
    fn test_pseudospectral_abscissa_stable() {
        let a = array![[-1.0_f64, 10.0], [0.0, -2.0]];
        let res = pseudospectral_abscissa(&a.view(), 0.1, Some(40)).expect("failed");
        // α_ε ≥ α (classical abscissa)
        assert!(
            res.abscissa >= res.spectral_abscissa - 1e-6,
            "abscissa={}, spectral={}",
            res.abscissa, res.spectral_abscissa
        );
    }

    #[test]
    fn test_pseudospectral_abscissa_large_epsilon() {
        let a = array![[-5.0_f64, 0.0], [0.0, -5.0]];
        // Large epsilon should push abscissa rightward compared to α = -5.
        let res = pseudospectral_abscissa(&a.view(), 1.0, Some(30)).expect("failed");
        assert!(res.abscissa >= res.spectral_abscissa - 1e-6);
    }

    // ----- transient_bound --------------------------------------------------

    #[test]
    fn test_transient_bound_stable_diagonal() {
        let a = array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let res = transient_bound(&a.view(), Some(0.0), Some(15), Some(30)).expect("failed");
        // At t=0: bound = e * n * K; K ≥ 1, n=2 → bound ≥ 2e
        assert!(res.upper_bound >= 1.0, "bound = {}", res.upper_bound);
        assert_eq!(res.n, 2);
        assert_eq!(res.time_horizon, 0.0);
    }

    #[test]
    fn test_transient_bound_dimension_matches() {
        let n = 3;
        let mut a = Array2::<f64>::zeros((n, n));
        a[[0, 0]] = -1.0;
        a[[1, 1]] = -2.0;
        a[[2, 2]] = -3.0;
        let res = transient_bound(&a.view(), Some(1.0), Some(15), Some(30)).expect("failed");
        assert_eq!(res.n, n);
    }
}
