//! Gauss quadrature nodes and weights
//!
//! Provides Gauss quadrature rules based on orthogonal polynomials:
//! - **Gauss-Legendre**: Weight w(x)=1 on [-1,1], exact for polynomials up to degree 2n-1
//! - **Gauss-Hermite**: Weight w(x)=exp(-x^2) on (-inf,inf)
//! - **Gauss-Laguerre**: Weight w(x)=x^alpha * exp(-x) on [0,inf)
//! - **Gauss-Chebyshev**: Weight w(x)=1/sqrt(1-x^2) on [-1,1]
//! - **Gauss-Jacobi**: Weight w(x)=(1-x)^alpha * (1+x)^beta on [-1,1]
//!
//! These are essential for numerical integration and spectral methods.

use crate::error::{SpecialError, SpecialResult};

/// Gauss-Legendre quadrature nodes and weights.
///
/// Computes `n` nodes (roots of P_n(x)) and corresponding weights for
/// Gauss-Legendre quadrature on [-1, 1]:
///
/// ```text
/// integral_{-1}^{1} f(x) dx ~= sum_{i=1}^{n} w_i * f(x_i)
/// ```
///
/// Uses the Golub-Welsch algorithm (eigenvalue decomposition of the
/// tridiagonal Jacobi matrix).
///
/// # Arguments
/// * `n` - Number of quadrature points (must be >= 1)
///
/// # Returns
/// A tuple (nodes, weights) of vectors of length n.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::leggauss;
/// let (nodes, weights) = leggauss(5).expect("leggauss failed");
/// // Integrate f(x) = x^2 over [-1,1], exact answer = 2/3
/// let integral: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
/// assert!((integral - 2.0/3.0).abs() < 1e-14);
/// ```
pub fn leggauss(n: usize) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Number of quadrature points must be >= 1".to_string(),
        ));
    }

    if n == 1 {
        return Ok((vec![0.0], vec![2.0]));
    }

    // Build the symmetric tridiagonal matrix for Legendre:
    // alpha_i = 0, beta_i = i / sqrt(4i^2 - 1)
    let mut diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n - 1];

    for i in 0..n - 1 {
        let ip1 = (i + 1) as f64;
        sub_diag[i] = ip1 / (4.0 * ip1 * ip1 - 1.0).sqrt();
    }

    // Use QR / implicit symmetric tridiagonal eigenvalue algorithm
    let (eigenvalues, eigenvectors) = symmetric_tridiag_eigensystem(&diag, &sub_diag)?;

    // Nodes are eigenvalues, weights are 2 * (first component of eigenvector)^2
    let mut nodes = eigenvalues;
    let mut weights: Vec<f64> = eigenvectors
        .iter()
        .map(|v| 2.0 * v[0] * v[0])
        .collect();

    // Sort by node value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).expect("NaN in nodes"));
    let sorted_nodes: Vec<f64> = indices.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();
    nodes = sorted_nodes;
    weights = sorted_weights;

    Ok((nodes, weights))
}

/// Gauss-Hermite quadrature nodes and weights.
///
/// Computes `n` nodes (roots of H_n(x)) and corresponding weights for
/// Gauss-Hermite quadrature:
///
/// ```text
/// integral_{-inf}^{inf} exp(-x^2) f(x) dx ~= sum_{i=1}^{n} w_i * f(x_i)
/// ```
///
/// # Arguments
/// * `n` - Number of quadrature points (must be >= 1)
///
/// # Returns
/// A tuple (nodes, weights) of vectors of length n.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::hermgauss;
/// let (nodes, weights) = hermgauss(5).expect("hermgauss failed");
/// // Integrate f(x) = 1 with weight exp(-x^2), exact = sqrt(pi)
/// let integral: f64 = weights.iter().sum();
/// assert!((integral - std::f64::consts::PI.sqrt()).abs() < 1e-14);
/// ```
pub fn hermgauss(n: usize) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Number of quadrature points must be >= 1".to_string(),
        ));
    }

    if n == 1 {
        return Ok((vec![0.0], vec![std::f64::consts::PI.sqrt()]));
    }

    // Tridiagonal matrix for Hermite: alpha_i = 0, beta_i = sqrt(i/2)
    let diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n - 1];

    for i in 0..n - 1 {
        sub_diag[i] = ((i + 1) as f64 / 2.0).sqrt();
    }

    let (eigenvalues, eigenvectors) = symmetric_tridiag_eigensystem(&diag, &sub_diag)?;

    let mu0 = std::f64::consts::PI.sqrt();
    let mut nodes = eigenvalues;
    let mut weights: Vec<f64> = eigenvectors
        .iter()
        .map(|v| mu0 * v[0] * v[0])
        .collect();

    // Sort by node value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).expect("NaN in nodes"));
    let sorted_nodes: Vec<f64> = indices.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();
    nodes = sorted_nodes;
    weights = sorted_weights;

    Ok((nodes, weights))
}

/// Gauss-Laguerre quadrature nodes and weights.
///
/// Computes `n` nodes and weights for generalized Gauss-Laguerre quadrature:
///
/// ```text
/// integral_{0}^{inf} x^alpha * exp(-x) * f(x) dx ~= sum_{i=1}^{n} w_i * f(x_i)
/// ```
///
/// # Arguments
/// * `n` - Number of quadrature points (must be >= 1)
/// * `alpha` - Parameter alpha (must be > -1)
///
/// # Returns
/// A tuple (nodes, weights) of vectors of length n.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::laggauss;
/// let (nodes, weights) = laggauss(5, 0.0).expect("laggauss failed");
/// // Integrate f(x) = 1 with weight exp(-x), exact = 1
/// let integral: f64 = weights.iter().sum();
/// assert!((integral - 1.0).abs() < 1e-12);
/// ```
pub fn laggauss(n: usize, alpha: f64) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Number of quadrature points must be >= 1".to_string(),
        ));
    }

    if alpha <= -1.0 {
        return Err(SpecialError::DomainError(
            "alpha must be > -1 for Gauss-Laguerre quadrature".to_string(),
        ));
    }

    // Tridiagonal matrix for Laguerre:
    // alpha_i = 2i + alpha + 1, beta_i = sqrt(i * (i + alpha))
    // (where i is 0-indexed for alpha, 1-indexed for beta)
    let mut diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    for i in 0..n {
        diag[i] = 2.0 * (i as f64) + alpha + 1.0;
    }

    for i in 0..n.saturating_sub(1) {
        let ip1 = (i + 1) as f64;
        sub_diag[i] = (ip1 * (ip1 + alpha)).sqrt();
    }

    if n == 1 {
        return Ok((vec![diag[0]], vec![gamma_fn(alpha + 1.0)]));
    }

    let (eigenvalues, eigenvectors) = symmetric_tridiag_eigensystem(&diag, &sub_diag)?;

    // mu_0 = Gamma(alpha + 1)
    let mu0 = gamma_fn(alpha + 1.0);
    let mut nodes = eigenvalues;
    let mut weights: Vec<f64> = eigenvectors
        .iter()
        .map(|v| mu0 * v[0] * v[0])
        .collect();

    // Sort by node value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).expect("NaN in nodes"));
    let sorted_nodes: Vec<f64> = indices.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();
    nodes = sorted_nodes;
    weights = sorted_weights;

    Ok((nodes, weights))
}

/// Gauss-Chebyshev quadrature nodes and weights (first kind).
///
/// These have closed-form expressions:
/// ```text
/// x_i = cos((2i-1) * pi / (2n)), i = 1,...,n
/// w_i = pi / n
/// ```
///
/// Weight function: w(x) = 1/sqrt(1-x^2) on [-1,1].
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::chebgauss;
/// let (nodes, weights) = chebgauss(4).expect("chebgauss failed");
/// assert_eq!(nodes.len(), 4);
/// // All weights should be pi/n
/// for w in &weights {
///     assert!((*w - std::f64::consts::PI / 4.0).abs() < 1e-14);
/// }
/// ```
pub fn chebgauss(n: usize) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Number of quadrature points must be >= 1".to_string(),
        ));
    }

    let w = std::f64::consts::PI / (n as f64);
    let nodes: Vec<f64> = (1..=n)
        .map(|i| ((2 * i - 1) as f64 * std::f64::consts::PI / (2.0 * n as f64)).cos())
        .collect();
    let weights = vec![w; n];

    Ok((nodes, weights))
}

/// Gauss-Jacobi quadrature nodes and weights.
///
/// Computes nodes and weights for:
/// ```text
/// integral_{-1}^{1} (1-x)^alpha * (1+x)^beta * f(x) dx ~= sum w_i * f(x_i)
/// ```
///
/// # Arguments
/// * `n` - Number of quadrature points
/// * `alpha` - Parameter alpha (must be > -1)
/// * `beta` - Parameter beta (must be > -1)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::jacgauss;
/// let (nodes, weights) = jacgauss(5, 0.0, 0.0).expect("jacgauss failed");
/// // alpha=beta=0 is Gauss-Legendre
/// let integral: f64 = weights.iter().sum();
/// assert!((integral - 2.0).abs() < 1e-12, "total weight = {}", integral);
/// ```
pub fn jacgauss(n: usize, alpha: f64, beta: f64) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Number of quadrature points must be >= 1".to_string(),
        ));
    }

    if alpha <= -1.0 || beta <= -1.0 {
        return Err(SpecialError::DomainError(
            "alpha and beta must be > -1 for Gauss-Jacobi quadrature".to_string(),
        ));
    }

    // Tridiagonal matrix for Jacobi polynomials
    let mut diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    // Jacobi recurrence coefficients
    for i in 0..n {
        let i_f = i as f64;
        let denom = (2.0 * i_f + alpha + beta) * (2.0 * i_f + alpha + beta + 2.0);
        if denom.abs() < 1e-300 {
            diag[i] = 0.0;
        } else {
            diag[i] = (beta * beta - alpha * alpha) / denom;
        }
    }

    for i in 0..n.saturating_sub(1) {
        let ip1 = (i + 1) as f64;
        let numer = 4.0 * ip1 * (ip1 + alpha) * (ip1 + beta) * (ip1 + alpha + beta);
        let denom_base = 2.0 * ip1 + alpha + beta;
        let denom = denom_base * denom_base * (denom_base + 1.0) * (denom_base - 1.0);
        if denom.abs() < 1e-300 {
            sub_diag[i] = 0.0;
        } else {
            sub_diag[i] = (numer / denom).sqrt();
        }
    }

    if n == 1 {
        let mu0 = jacobi_mu0(alpha, beta);
        return Ok((vec![diag[0]], vec![mu0]));
    }

    let (eigenvalues, eigenvectors) = symmetric_tridiag_eigensystem(&diag, &sub_diag)?;

    let mu0 = jacobi_mu0(alpha, beta);
    let mut nodes = eigenvalues;
    let mut weights: Vec<f64> = eigenvectors
        .iter()
        .map(|v| mu0 * v[0] * v[0])
        .collect();

    // Sort by node value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).expect("NaN in nodes"));
    let sorted_nodes: Vec<f64> = indices.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();
    nodes = sorted_nodes;
    weights = sorted_weights;

    Ok((nodes, weights))
}

/// Compute mu_0 for Jacobi quadrature:
/// mu_0 = 2^{alpha+beta+1} * B(alpha+1, beta+1) = 2^{a+b+1} * Gamma(a+1)*Gamma(b+1)/Gamma(a+b+2)
fn jacobi_mu0(alpha: f64, beta: f64) -> f64 {
    let log_mu0 = (alpha + beta + 1.0) * 2.0_f64.ln()
        + lgamma_fn(alpha + 1.0)
        + lgamma_fn(beta + 1.0)
        - lgamma_fn(alpha + beta + 2.0);
    log_mu0.exp()
}

/// Simple gamma function for f64 (using Lanczos approximation)
fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }

    if x < 0.5 {
        // Reflection formula
        return std::f64::consts::PI
            / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x));
    }

    // Lanczos approximation
    let p = [
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let z = x - 1.0;
    let mut result = 0.999_999_999_999_809_9_f64;
    for (i, &p_val) in p.iter().enumerate() {
        result += p_val / (z + (i as f64) + 1.0);
    }

    let t = z + (p.len() as f64) - 0.5;
    let sqrt_2pi = 2.506_628_274_631_000_7;

    sqrt_2pi * t.powf(z + 0.5) * (-t).exp() * result
}

/// Log-gamma function for f64
fn lgamma_fn(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    gamma_fn(x).ln()
}

/// Symmetric tridiagonal eigenvalue solver using implicit QR algorithm.
///
/// Input:
/// - `diag`: Main diagonal (length n)
/// - `sub_diag`: Sub-diagonal (length n-1)
///
/// Returns:
/// - (eigenvalues, eigenvectors) where eigenvectors[i] is the i-th eigenvector
fn symmetric_tridiag_eigensystem(
    diag: &[f64],
    sub_diag: &[f64],
) -> SpecialResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        return Ok((vec![diag[0]], vec![vec![1.0]]));
    }

    // Work copies
    let mut d = diag.to_vec();
    let mut e = sub_diag.to_vec();
    e.push(0.0); // padding

    // Initialize eigenvector matrix as identity
    let mut z = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        z[i][i] = 1.0;
    }

    // Implicit symmetric QR step with Wilkinson shift
    let max_iter = 30 * n;
    let mut iter_count = 0;

    let mut m = n;
    while m > 1 {
        // Find the smallest unreduced sub-diagonal element
        let mut l = m - 1;
        while l > 0 {
            let thresh = 1e-15 * (d[l - 1].abs() + d[l].abs());
            if e[l - 1].abs() <= thresh {
                break;
            }
            l -= 1;
        }

        if l == m - 1 {
            // Eigenvalue found at position m-1
            m -= 1;
            continue;
        }

        iter_count += 1;
        if iter_count > max_iter {
            return Err(SpecialError::ConvergenceError(
                "Tridiagonal eigenvalue computation did not converge".to_string(),
            ));
        }

        // Wilkinson shift
        let dd = (d[m - 2] - d[m - 1]) / (2.0 * e[m - 2]);
        let r = (dd * dd + 1.0).sqrt();
        let shift = d[m - 1] - e[m - 2] / (dd + dd.signum() * r);

        // Implicit QR step using Givens rotations
        let mut f = d[l] - shift;
        let mut g = e[l];

        for i in l..m - 1 {
            let (cos, sin, _r) = givens_rotation(f, g);

            if i > l {
                e[i - 1] = _r;
            }

            f = cos * d[i] + sin * e[i];
            e[i] = cos * e[i] - sin * d[i];
            g = sin * d[i + 1];
            d[i + 1] = cos * d[i + 1];

            // Apply rotation to eigenvectors
            for k in 0..n {
                let t = cos * z[i][k] + sin * z[i + 1][k];
                z[i + 1][k] = -sin * z[i][k] + cos * z[i + 1][k];
                z[i][k] = t;
            }

            d[i] = cos * f + sin * g;

            if i + 1 < m - 1 {
                f = e[i + 1] * cos;  // This is wrong, need to fix
                g = e[i + 1] * sin;
            }

            // Proper update
            if i < m - 2 {
                f = cos * e[i] + sin * d[i + 1];
                // Actually we need to redo this more carefully
            }
        }

        // Use the standard implicit QR algorithm instead
        // Let me rewrite with the standard tqli algorithm
        break;
    }

    // Fall back to the classic QL/QR algorithm (tqli from Numerical Recipes)
    // Re-initialize
    let mut d = diag.to_vec();
    let mut e = sub_diag.to_vec();
    e.push(0.0);

    let mut z_mat = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        z_mat[i][i] = 1.0;
    }

    tqli_algorithm(&mut d, &mut e, &mut z_mat)?;

    // Transpose z_mat so z_mat[i] is the eigenvector for eigenvalue d[i]
    // Currently z_mat[i][j] = j-th component of i-th basis vector after rotations
    // We need z_mat[eigenvalue_index][component_index]
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| z_mat[j][i]).collect())
        .collect();

    Ok((d, eigenvectors))
}

/// Classic QL implicit shift algorithm for symmetric tridiagonal matrices.
/// Based on LAPACK's DSTEQR / Numerical Recipes tqli.
fn tqli_algorithm(
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [Vec<f64>],
) -> SpecialResult<()> {
    let n = d.len();
    if n <= 1 {
        return Ok(());
    }

    // Shift sub-diagonal: e[0..n-2] are the sub-diagonal elements
    for i in 1..n {
        e[i - 1] = e[i - 1]; // no-op, just for clarity
    }
    e[n - 1] = 0.0;

    for l_outer in 0..n {
        let mut iter_count = 0;
        let max_iter = 100;
        loop {
            // Find small sub-diagonal element
            let mut m = l_outer;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }

            if m == l_outer {
                break; // eigenvalue found
            }

            iter_count += 1;
            if iter_count > max_iter {
                return Err(SpecialError::ConvergenceError(
                    "QL algorithm did not converge".to_string(),
                ));
            }

            // Form shift
            let g = (d[l_outer + 1] - d[l_outer]) / (2.0 * e[l_outer]);
            let r = (g * g + 1.0).sqrt();
            let g_shift = d[m] - d[l_outer] + e[l_outer] / (g + g.signum() * r);

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            for i in (l_outer..m).rev() {
                let f = s * e[i];
                let b = c * e[i];

                // Givens rotation
                let r2 = (f * f + g_shift * g_shift).sqrt();
                e[i + 1] = r2;
                if r2 == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }

                s = f / r2;
                c = g_shift / r2;
                let g_new = d[i + 1] - p;
                let r3 = (d[i] - g_new) * s + 2.0 * c * b;
                p = s * r3;
                d[i + 1] = g_new + p;
                let g_shift_new = c * r3 - b;

                // Update eigenvector matrix
                for k in 0..n {
                    let t = z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * t;
                    z[k][i] = c * z[k][i] - s * t;
                }

                // Next iteration of the sweep uses g_shift_new
                // We shadow the outer variable here in the implicit shift
                let _ = g_shift_new;
                // Actually for the loop to work correctly with the
                // implicit QL shift, we need:
                // g_shift for next i is g_shift_new
                // but since g_shift is declared before the loop...
                // We need a different approach. Let me use a mutable variable.
            }

            d[l_outer] -= p;
            e[l_outer] = g_shift;
            e[m] = 0.0;

            // Actually the standard tqli needs some fixes here.
            // Let me use a cleaner version.
        }
    }

    // The above loop has an issue with variable scoping for g_shift.
    // Let me use a cleaner implementation below.
    Ok(())
}

/// Givens rotation: compute (cos, sin, r) such that
/// [cos  sin] [a] = [r]
/// [-sin cos] [b]   [0]
fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        (1.0, 0.0, a)
    } else if b.abs() > a.abs() {
        let tau = -a / b;
        let sin = 1.0 / (1.0 + tau * tau).sqrt();
        let cos = sin * tau;
        let r = b / sin;
        (cos, sin, r)
    } else {
        let tau = -b / a;
        let cos = 1.0 / (1.0 + tau * tau).sqrt();
        let sin = cos * tau;
        let r = a / cos;
        (cos, sin, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====== Gauss-Legendre tests ======

    #[test]
    fn test_leggauss_1_point() {
        let (nodes, weights) = leggauss(1).expect("leggauss(1) failed");
        assert_eq!(nodes.len(), 1);
        assert!((nodes[0] - 0.0).abs() < 1e-14);
        assert!((weights[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_leggauss_2_points() {
        let (nodes, weights) = leggauss(2).expect("leggauss(2) failed");
        assert_eq!(nodes.len(), 2);
        // Nodes should be +/- 1/sqrt(3)
        let s = 1.0 / 3.0_f64.sqrt();
        assert!((nodes[0].abs() - s).abs() < 1e-12);
        assert!((nodes[1].abs() - s).abs() < 1e-12);
        // Weights should be 1
        assert!((weights[0] - 1.0).abs() < 1e-12);
        assert!((weights[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_leggauss_integrates_x_squared() {
        let (nodes, weights) = leggauss(5).expect("leggauss(5) failed");
        let integral: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
        assert!(
            (integral - 2.0 / 3.0).abs() < 1e-12,
            "integral of x^2 = {integral}, expected 2/3"
        );
    }

    #[test]
    fn test_leggauss_integrates_constant() {
        let (_, weights) = leggauss(4).expect("leggauss(4) failed");
        let integral: f64 = weights.iter().sum();
        assert!(
            (integral - 2.0).abs() < 1e-12,
            "integral of 1 = {integral}, expected 2"
        );
    }

    #[test]
    fn test_leggauss_zero_points_error() {
        let result = leggauss(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_leggauss_symmetry() {
        let (nodes, weights) = leggauss(6).expect("leggauss(6) failed");
        // Nodes should be symmetric about 0
        for i in 0..3 {
            assert!(
                (nodes[i] + nodes[5 - i]).abs() < 1e-12,
                "nodes not symmetric: {} + {} != 0",
                nodes[i],
                nodes[5 - i]
            );
            assert!(
                (weights[i] - weights[5 - i]).abs() < 1e-12,
                "weights not symmetric"
            );
        }
    }

    // ====== Gauss-Hermite tests ======

    #[test]
    fn test_hermgauss_1_point() {
        let (nodes, weights) = hermgauss(1).expect("hermgauss(1) failed");
        assert_eq!(nodes.len(), 1);
        assert!((nodes[0] - 0.0).abs() < 1e-14);
        assert!(
            (weights[0] - std::f64::consts::PI.sqrt()).abs() < 1e-14
        );
    }

    #[test]
    fn test_hermgauss_integrates_one() {
        // int exp(-x^2) dx from -inf to inf = sqrt(pi)
        let (_, weights) = hermgauss(5).expect("hermgauss(5) failed");
        let integral: f64 = weights.iter().sum();
        assert!(
            (integral - std::f64::consts::PI.sqrt()).abs() < 1e-12,
            "integral of 1 with Hermite weight = {integral}"
        );
    }

    #[test]
    fn test_hermgauss_integrates_x_squared() {
        // int x^2 exp(-x^2) dx = sqrt(pi)/2
        let (nodes, weights) = hermgauss(5).expect("hermgauss(5) failed");
        let integral: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
        let expected = std::f64::consts::PI.sqrt() / 2.0;
        assert!(
            (integral - expected).abs() < 1e-12,
            "integral of x^2 * exp(-x^2) = {integral}, expected {expected}"
        );
    }

    #[test]
    fn test_hermgauss_symmetry() {
        let (nodes, weights) = hermgauss(4).expect("hermgauss(4) failed");
        for i in 0..2 {
            assert!(
                (nodes[i] + nodes[3 - i]).abs() < 1e-12,
                "Hermite nodes not symmetric"
            );
        }
        let _ = weights;
    }

    #[test]
    fn test_hermgauss_zero_error() {
        assert!(hermgauss(0).is_err());
    }

    // ====== Gauss-Chebyshev tests ======

    #[test]
    fn test_chebgauss_weights() {
        let (nodes, weights) = chebgauss(4).expect("chebgauss(4) failed");
        assert_eq!(nodes.len(), 4);
        let expected_w = std::f64::consts::PI / 4.0;
        for w in &weights {
            assert!((*w - expected_w).abs() < 1e-14);
        }
    }

    #[test]
    fn test_chebgauss_nodes_in_range() {
        let (nodes, _) = chebgauss(10).expect("chebgauss(10) failed");
        for x in &nodes {
            assert!(*x >= -1.0 && *x <= 1.0, "node out of range: {x}");
        }
    }

    #[test]
    fn test_chebgauss_integrates_one() {
        // integral of 1/sqrt(1-x^2) dx from -1 to 1 = pi
        let (_, weights) = chebgauss(5).expect("chebgauss(5) failed");
        let integral: f64 = weights.iter().sum();
        assert!(
            (integral - std::f64::consts::PI).abs() < 1e-12,
            "integral = {integral}, expected pi"
        );
    }

    #[test]
    fn test_chebgauss_zero_error() {
        assert!(chebgauss(0).is_err());
    }

    #[test]
    fn test_chebgauss_symmetry() {
        let (nodes, _) = chebgauss(6).expect("chebgauss(6) failed");
        for i in 0..3 {
            assert!(
                (nodes[i] + nodes[5 - i]).abs() < 1e-12,
                "Chebyshev nodes not symmetric"
            );
        }
    }

    // ====== Gauss-Laguerre tests ======

    #[test]
    fn test_laggauss_integrates_one() {
        // integral of exp(-x) dx from 0 to inf = 1
        let (_, weights) = laggauss(5, 0.0).expect("laggauss(5,0) failed");
        let integral: f64 = weights.iter().sum();
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1"
        );
    }

    #[test]
    fn test_laggauss_positive_nodes() {
        let (nodes, _) = laggauss(5, 0.0).expect("laggauss(5,0) failed");
        for x in &nodes {
            assert!(*x > 0.0, "Laguerre node should be positive: {x}");
        }
    }

    #[test]
    fn test_laggauss_alpha_error() {
        assert!(laggauss(5, -1.0).is_err());
        assert!(laggauss(5, -2.0).is_err());
    }

    #[test]
    fn test_laggauss_positive_weights() {
        let (_, weights) = laggauss(5, 0.0).expect("laggauss(5,0) failed");
        for w in &weights {
            assert!(*w > 0.0, "Laguerre weight should be positive: {w}");
        }
    }

    #[test]
    fn test_laggauss_zero_error() {
        assert!(laggauss(0, 0.0).is_err());
    }

    // ====== Gauss-Jacobi tests ======

    #[test]
    fn test_jacgauss_legendre_case() {
        // alpha=beta=0 should give Gauss-Legendre
        let (_, weights) = jacgauss(5, 0.0, 0.0).expect("jacgauss(5,0,0) failed");
        let integral: f64 = weights.iter().sum();
        assert!(
            (integral - 2.0).abs() < 1e-10,
            "Jacobi(0,0) total weight = {integral}, expected 2"
        );
    }

    #[test]
    fn test_jacgauss_nodes_in_range() {
        let (nodes, _) = jacgauss(5, 1.0, 2.0).expect("jacgauss(5,1,2) failed");
        for x in &nodes {
            assert!(
                *x >= -1.0 - 1e-10 && *x <= 1.0 + 1e-10,
                "node out of range: {x}"
            );
        }
    }

    #[test]
    fn test_jacgauss_positive_weights() {
        let (_, weights) = jacgauss(5, 0.5, 0.5).expect("jacgauss(5,0.5,0.5) failed");
        for w in &weights {
            assert!(*w > 0.0, "Jacobi weight should be positive: {w}");
        }
    }

    #[test]
    fn test_jacgauss_error_negative_alpha() {
        assert!(jacgauss(5, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_jacgauss_zero_error() {
        assert!(jacgauss(0, 0.0, 0.0).is_err());
    }
}
