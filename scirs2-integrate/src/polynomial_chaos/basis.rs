//! Orthogonal polynomial basis evaluation and multi-index generation.
//!
//! Provides evaluation of classical orthogonal polynomials via three-term
//! recurrence relations, multi-index generation for multi-dimensional expansions,
//! and Gauss quadrature rules for each polynomial family.

use crate::error::{IntegrateError, IntegrateResult};
use std::f64::consts::PI;

use super::types::{PolynomialBasis, TruncationScheme};

// ---------------------------------------------------------------------------
// 1-D polynomial evaluation
// ---------------------------------------------------------------------------

/// Evaluate a 1-D orthogonal polynomial of the given `degree` at point `x`.
///
/// Uses the three-term recurrence relation for each polynomial family.
///
/// # Polynomial families
///
/// | Family | Recurrence |
/// |--------|-----------|
/// | Hermite (probabilist) | H_{n+1} = x H_n - n H_{n-1} |
/// | Legendre | (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1} |
/// | Laguerre | (n+1) L_{n+1} = (2n+1-x) L_n - n L_{n-1} |
/// | Jacobi | Standard three-term for P_n^{(a,b)}(x) |
pub fn evaluate_basis_1d(basis: &PolynomialBasis, degree: usize, x: f64) -> IntegrateResult<f64> {
    match basis {
        PolynomialBasis::Hermite => evaluate_hermite(degree, x),
        PolynomialBasis::Legendre => evaluate_legendre(degree, x),
        PolynomialBasis::Laguerre => evaluate_laguerre(degree, x),
        PolynomialBasis::Jacobi { alpha, beta } => evaluate_jacobi(degree, x, *alpha, *beta),
    }
}

/// Probabilist's Hermite polynomial H_n(x).
///
/// H_0 = 1, H_1 = x, H_{n+1} = x H_n - n H_{n-1}.
fn evaluate_hermite(degree: usize, x: f64) -> IntegrateResult<f64> {
    if degree == 0 {
        return Ok(1.0);
    }
    if degree == 1 {
        return Ok(x);
    }
    let mut h_prev = 1.0_f64;
    let mut h_curr = x;
    for n in 1..degree {
        let h_next = x * h_curr - (n as f64) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    Ok(h_curr)
}

/// Legendre polynomial P_n(x).
///
/// P_0 = 1, P_1 = x, (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.
fn evaluate_legendre(degree: usize, x: f64) -> IntegrateResult<f64> {
    if degree == 0 {
        return Ok(1.0);
    }
    if degree == 1 {
        return Ok(x);
    }
    let mut p_prev = 1.0_f64;
    let mut p_curr = x;
    for n in 1..degree {
        let nf = n as f64;
        let p_next = ((2.0 * nf + 1.0) * x * p_curr - nf * p_prev) / (nf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }
    Ok(p_curr)
}

/// Laguerre polynomial L_n(x).
///
/// L_0 = 1, L_1 = 1 - x, (n+1) L_{n+1} = (2n+1-x) L_n - n L_{n-1}.
fn evaluate_laguerre(degree: usize, x: f64) -> IntegrateResult<f64> {
    if degree == 0 {
        return Ok(1.0);
    }
    if degree == 1 {
        return Ok(1.0 - x);
    }
    let mut l_prev = 1.0_f64;
    let mut l_curr = 1.0 - x;
    for n in 1..degree {
        let nf = n as f64;
        let l_next = ((2.0 * nf + 1.0 - x) * l_curr - nf * l_prev) / (nf + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    Ok(l_curr)
}

/// Jacobi polynomial P_n^{(alpha, beta)}(x).
///
/// Uses the standard three-term recurrence relation.
fn evaluate_jacobi(degree: usize, x: f64, alpha: f64, beta: f64) -> IntegrateResult<f64> {
    if alpha <= -1.0 || beta <= -1.0 {
        return Err(IntegrateError::ValueError(
            "Jacobi parameters alpha and beta must be > -1".to_string(),
        ));
    }
    if degree == 0 {
        return Ok(1.0);
    }
    if degree == 1 {
        return Ok(0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x));
    }
    let mut p_prev = 1.0_f64;
    let mut p_curr = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
    for n in 1..degree {
        let nf = n as f64;
        let a = alpha;
        let b = beta;
        let c1 = 2.0 * (nf + 1.0) * (nf + a + b + 1.0) * (2.0 * nf + a + b);
        let c2 = (2.0 * nf + a + b + 1.0) * (a * a - b * b);
        let c3 = (2.0 * nf + a + b) * (2.0 * nf + a + b + 1.0) * (2.0 * nf + a + b + 2.0);
        let c4 = 2.0 * (nf + a) * (nf + b) * (2.0 * nf + a + b + 2.0);

        if c1.abs() < 1e-30 {
            return Err(IntegrateError::ComputationError(
                "Degenerate Jacobi recurrence coefficient".to_string(),
            ));
        }
        let p_next = ((c2 + c3 * x) * p_curr - c4 * p_prev) / c1;
        p_prev = p_curr;
        p_curr = p_next;
    }
    Ok(p_curr)
}

// ---------------------------------------------------------------------------
// Multi-dimensional basis evaluation
// ---------------------------------------------------------------------------

/// Evaluate a multi-dimensional basis function as a product of 1-D polynomials.
///
/// Given bases \[B_1, ..., B_d\] and multi-index \[k_1, ..., k_d\],
/// returns the product: Psi_k(x) = prod_i B_i(k_i, x_i).
pub fn evaluate_basis_nd(
    bases: &[PolynomialBasis],
    multi_index: &[usize],
    x: &[f64],
) -> IntegrateResult<f64> {
    if bases.len() != multi_index.len() || bases.len() != x.len() {
        return Err(IntegrateError::DimensionMismatch(format!(
            "bases ({}), multi_index ({}), and x ({}) must have the same length",
            bases.len(),
            multi_index.len(),
            x.len()
        )));
    }
    let mut product = 1.0_f64;
    for i in 0..bases.len() {
        product *= evaluate_basis_1d(&bases[i], multi_index[i], x[i])?;
    }
    Ok(product)
}

// ---------------------------------------------------------------------------
// Squared norms
// ---------------------------------------------------------------------------

/// Compute the squared norm ||Psi_k||^2 = E\[Psi_k^2\] for a 1-D basis of given degree.
///
/// | Family | ||P_n||^2 |
/// |--------|-----------|
/// | Hermite (probabilist) | n! |
/// | Legendre | 1 / (2n + 1) |
/// | Laguerre | 1 |
/// | Jacobi(a,b) | analytic formula |
pub fn basis_norm_squared_1d(basis: &PolynomialBasis, degree: usize) -> f64 {
    match basis {
        PolynomialBasis::Hermite => factorial(degree),
        PolynomialBasis::Legendre => 1.0 / (2.0 * degree as f64 + 1.0),
        PolynomialBasis::Laguerre => 1.0,
        PolynomialBasis::Jacobi { alpha, beta } => jacobi_norm_squared(degree, *alpha, *beta),
    }
}

/// Compute the multi-dimensional squared norm as a product of 1-D squared norms.
pub fn basis_norm_squared_nd(bases: &[PolynomialBasis], multi_index: &[usize]) -> f64 {
    let mut product = 1.0_f64;
    for (basis, &deg) in bases.iter().zip(multi_index.iter()) {
        product *= basis_norm_squared_1d(basis, deg);
    }
    product
}

/// n! as f64.
fn factorial(n: usize) -> f64 {
    let mut result = 1.0_f64;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Squared norm for Jacobi polynomial P_n^{(a,b)} with weight (1-x)^a (1+x)^b on \[-1,1\].
///
/// ||P_n^{(a,b)}||^2 = 2^{a+b+1} / (2n+a+b+1) * Gamma(n+a+1)*Gamma(n+b+1) / (n! * Gamma(n+a+b+1))
fn jacobi_norm_squared(n: usize, a: f64, b: f64) -> f64 {
    if n == 0 {
        // 2^{a+b+1} * B(a+1, b+1)
        let log_norm = (a + b + 1.0) * 2.0_f64.ln() + ln_gamma(a + 1.0) + ln_gamma(b + 1.0)
            - ln_gamma(a + b + 2.0);
        return log_norm.exp();
    }
    let nf = n as f64;
    let log_num = (a + b + 1.0) * 2.0_f64.ln() + ln_gamma(nf + a + 1.0) + ln_gamma(nf + b + 1.0);
    let log_den = (2.0 * nf + a + b + 1.0).ln() + ln_gamma(nf + 1.0) + ln_gamma(nf + a + b + 1.0);
    (log_num - log_den).exp()
}

/// Natural log of the Gamma function (Stirling series for large arguments, Lanczos for small).
fn ln_gamma(x: f64) -> f64 {
    // Use the Lanczos approximation (g=7, n=9 coefficients)
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        let log_pi_over_sin = (PI / (PI * x).sin()).abs().ln();
        return log_pi_over_sin - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let lanczos_g = 7.0_f64;
    let coefficients = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let mut sum = coefficients[0];
    for (i, &c) in coefficients.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }

    let t = z + lanczos_g + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

// ---------------------------------------------------------------------------
// Multi-index generation
// ---------------------------------------------------------------------------

/// Generate multi-indices for `dim` dimensions with maximum degree `max_degree`
/// under the given truncation scheme.
///
/// Returns a vector of multi-indices sorted in graded lexicographic order.
pub fn generate_multi_indices(
    dim: usize,
    max_degree: usize,
    scheme: &TruncationScheme,
) -> Vec<Vec<usize>> {
    if dim == 0 {
        return vec![vec![]];
    }
    match scheme {
        TruncationScheme::TotalDegree => generate_total_degree(dim, max_degree),
        TruncationScheme::Hyperbolic { q } => generate_hyperbolic(dim, max_degree, *q),
        TruncationScheme::Tensor => generate_tensor(dim, max_degree),
    }
}

/// Total degree truncation: all alpha with sum(alpha_i) <= p.
fn generate_total_degree(dim: usize, max_degree: usize) -> Vec<Vec<usize>> {
    let mut indices = Vec::new();
    let mut current = vec![0usize; dim];
    generate_total_degree_recursive(&mut indices, &mut current, 0, max_degree);
    indices
}

fn generate_total_degree_recursive(
    indices: &mut Vec<Vec<usize>>,
    current: &mut Vec<usize>,
    pos: usize,
    remaining: usize,
) {
    let dim = current.len();
    if pos == dim {
        indices.push(current.clone());
        return;
    }
    let max_val = remaining;
    for val in 0..=max_val {
        current[pos] = val;
        generate_total_degree_recursive(indices, current, pos + 1, remaining - val);
    }
}

/// Hyperbolic truncation: (sum alpha_i^q)^{1/q} <= p.
fn generate_hyperbolic(dim: usize, max_degree: usize, q: f64) -> Vec<Vec<usize>> {
    // Generate all total-degree candidates and filter by hyperbolic criterion.
    let total_degree_indices = generate_total_degree(dim, max_degree);
    let p_f = max_degree as f64;

    total_degree_indices
        .into_iter()
        .filter(|alpha| {
            let norm: f64 = alpha.iter().map(|&a| (a as f64).powf(q)).sum();
            norm.powf(1.0 / q) <= p_f + 1e-12
        })
        .collect()
}

/// Tensor product truncation: each alpha_i <= p.
fn generate_tensor(dim: usize, max_degree: usize) -> Vec<Vec<usize>> {
    let mut indices = Vec::new();
    let mut current = vec![0usize; dim];
    generate_tensor_recursive(&mut indices, &mut current, 0, max_degree);
    indices
}

fn generate_tensor_recursive(
    indices: &mut Vec<Vec<usize>>,
    current: &mut Vec<usize>,
    pos: usize,
    max_degree: usize,
) {
    if pos == current.len() {
        indices.push(current.clone());
        return;
    }
    for val in 0..=max_degree {
        current[pos] = val;
        generate_tensor_recursive(indices, current, pos + 1, max_degree);
    }
}

// ---------------------------------------------------------------------------
// Gauss quadrature rules
// ---------------------------------------------------------------------------

/// Compute Gauss quadrature nodes and weights for the given polynomial basis.
///
/// Returns `(nodes, weights)` for an `order`-point rule that integrates
/// polynomials of degree up to `2*order - 1` exactly.
pub fn gauss_quadrature(
    basis: &PolynomialBasis,
    order: usize,
) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    if order == 0 {
        return Err(IntegrateError::ValueError(
            "Quadrature order must be >= 1".to_string(),
        ));
    }
    match basis {
        PolynomialBasis::Hermite => gauss_hermite(order),
        PolynomialBasis::Legendre => gauss_legendre_nodes_weights(order),
        PolynomialBasis::Laguerre => gauss_laguerre(order),
        PolynomialBasis::Jacobi { alpha, beta } => gauss_jacobi(order, *alpha, *beta),
    }
}

/// Gauss-Hermite quadrature for probabilist's Hermite (standard normal measure).
///
/// Returns (nodes, weights) such that:
/// sum_i w_i f(x_i) ≈ E\[f(X)\] = integral f(x) exp(-x^2/2)/sqrt(2 pi) dx
///
/// The Golub-Welsch algorithm returns z_i^2 (summing to 1), which directly
/// gives the expectation weights since mu_0/sqrt(2pi) = 1 for the probabilist
/// convention (mu_0 = sqrt(2pi)).
fn gauss_hermite(n: usize) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let mut diag = vec![0.0_f64; n];
    let mut off_diag = vec![0.0_f64; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        off_diag[i] = ((i + 1) as f64).sqrt();
    }
    golub_welsch(&diag, &off_diag)
}

/// Gauss-Legendre quadrature on \[-1, 1\] (uniform probability measure).
///
/// Returns (nodes, weights) such that:
/// sum_i w_i f(x_i) ≈ E\[f(X)\] = (1/2) integral_{-1}^{1} f(x) dx
///
/// Weights sum to 1 (probability measure normalization).
fn gauss_legendre_nodes_weights(n: usize) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let mut diag = vec![0.0_f64; n];
    let mut off_diag = vec![0.0_f64; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        let ip1 = (i + 1) as f64;
        off_diag[i] = ip1 / (4.0 * ip1 * ip1 - 1.0).sqrt();
    }
    // Golub-Welsch returns z_i^2 summing to 1, which directly gives
    // E[f(X)] for X ~ Uniform[-1,1] since mu_0/2 = 2/2 = 1.
    golub_welsch(&diag, &off_diag)
}

/// Gauss-Laguerre quadrature on \[0, inf) with exp(-x) probability measure.
///
/// Returns (nodes, weights) such that:
/// sum_i w_i f(x_i) ≈ E\[f(X)\] = integral_0^inf f(x) exp(-x) dx
///
/// For Laguerre, mu_0 = integral exp(-x) dx = 1, so Golub-Welsch z_i^2
/// directly gives expectation weights.
fn gauss_laguerre(n: usize) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let mut diag = vec![0.0_f64; n];
    let mut off_diag = vec![0.0_f64; n.saturating_sub(1)];
    for i in 0..n {
        diag[i] = 2.0 * i as f64 + 1.0;
    }
    for i in 0..n.saturating_sub(1) {
        off_diag[i] = (i + 1) as f64;
    }
    golub_welsch(&diag, &off_diag)
}

/// Gauss-Jacobi quadrature on \[-1, 1\] with weight (1-x)^a (1+x)^b.
fn gauss_jacobi(n: usize, a: f64, b: f64) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    if a <= -1.0 || b <= -1.0 {
        return Err(IntegrateError::ValueError(
            "Jacobi parameters alpha and beta must be > -1".to_string(),
        ));
    }
    let mut diag = vec![0.0_f64; n];
    let mut off_diag = vec![0.0_f64; n.saturating_sub(1)];

    // Three-term recurrence coefficients for monic Jacobi polynomials
    for i in 0..n {
        let nf = i as f64;
        let denom = 2.0 * nf + a + b;
        if denom.abs() < 1e-30 && i == 0 {
            diag[i] = (b - a) / (a + b + 2.0);
        } else {
            diag[i] = (b * b - a * a) / (denom * (denom + 2.0));
        }
    }
    for i in 0..n.saturating_sub(1) {
        let nf = (i + 1) as f64;
        let denom = 2.0 * nf + a + b;
        off_diag[i] = (4.0 * nf * (nf + a) * (nf + b) * (nf + a + b)
            / (denom * denom * (denom + 1.0) * (denom - 1.0)))
            .sqrt();
    }

    // Golub-Welsch returns z_i^2 (summing to 1), which already gives
    // expectation weights for the Beta(a+1, b+1) probability measure
    // on [-1, 1] (the weight function normalized to integrate to 1).
    golub_welsch(&diag, &off_diag)
}

// ---------------------------------------------------------------------------
// Golub-Welsch algorithm
// ---------------------------------------------------------------------------

/// Golub-Welsch algorithm: compute quadrature nodes and weights from the
/// symmetric tridiagonal Jacobi matrix of recurrence coefficients.
///
/// Input: diagonal `diag` (length n) and sub-diagonal `off_diag` (length n-1).
///
/// Output: (nodes, weights) where nodes are eigenvalues and
/// weights\[i\] = first_component^2 of eigenvector i.
fn golub_welsch(diag: &[f64], off_diag: &[f64]) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    symmetric_tridiag_eig(diag, off_diag)
}

/// Symmetric tridiagonal eigenvalue/eigenvector computation.
///
/// Uses the Jacobi eigenvalue algorithm on the full matrix built from the
/// tridiagonal form. Returns eigenvalues and the squared first components
/// of eigenvectors (for Golub-Welsch quadrature weights).
fn symmetric_tridiag_eig(diag: &[f64], off_diag: &[f64]) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let n = diag.len();
    if n == 0 {
        return Err(IntegrateError::ValueError(
            "Empty quadrature matrix".to_string(),
        ));
    }
    if n == 1 {
        return Ok((vec![diag[0]], vec![1.0]));
    }
    if n == 2 {
        // 2x2 symmetric: [d0, e; e, d1]
        let d0 = diag[0];
        let d1 = diag[1];
        let e = off_diag[0];
        let avg = (d0 + d1) / 2.0;
        let diff = (d0 - d1) / 2.0;
        let disc = (diff * diff + e * e).sqrt();
        let lam1 = avg - disc;
        let lam2 = avg + disc;

        // Eigenvectors
        let (v10, v11, v20, v21) = if e.abs() < 1e-30 {
            (1.0, 0.0, 0.0, 1.0)
        } else {
            let theta = 0.5 * (d1 - d0).atan2(2.0 * e);
            let c = theta.cos();
            let s = theta.sin();
            // Eigenvector for lam1: (c, s), for lam2: (-s, c)
            // But we need to verify which is which
            (c, s, -s, c)
        };

        let nodes = vec![lam1, lam2];
        let weights = vec![v10 * v10, v20 * v20];
        return Ok((nodes, weights));
    }

    // Build full symmetric matrix from tridiagonal form
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = diag[i];
    }
    for i in 0..off_diag.len() {
        a[i * n + (i + 1)] = off_diag[i];
        a[(i + 1) * n + i] = off_diag[i];
    }

    // Eigenvector matrix (starts as identity)
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Jacobi eigenvalue algorithm
    let max_sweeps = 100;
    for _sweep in 0..max_sweeps {
        // Check convergence: sum of off-diagonal elements
        let mut off_diag_sum = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sum += a[i * n + j].abs();
            }
        }
        if off_diag_sum < 1e-15 * n as f64 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-30 {
                    continue;
                }

                let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_val = s / (1.0 + c);

                // Update matrix
                a[p * n + p] -= t * apq;
                a[q * n + q] += t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = arp - s * (arq + tau_val * arp);
                        a[p * n + r] = a[r * n + p];
                        a[r * n + q] = arq + s * (arp - tau_val * arq);
                        a[q * n + r] = a[r * n + q];
                    }
                }

                // Update eigenvectors
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = vrp - s * (vrq + tau_val * vrp);
                    v[r * n + q] = vrq + s * (vrp - tau_val * vrq);
                }
            }
        }
    }

    // Extract eigenvalues (diagonal of a) and first-row components of eigenvectors
    let mut evals = Vec::with_capacity(n);
    let mut first_components = Vec::with_capacity(n);
    for i in 0..n {
        evals.push(a[i * n + i]);
        first_components.push(v[i]); // first row, i-th column
    }

    // Weights = first_component^2
    let weights: Vec<f64> = first_components.iter().map(|&z| z * z).collect();

    // Sort by eigenvalue
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        evals[a]
            .partial_cmp(&evals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_evals: Vec<f64> = idx.iter().map(|&i| evals[i]).collect();
    let sorted_weights: Vec<f64> = idx.iter().map(|&i| weights[i]).collect();

    Ok((sorted_evals, sorted_weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_values() {
        // H_0(x) = 1
        assert!((evaluate_hermite(0, 2.5).expect("eval failed") - 1.0).abs() < 1e-14);
        // H_1(x) = x
        assert!((evaluate_hermite(1, 2.5).expect("eval failed") - 2.5).abs() < 1e-14);
        // H_2(x) = x^2 - 1
        assert!((evaluate_hermite(2, 2.5).expect("eval failed") - (2.5 * 2.5 - 1.0)).abs() < 1e-14);
        // H_3(x) = x^3 - 3x
        let x = 1.5;
        let expected = x * x * x - 3.0 * x;
        assert!((evaluate_hermite(3, x).expect("eval failed") - expected).abs() < 1e-13);
    }

    #[test]
    fn test_legendre_values() {
        // P_0(x) = 1
        assert!((evaluate_legendre(0, 0.5).expect("eval failed") - 1.0).abs() < 1e-14);
        // P_1(x) = x
        assert!((evaluate_legendre(1, 0.5).expect("eval failed") - 0.5).abs() < 1e-14);
        // P_2(x) = (3x^2 - 1)/2
        let x = 0.5;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert!((evaluate_legendre(2, x).expect("eval failed") - expected).abs() < 1e-14);
    }

    #[test]
    fn test_laguerre_values() {
        // L_0(x) = 1
        assert!((evaluate_laguerre(0, 2.0).expect("eval failed") - 1.0).abs() < 1e-14);
        // L_1(x) = 1 - x
        assert!((evaluate_laguerre(1, 2.0).expect("eval failed") - (-1.0)).abs() < 1e-14);
        // L_2(x) = (x^2 - 4x + 2)/2
        let x = 2.0;
        let expected = (x * x - 4.0 * x + 2.0) / 2.0;
        assert!((evaluate_laguerre(2, x).expect("eval failed") - expected).abs() < 1e-14);
    }

    #[test]
    fn test_multi_index_total_degree() {
        // d=2, p=2: should give C(2+2, 2) = 6 indices
        let indices = generate_multi_indices(2, 2, &TruncationScheme::TotalDegree);
        assert_eq!(indices.len(), 6);
        // All should have sum <= 2
        for idx in &indices {
            assert!(idx.iter().sum::<usize>() <= 2);
        }
    }

    #[test]
    fn test_multi_index_tensor() {
        // d=2, p=2: should give (2+1)^2 = 9 indices
        let indices = generate_multi_indices(2, 2, &TruncationScheme::Tensor);
        assert_eq!(indices.len(), 9);
    }

    #[test]
    fn test_multi_index_total_degree_3d() {
        // d=3, p=2: C(3+2, 2) = C(5,2) = 10
        let indices = generate_multi_indices(3, 2, &TruncationScheme::TotalDegree);
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_gauss_legendre_integrates_polynomial() {
        // 3-point Gauss-Legendre should compute E[X^4] for X ~ Uniform[-1,1]
        // E[X^4] = (1/2) integral_{-1}^{1} x^4 dx = (1/2)(2/5) = 1/5
        let (nodes, weights) = gauss_legendre_nodes_weights(3).expect("quadrature failed");
        let expectation: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x.powi(4))
            .sum();
        assert!(
            (expectation - 1.0 / 5.0).abs() < 1e-12,
            "Got {expectation}, expected {}",
            1.0 / 5.0
        );
    }

    #[test]
    fn test_gauss_hermite_integrates_polynomial() {
        // 3-point Gauss-Hermite should integrate x^2 * exp(-x^2/2)/sqrt(2pi) = 1 (variance)
        let (nodes, weights) = gauss_hermite(3).expect("quadrature failed");
        let integral: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x * x)
            .sum();
        // E[X^2] for standard normal = 1
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "Got {integral}, expected 1.0"
        );
    }

    #[test]
    fn test_basis_norm_squared() {
        // Hermite: ||H_3||^2 = 3! = 6
        assert!((basis_norm_squared_1d(&PolynomialBasis::Hermite, 3) - 6.0).abs() < 1e-14);
        // Legendre: ||P_2||^2 = 1/5
        assert!((basis_norm_squared_1d(&PolynomialBasis::Legendre, 2) - 1.0 / 5.0).abs() < 1e-14);
        // Laguerre: ||L_n||^2 = 1
        assert!((basis_norm_squared_1d(&PolynomialBasis::Laguerre, 5) - 1.0).abs() < 1e-14);
    }
}
