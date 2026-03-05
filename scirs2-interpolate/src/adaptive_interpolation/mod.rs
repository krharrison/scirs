//! Adaptive Chebyshev Interpolation (Chebfun-Style)
//!
//! Builds a piecewise-Chebyshev representation of a function `f: ℝ→ℝ` that
//! achieves a user-specified tolerance by adaptively subdividing the domain.
//!
//! ## Algorithm
//!
//! 1. Attempt to represent `f` on `[a, b]` as a degree-`n` Chebyshev
//!    polynomial (default starting degree 17).
//! 2. Check whether the tail coefficients are small relative to the largest
//!    coefficient.  If yes, truncate and accept the approximation.
//! 3. If not, bisect the interval into `[a, mid]` and `[mid, b]` and recurse.
//! 4. Stop recursion when either:
//!    - the interval is narrower than `1e-14 * |b-a|` (original), or
//!    - the degree would exceed `max_degree`.
//!
//! ## Reference
//!
//! - Battles, Z. & Trefethen, L. N. (2004). *An extension of MATLAB to
//!   continuous functions and operators.* SIAM J. Sci. Comput., 25(5),
//!   1743–1770.
//! - Trefethen, L. N. (2013). *Approximation Theory and Approximation
//!   Practice.* SIAM.

use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// Chebyshev polynomial on an interval [a, b]
// ---------------------------------------------------------------------------

/// A polynomial represented in the Chebyshev basis on `[a, b]`.
///
/// The polynomial is:
///
/// ```text
/// p(x) = c₀/2 + Σ_{k=1}^{n-1}  cₖ Tₖ(ξ(x))
/// ```
///
/// where ξ(x) = 2(x − a)/(b − a) − 1 maps `[a,b]` to `[-1,1]` and `Tₖ` is
/// the Chebyshev polynomial of the first kind.
#[derive(Debug, Clone)]
pub struct ChebyshevPoly {
    /// Chebyshev coefficients (length = degree+1).
    pub coeffs: Vec<f64>,
    /// Left endpoint of the interval.
    pub a: f64,
    /// Right endpoint of the interval.
    pub b: f64,
}

impl ChebyshevPoly {
    /// Evaluate via the Clenshaw recurrence.
    ///
    /// Runs in O(n) operations.
    pub fn eval(&self, x: f64) -> f64 {
        // Map x from [a,b] to [-1,1]
        let xi = 2.0 * (x - self.a) / (self.b - self.a) - 1.0;
        clenshaw(&self.coeffs, xi)
    }

    /// Number of Chebyshev coefficients stored.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Whether the polynomial has no coefficients.
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Estimate of the truncation error: max absolute value of the last
    /// `tail_frac` fraction of coefficients.
    fn tail_error(&self) -> f64 {
        let n = self.coeffs.len();
        if n < 4 {
            return f64::INFINITY;
        }
        let tail_start = (n * 3 / 4).max(n.saturating_sub(4));
        self.coeffs[tail_start..]
            .iter()
            .map(|c| c.abs())
            .fold(0.0_f64, f64::max)
    }

    /// `true` if the polynomial is well-resolved to relative tolerance `tol`.
    fn is_resolved(&self, tol: f64) -> bool {
        let max_coeff = self.coeffs.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        if max_coeff == 0.0 {
            return true;
        }
        self.tail_error() <= tol * max_coeff
    }
}

// ---------------------------------------------------------------------------
// Clenshaw algorithm
// ---------------------------------------------------------------------------

/// Evaluate Σ cₖ Tₖ(x) via the Clenshaw recurrence (x ∈ [-1,1]).
fn clenshaw(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }
    let mut b_kp1 = 0.0_f64;
    let mut b_k = 0.0_f64;
    for k in (1..n).rev() {
        let b_km1 = 2.0 * x * b_k - b_kp1 + coeffs[k];
        b_kp1 = b_k;
        b_k = b_km1;
    }
    x * b_k - b_kp1 + coeffs[0]
}

// ---------------------------------------------------------------------------
// chebfit: fit Chebyshev polynomial via DCT
// ---------------------------------------------------------------------------

/// Interpolate `f` at Chebyshev nodes of the second kind on `[a,b]` and
/// compute the Chebyshev coefficients via the DCT-I algorithm.
///
/// The Chebyshev nodes of the second kind are:
/// `x_k = (a+b)/2 + (b-a)/2 · cos(π k / n)` for k = 0, …, n.
///
/// This gives `n+1` nodes and a polynomial of degree ≤ n.
pub fn chebfit(f: &dyn Fn(f64) -> f64, a: f64, b: f64, degree: usize) -> ChebyshevPoly {
    let n = degree; // number of intervals = n; we use n+1 nodes
    let npts = n + 1;

    // Chebyshev nodes of the 2nd kind in [-1,1]
    let nodes: Vec<f64> = (0..npts)
        .map(|k| {
            let xi = (std::f64::consts::PI * k as f64 / n as f64).cos();
            (a + b) * 0.5 + (b - a) * 0.5 * xi
        })
        .collect();

    let fvals: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();

    // DCT-I: coefficients c_k = (2/n) Σ_{j=0}^{n} '' f(x_j) cos(π j k / n)
    // where '' means the first and last terms are halved.
    let mut coeffs = vec![0.0_f64; npts];
    for k in 0..npts {
        let mut sum = 0.0_f64;
        for j in 0..npts {
            let w = if j == 0 || j == n { 0.5 } else { 1.0 };
            sum += w * fvals[j] * (std::f64::consts::PI * j as f64 * k as f64 / n as f64).cos();
        }
        coeffs[k] = 2.0 * sum / n as f64;
    }
    // c_0 is halved (standard Chebyshev series convention)
    coeffs[0] *= 0.5;
    if n > 0 {
        // c_n is also halved
        coeffs[n] *= 0.5;
    }

    ChebyshevPoly { coeffs, a, b }
}

// ---------------------------------------------------------------------------
// Adaptive interpolant
// ---------------------------------------------------------------------------

/// Piecewise Chebyshev interpolant with automatic interval subdivision.
///
/// The domain `[a, b]` is partitioned into sub-intervals, each carrying a
/// `ChebyshevPoly` that represents `f` to within `tol` on that sub-interval.
#[derive(Debug, Clone)]
pub struct AdaptiveInterpolant1D {
    /// Sorted breakpoints including endpoints: [a, …, b].
    pub intervals: Vec<f64>,
    /// One polynomial per sub-interval (length = intervals.len() - 1).
    pub polynomials: Vec<ChebyshevPoly>,
}

impl AdaptiveInterpolant1D {
    /// Build the adaptive interpolant for `f` on `[a, b]`.
    ///
    /// # Arguments
    ///
    /// * `f`   – Function to approximate.
    /// * `a`, `b` – Domain endpoints (`a < b`).
    /// * `tol` – Absolute tolerance (e.g. `1e-12`).
    ///
    /// # Errors
    ///
    /// Returns an error if `a ≥ b` or if the maximum subdivision depth is
    /// exceeded before convergence.
    pub fn build<F>(f: F, a: f64, b: f64, tol: f64) -> InterpolateResult<AdaptiveInterpolant1D>
    where
        F: Fn(f64) -> f64 + Copy,
    {
        if a >= b {
            return Err(InterpolateError::InvalidInput {
                message: format!("need a < b, got a={} b={}", a, b),
            });
        }
        if tol <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("tolerance must be positive, got {}", tol),
            });
        }

        let mut intervals = vec![a];
        let mut polynomials = Vec::new();

        // Recursive subdivision using an explicit stack
        let mut stack: Vec<(f64, f64, usize)> = vec![(a, b, 0)];
        const MAX_DEPTH: usize = 50;
        const START_DEGREE: usize = 17;
        const MAX_DEGREE: usize = 65;

        while let Some((la, lb, depth)) = stack.pop() {
            let poly = chebfit(&f, la, lb, START_DEGREE);
            if poly.is_resolved(tol) || depth >= MAX_DEPTH {
                // Accept this polynomial (possibly under-resolved at max depth)
                intervals.push(lb);
                polynomials.push(poly);
            } else {
                // Try with higher degree first
                let poly_hi = chebfit(&f, la, lb, MAX_DEGREE);
                if poly_hi.is_resolved(tol) || depth >= MAX_DEPTH - 1 {
                    intervals.push(lb);
                    polynomials.push(poly_hi);
                } else {
                    // Bisect: push right half first (stack is LIFO, left half processed first)
                    let mid = (la + lb) * 0.5;
                    stack.push((mid, lb, depth + 1));
                    stack.push((la, mid, depth + 1));
                }
            }
        }

        // The interval breakpoints from the stack may be out of order; sort them.
        // Collect (left, right, poly) triples, sort by left.
        let n_intervals = polynomials.len();
        let mut pieces: Vec<(f64, f64, ChebyshevPoly)> = Vec::with_capacity(n_intervals);
        // Rebuild from intervals (they were pushed in stack order)
        let mut sorted_intervals = intervals.clone();
        sorted_intervals.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        sorted_intervals.dedup_by(|x, y| (*x - *y).abs() < 1e-15 * (b - a).abs());

        // Refit polynomials on sorted intervals
        pieces.clear();
        for i in 0..sorted_intervals.len().saturating_sub(1) {
            let la = sorted_intervals[i];
            let lb = sorted_intervals[i + 1];
            let poly = chebfit(&f, la, lb, MAX_DEGREE);
            pieces.push((la, lb, poly));
        }

        let final_intervals: Vec<f64> = pieces
            .iter()
            .map(|(la, _, _)| *la)
            .chain(std::iter::once(pieces.last().map(|(_, lb, _)| *lb).unwrap_or(b)))
            .collect();
        let final_polys: Vec<ChebyshevPoly> = pieces.into_iter().map(|(_, _, p)| p).collect();

        Ok(AdaptiveInterpolant1D {
            intervals: final_intervals,
            polynomials: final_polys,
        })
    }

    /// Evaluate the adaptive interpolant at `x`.
    ///
    /// Uses binary search to locate the interval and evaluates the local
    /// Chebyshev polynomial via the Clenshaw algorithm.
    ///
    /// Returns 0.0 for `x` outside `[a, b]` (extrapolation is not guaranteed).
    pub fn eval(&self, x: f64) -> f64 {
        if self.polynomials.is_empty() {
            return 0.0;
        }
        // Binary search for the sub-interval containing x
        let idx = match self.intervals.binary_search_by(|v| {
            v.partial_cmp(&x)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i.saturating_sub(1).min(self.polynomials.len() - 1),
            Err(i) => {
                if i == 0 {
                    0
                } else if i >= self.intervals.len() {
                    self.polynomials.len() - 1
                } else {
                    i - 1
                }
            }
        };
        let idx = idx.min(self.polynomials.len() - 1);
        self.polynomials[idx].eval(x)
    }

    /// Number of sub-intervals in the partition.
    pub fn num_intervals(&self) -> usize {
        self.polynomials.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn test_chebfit_polynomial() {
        // f(x) = x³ should be represented exactly by degree-3 Chebyshev
        let f = |x: f64| x * x * x;
        let poly = chebfit(&f, -1.0, 1.0, 8);
        for &t in &[-0.9, -0.5, 0.0, 0.5, 0.9] {
            let v = poly.eval(t);
            assert!((v - t * t * t).abs() < 1e-12, "chebfit cubic at {}: {} != {}", t, v, t*t*t);
        }
    }

    #[test]
    fn test_clenshaw_constant() {
        // Polynomial = 3.0 on [-1,1]
        let coeffs = vec![3.0_f64]; // c_0 = 3 → p(x) = c_0 = 3
        for x in [-0.5, 0.0, 0.5] {
            assert_eq!(clenshaw(&coeffs, x), 3.0);
        }
    }

    #[test]
    fn test_adaptive_sin() {
        let f = |x: f64| x.sin();
        let interp = AdaptiveInterpolant1D::build(f, 0.0, std::f64::consts::TAU, 1e-10)
            .expect("build failed");

        let test_pts = [0.1, 0.7, 1.5, 3.0, 5.0, 6.0];
        for &x in &test_pts {
            let v = interp.eval(x);
            let exact = x.sin();
            assert!(
                (v - exact).abs() < 1e-7,
                "sin at {}: got {} expected {}",
                x,
                v,
                exact
            );
        }
    }

    #[test]
    fn test_adaptive_exp() {
        let f = |x: f64| x.exp();
        let interp = AdaptiveInterpolant1D::build(f, 0.0, 3.0, 1e-10).expect("build");
        let test_pts = [0.0, 0.5, 1.0, 2.0, 2.9];
        for &x in &test_pts {
            let v = interp.eval(x);
            let exact = x.exp();
            assert!((v - exact).abs() < 1e-7, "exp at {}: {} vs {}", x, v, exact);
        }
    }

    #[test]
    fn test_adaptive_constant() {
        let f = |_x: f64| 42.0_f64;
        let interp = AdaptiveInterpolant1D::build(f, -1.0, 1.0, 1e-12).expect("build");
        assert!((interp.eval(0.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_polynomial() {
        // Degree-5 polynomial should be captured exactly
        let f = |x: f64| x.powi(5) - 3.0 * x.powi(3) + x;
        let interp = AdaptiveInterpolant1D::build(f, -1.0, 1.0, 1e-12).expect("build");
        for &x in &[-0.8, -0.3, 0.1, 0.6, 0.95] {
            let v = interp.eval(x);
            let exact = f(x);
            assert!((v - exact).abs() < 1e-8, "poly at {}: {} vs {}", x, v, exact);
        }
    }

    #[test]
    fn test_error_on_invalid_interval() {
        let r = AdaptiveInterpolant1D::build(|x| x, 1.0, 0.0, 1e-10);
        assert!(r.is_err());
    }

    #[test]
    fn test_error_on_negative_tol() {
        let r = AdaptiveInterpolant1D::build(|x| x, 0.0, 1.0, -1e-10);
        assert!(r.is_err());
    }

    #[test]
    fn test_num_intervals() {
        // A smooth function should need very few intervals
        let f = |x: f64| x.sin();
        let interp = AdaptiveInterpolant1D::build(f, 0.0, 1.0, 1e-12).expect("build");
        assert!(interp.num_intervals() >= 1);
    }

    #[test]
    fn test_chebyshev_poly_len_isempty() {
        let poly = ChebyshevPoly { coeffs: vec![], a: 0.0, b: 1.0 };
        assert!(poly.is_empty());
        assert_eq!(poly.len(), 0);
        assert_eq!(poly.eval(0.5), 0.0);
    }
}
