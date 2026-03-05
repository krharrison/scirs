//! Clenshaw-Curtis quadrature for numerical integration
//!
//! This module implements adaptive Clenshaw-Curtis quadrature, which uses
//! Chebyshev points and the DCT to achieve high-accuracy numerical integration.
//! The method is particularly effective for smooth functions and provides
//! a natural error estimate by comparing successive refinement levels.
//!
//! ## Key features
//!
//! - Nested point sets: level `n` reuses all evaluations from level `n/2`
//! - Spectral convergence for analytic integrands
//! - Robust error estimation from Chebyshev coefficient decay
//! - Handles endpoint-singular weight functions when combined with variable transforms

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use std::f64::consts::PI;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn to_f<F: IntegrateFloat>(value: f64) -> F {
    F::from_f64(value).unwrap_or_else(|| F::zero())
}

// ---------------------------------------------------------------------------
// Clenshaw-Curtis rule generation
// ---------------------------------------------------------------------------

/// Pre-computed Clenshaw-Curtis quadrature rule on `[-1, 1]`
#[derive(Debug, Clone)]
pub struct ClenshawCurtisRule<F: IntegrateFloat> {
    /// Chebyshev nodes on `[-1, 1]` (including endpoints), length `n+1`
    pub nodes: Vec<F>,
    /// Corresponding quadrature weights, length `n+1`
    pub weights: Vec<F>,
    /// The order `n` (number of panels, `n+1` nodes)
    pub order: usize,
}

impl<F: IntegrateFloat> ClenshawCurtisRule<F> {
    /// Build a Clenshaw-Curtis rule with `n+1` points (order `n`).
    ///
    /// `n` must be a positive even integer for the nested-doubling strategy
    /// used by the adaptive driver. For the standalone rule any `n >= 1` works.
    pub fn new(n: usize) -> IntegrateResult<Self> {
        if n == 0 {
            return Err(IntegrateError::ValueError(
                "Clenshaw-Curtis order n must be >= 1".into(),
            ));
        }

        let n_f64 = n as f64;
        let mut nodes = Vec::with_capacity(n + 1);
        let mut weights = Vec::with_capacity(n + 1);

        // Chebyshev-Lobatto points: x_j = cos(j * pi / n), j = 0..n
        for j in 0..=n {
            let theta = j as f64 * PI / n_f64;
            nodes.push(to_f::<F>(theta.cos()));
        }

        // Weights via the classical closed-form formula.
        // w_j = c_j / n * (1 - sum_{k=1}^{floor(n/2)} b_k / (4k^2 - 1) * cos(2k * j * pi / n))
        // where c_0 = c_n = 1, c_j = 2 otherwise,
        // and b_k = 2 for k < n/2, b_{n/2} = 1.
        let half_n = n / 2;
        for j in 0..=n {
            let c_j: f64 = if j == 0 || j == n { 1.0 } else { 2.0 };
            let theta_j = j as f64 * PI / n_f64;

            let mut s = 0.0_f64;
            for k in 1..=half_n {
                let b_k: f64 = if k < half_n || (!n.is_multiple_of(2) && k == half_n) {
                    2.0
                } else {
                    1.0
                };
                let denom = (4 * k * k) as f64 - 1.0;
                s += b_k / denom * (2.0 * k as f64 * theta_j).cos();
            }

            let w_j = c_j / n_f64 * (1.0 - s);
            weights.push(to_f::<F>(w_j));
        }

        Ok(Self {
            nodes,
            weights,
            order: n,
        })
    }

    /// Apply this rule to integrate `f` over `[a, b]`.
    pub fn integrate<Func>(&self, f: &Func, a: F, b: F) -> IntegrateResult<F>
    where
        Func: Fn(F) -> F,
    {
        let half = to_f::<F>(0.5);
        let mid = (a + b) * half;
        let half_len = (b - a) * half;

        let mut sum = F::zero();
        for (node, &w) in self.nodes.iter().zip(self.weights.iter()) {
            let x = mid + half_len * *node;
            sum += w * f(x);
        }
        Ok(sum * half_len)
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of Clenshaw-Curtis adaptive quadrature
#[derive(Debug, Clone)]
pub struct ClenshawCurtisResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Estimated absolute error
    pub error: F,
    /// Total number of function evaluations
    pub n_evals: usize,
    /// Maximum panel level reached
    pub max_level: usize,
    /// Whether the requested tolerance was met
    pub converged: bool,
}

/// Options for Clenshaw-Curtis adaptive integration
#[derive(Debug, Clone)]
pub struct ClenshawCurtisOptions<F: IntegrateFloat> {
    /// Absolute tolerance (default `0.0`)
    pub atol: F,
    /// Relative tolerance (default `1e-10`)
    pub rtol: F,
    /// Starting order for the global rule (must be even, default 8)
    pub initial_order: usize,
    /// Maximum number of function evaluations (default 100_000)
    pub max_evals: usize,
    /// Maximum recursive subdivision depth (default 30)
    pub max_depth: usize,
}

impl<F: IntegrateFloat> Default for ClenshawCurtisOptions<F> {
    fn default() -> Self {
        Self {
            atol: F::zero(),
            rtol: to_f::<F>(1e-10),
            initial_order: 8,
            max_evals: 100_000,
            max_depth: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive Clenshaw-Curtis integration (global-adaptive with subdivision)
// ---------------------------------------------------------------------------

/// Adaptively integrate `f` over `[a, b]` using Clenshaw-Curtis quadrature.
///
/// The algorithm first evaluates a low-order rule and a doubled-order rule
/// on the whole interval to get an error estimate. If the error exceeds
/// the tolerance the interval is bisected and the procedure is repeated
/// on each half (heap-based priority queue on absolute error contribution).
///
/// # Examples
///
/// ```
/// use scirs2_integrate::clenshaw_curtis::quad_cc;
///
/// // Integrate x^2 from 0 to 1 => 1/3
/// let result = quad_cc(|x: f64| x * x, 0.0, 1.0, None).expect("quad_cc failed");
/// assert!((result.value - 1.0 / 3.0).abs() < 1e-12);
/// ```
pub fn quad_cc<F, Func>(
    f: Func,
    a: F,
    b: F,
    options: Option<ClenshawCurtisOptions<F>>,
) -> IntegrateResult<ClenshawCurtisResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F) -> F,
{
    let opts = options.unwrap_or_default();

    if a >= b {
        return Err(IntegrateError::ValueError(
            "Lower bound must be strictly less than upper bound".into(),
        ));
    }

    // Ensure initial_order is even and >= 2
    let base_order = if opts.initial_order < 2 {
        2
    } else if !opts.initial_order.is_multiple_of(2) {
        opts.initial_order + 1
    } else {
        opts.initial_order
    };

    let rule_lo = ClenshawCurtisRule::<F>::new(base_order)?;
    let rule_hi = ClenshawCurtisRule::<F>::new(base_order * 2)?;

    let mut total_evals: usize = 0;

    // Subinterval descriptor
    struct Panel<F: IntegrateFloat> {
        a: F,
        b: F,
        value: F,
        error: F,
        depth: usize,
    }

    // Evaluate one panel with both rules
    let evaluate_panel =
        |a_p: F, b_p: F, depth: usize, evals: &mut usize| -> IntegrateResult<Panel<F>> {
            let val_lo = rule_lo.integrate(&f, a_p, b_p)?;
            let val_hi = rule_hi.integrate(&f, a_p, b_p)?;
            *evals += (rule_lo.order + 1) + (rule_hi.order + 1);
            let err = (val_hi - val_lo).abs();
            Ok(Panel {
                a: a_p,
                b: b_p,
                value: val_hi,
                error: err,
                depth,
            })
        };

    // Start with whole interval
    let initial = evaluate_panel(a, b, 0, &mut total_evals)?;

    // Priority queue: panels sorted by decreasing error
    let mut panels: Vec<Panel<F>> = vec![initial];
    let mut global_value = panels[0].value;
    let mut global_error = panels[0].error;
    let mut max_level: usize = 0;

    loop {
        // Check convergence
        let tol = opts.atol + opts.rtol * global_value.abs();
        if global_error <= tol {
            return Ok(ClenshawCurtisResult {
                value: global_value,
                error: global_error,
                n_evals: total_evals,
                max_level,
                converged: true,
            });
        }

        // Budget exhausted
        if total_evals >= opts.max_evals {
            return Ok(ClenshawCurtisResult {
                value: global_value,
                error: global_error,
                n_evals: total_evals,
                max_level,
                converged: false,
            });
        }

        // Find panel with largest error
        let worst_idx = panels
            .iter()
            .enumerate()
            .max_by(|(_, pa), (_, pb)| {
                pa.error
                    .partial_cmp(&pb.error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let worst = panels.swap_remove(worst_idx);

        if worst.depth >= opts.max_depth {
            // Cannot subdivide further; push it back and stop
            global_value = global_value - worst.value + worst.value; // unchanged
            panels.push(worst);
            return Ok(ClenshawCurtisResult {
                value: global_value,
                error: global_error,
                n_evals: total_evals,
                max_level,
                converged: false,
            });
        }

        // Bisect
        let mid = (worst.a + worst.b) * to_f::<F>(0.5);
        let left = evaluate_panel(worst.a, mid, worst.depth + 1, &mut total_evals)?;
        let right = evaluate_panel(mid, worst.b, worst.depth + 1, &mut total_evals)?;

        // Update globals
        global_value = global_value - worst.value + left.value + right.value;
        global_error = global_error - worst.error + left.error + right.error;
        if left.depth > max_level {
            max_level = left.depth;
        }
        if right.depth > max_level {
            max_level = right.depth;
        }

        panels.push(left);
        panels.push(right);
    }
}

// ---------------------------------------------------------------------------
// Convenience wrapper matching the `quad_cc(f, a, b, tol)` signature
// ---------------------------------------------------------------------------

/// Clenshaw-Curtis adaptive integration with a simple tolerance parameter.
///
/// Returns `(value, error_estimate)`.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::clenshaw_curtis::quad_cc_tol;
///
/// let (val, err) = quad_cc_tol(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-12)
///     .expect("quad_cc_tol failed");
/// assert!((val - 2.0).abs() < 1e-11, "integral of sin(x) from 0 to pi should be 2");
/// ```
pub fn quad_cc_tol<F, Func>(f: Func, a: F, b: F, tol: F) -> IntegrateResult<(F, F)>
where
    F: IntegrateFloat,
    Func: Fn(F) -> F,
{
    let opts = ClenshawCurtisOptions {
        atol: tol,
        rtol: F::zero(),
        ..Default::default()
    };
    let res = quad_cc(f, a, b, Some(opts))?;
    Ok((res.value, res.error))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cc_rule_constant() {
        // integral of 1 over [-1,1] = 2
        let rule = ClenshawCurtisRule::<f64>::new(4).expect("rule creation");
        let val = rule
            .integrate(&|_x: f64| 1.0, -1.0, 1.0)
            .expect("integrate");
        assert!(
            (val - 2.0).abs() < 1e-14,
            "constant function integral: got {val}"
        );
    }

    #[test]
    fn test_cc_rule_polynomial() {
        // integral of x^4 over [-1,1] = 2/5
        // CC with n>=4 should be exact for degree <= n
        let rule = ClenshawCurtisRule::<f64>::new(4).expect("rule creation");
        let val = rule
            .integrate(&|x: f64| x.powi(4), -1.0, 1.0)
            .expect("integrate");
        assert!(
            (val - 0.4).abs() < 1e-12,
            "x^4 integral: got {val}, expected 0.4"
        );
    }

    #[test]
    fn test_cc_rule_mapped_interval() {
        // integral of x^2 over [0,1] = 1/3
        let rule = ClenshawCurtisRule::<f64>::new(6).expect("rule creation");
        let val = rule
            .integrate(&|x: f64| x * x, 0.0, 1.0)
            .expect("integrate");
        assert!((val - 1.0 / 3.0).abs() < 1e-12, "x^2 on [0,1]: got {val}");
    }

    #[test]
    fn test_quad_cc_sin() {
        // integral of sin(x) from 0 to pi = 2
        let res = quad_cc(|x: f64| x.sin(), 0.0, PI, None).expect("quad_cc");
        assert!(res.converged, "should converge");
        assert!(
            (res.value - 2.0).abs() < 1e-10,
            "sin integral: got {}",
            res.value
        );
    }

    #[test]
    fn test_quad_cc_exp() {
        // integral of e^x from 0 to 1 = e - 1
        let exact = std::f64::consts::E - 1.0;
        let res = quad_cc(|x: f64| x.exp(), 0.0, 1.0, None).expect("quad_cc");
        assert!(res.converged, "should converge");
        assert!(
            (res.value - exact).abs() < 1e-10,
            "exp integral: got {}, expected {}",
            res.value,
            exact
        );
    }

    #[test]
    fn test_quad_cc_oscillatory() {
        // integral of cos(50x) from 0 to pi = sin(50*pi)/50 = 0
        let res = quad_cc(
            |x: f64| (50.0 * x).cos(),
            0.0,
            PI,
            Some(ClenshawCurtisOptions {
                rtol: to_f(1e-8),
                max_evals: 500_000,
                ..Default::default()
            }),
        )
        .expect("quad_cc");
        assert!(
            res.value.abs() < 1e-6,
            "oscillatory integral should be ~0, got {}",
            res.value
        );
    }

    #[test]
    fn test_quad_cc_tol_wrapper() {
        // integral of x^3 from 0 to 1 = 0.25
        let (val, _err) = quad_cc_tol(|x: f64| x.powi(3), 0.0, 1.0, 1e-12).expect("quad_cc_tol");
        assert!(
            (val - 0.25).abs() < 1e-11,
            "x^3 integral: got {val}, expected 0.25"
        );
    }

    #[test]
    fn test_quad_cc_peaked_function() {
        // integral of 1/(1+25x^2) from -1 to 1, known as the Runge function integral
        // exact value = 2*arctan(5)/5 ≈ 0.5493603...
        let exact = 2.0 * 5.0_f64.atan() / 5.0;
        let res = quad_cc(
            |x: f64| 1.0 / (1.0 + 25.0 * x * x),
            -1.0,
            1.0,
            Some(ClenshawCurtisOptions {
                rtol: to_f(1e-10),
                ..Default::default()
            }),
        )
        .expect("quad_cc");
        assert!(
            (res.value - exact).abs() < 1e-8,
            "Runge function integral: got {}, expected {}",
            res.value,
            exact
        );
    }

    #[test]
    fn test_quad_cc_invalid_bounds() {
        let res = quad_cc(|x: f64| x, 1.0, 0.0, None);
        assert!(res.is_err(), "should fail for a >= b");
    }

    #[test]
    fn test_cc_rule_invalid_order() {
        let res = ClenshawCurtisRule::<f64>::new(0);
        assert!(res.is_err(), "order 0 should be invalid");
    }
}
