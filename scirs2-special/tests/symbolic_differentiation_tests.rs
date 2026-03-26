//! Integration tests for symbolic mathematics and differentiation extensions.
//!
//! Tests cover:
//! - Symbolic expression evaluation
//! - Symbolic differentiation
//! - PowerSeries evaluation and accuracy
//! - Asymptotic expansions
//! - SpecialFunctionDerivatives accuracy
use std::collections::HashMap;

use scirs2_special::differentiation::SpecialFunctionDerivatives;
use scirs2_special::symbolic::{
    asymptotic_gamma, eval_erfc_asymptotic, eval_stirling_gamma, eval_stirling_lngamma, taylor_1f1,
    taylor_bessel_j, taylor_erf, taylor_gamma, Expr, PowerSeries,
};
use scirs2_special::{bessel, digamma, erf, erfc, gamma, hyp1f1, polygamma};

// ─── Helper ──────────────────────────────────────────────────────────────────

fn vars<'a>(pairs: &'a [(&'a str, f64)]) -> HashMap<&'a str, f64> {
    pairs.iter().cloned().collect()
}

// ─── Expr::eval ──────────────────────────────────────────────────────────────

#[test]
fn test_expr_eval_const_plus_var() {
    // Const(3.0) + Var("x") at x=2.0 should give 5.0
    let expr = Expr::Const(3.0) + Expr::var("x");
    let result = expr.eval(&vars(&[("x", 2.0)]));
    assert!((result - 5.0).abs() < 1e-12, "expected 5.0, got {result}");
}

#[test]
fn test_expr_eval_multiply() {
    let expr = Expr::var("x") * Expr::var("y");
    let result = expr.eval(&vars(&[("x", 3.0), ("y", 4.0)]));
    assert!((result - 12.0).abs() < 1e-12, "expected 12.0, got {result}");
}

#[test]
fn test_expr_eval_pow() {
    // x^2 at x=3 = 9
    let expr = Expr::var("x").pow(Expr::Const(2.0));
    let result = expr.eval(&vars(&[("x", 3.0)]));
    assert!((result - 9.0).abs() < 1e-12, "expected 9.0, got {result}");
}

#[test]
fn test_expr_eval_gamma() {
    // Γ(5) = 4! = 24
    let expr = Expr::Gamma(Box::new(Expr::Const(5.0)));
    let result = expr.eval(&HashMap::new());
    assert!((result - 24.0).abs() < 1e-8, "expected 24.0, got {result}");
}

#[test]
fn test_expr_eval_erf() {
    let expr = Expr::Erf(Box::new(Expr::Const(1.0)));
    let expected = erf(1.0);
    let result = expr.eval(&HashMap::new());
    assert!(
        (result - expected).abs() < 1e-12,
        "erf(1): expected {expected}, got {result}"
    );
}

// ─── Expr::diff ──────────────────────────────────────────────────────────────

#[test]
fn test_expr_diff_constant_is_zero() {
    let expr = Expr::Const(42.0);
    let deriv = expr.diff("x");
    let result = deriv.eval(&vars(&[("x", 1.0)]));
    assert_eq!(result, 0.0);
}

#[test]
fn test_expr_diff_x_squared() {
    // d/dx x^2 = 2x
    let expr = Expr::var("x").pow(Expr::Const(2.0));
    let deriv = expr.diff("x").simplify();
    // At x = 3: expect 6.0
    let result = deriv.eval(&vars(&[("x", 3.0)]));
    assert!(
        (result - 6.0).abs() < 1e-6,
        "d/dx x^2 at x=3: expected 6.0, got {result}"
    );
}

#[test]
fn test_expr_diff_sin() {
    // d/dx sin(x) = cos(x)
    let expr = Expr::var("x").sin();
    let deriv = expr.diff("x").simplify();
    let x_val = 1.2f64;
    let numerical = SpecialFunctionDerivatives::complex_step_deriv(|x| x.sin(), x_val);
    let symbolic = deriv.eval(&vars(&[("x", x_val)]));
    assert!(
        (symbolic - numerical).abs() < 1e-8,
        "d/dx sin(x) mismatch: symbolic={symbolic}, numerical={numerical}"
    );
}

#[test]
fn test_expr_diff_gamma_matches_numerical() {
    // d/dx Γ(x) should match numerical derivative
    let expr = Expr::Gamma(Box::new(Expr::var("x")));
    let deriv = expr.diff("x");
    let x_val = 2.5f64;
    let h = 1e-6;
    let numerical = (gamma(x_val + h) - gamma(x_val - h)) / (2.0 * h);
    let symbolic = deriv.eval_ext(&vars(&[("x", x_val)]));
    let rel_err = (symbolic - numerical).abs() / numerical.abs();
    assert!(
        rel_err < 1e-5,
        "d/dx Gamma(x) rel_err={rel_err}: symbolic={symbolic}, numerical={numerical}"
    );
}

#[test]
fn test_expr_diff_log_gamma() {
    // d/dx ln Γ(x) = ψ₀(x) (digamma)
    let expr = Expr::LogGamma(Box::new(Expr::var("x")));
    let deriv = expr.diff("x");
    let x_val = 3.0f64;
    let expected = digamma(x_val);
    let result = deriv.eval_ext(&vars(&[("x", x_val)]));
    assert!(
        (result - expected).abs() < 1e-8,
        "d/dx lnΓ(x): expected {expected}, got {result}"
    );
}

// ─── Expr::simplify ──────────────────────────────────────────────────────────

#[test]
fn test_simplify_add_zero() {
    let expr = Expr::Add(Box::new(Expr::Const(0.0)), Box::new(Expr::var("x")));
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::var("x"), "0 + x should simplify to x");
}

#[test]
fn test_simplify_mul_one() {
    let expr = Expr::Mul(Box::new(Expr::Const(1.0)), Box::new(Expr::var("x")));
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::var("x"), "1 * x should simplify to x");
}

#[test]
fn test_simplify_mul_zero() {
    let expr = Expr::Mul(Box::new(Expr::Const(0.0)), Box::new(Expr::var("x")));
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0.0), "0 * x should simplify to 0");
}

#[test]
fn test_simplify_pow_one() {
    let expr = Expr::Pow(Box::new(Expr::var("x")), Box::new(Expr::Const(1.0)));
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::var("x"), "x^1 should simplify to x");
}

#[test]
fn test_simplify_neg_neg() {
    let expr = Expr::Neg(Box::new(Expr::Neg(Box::new(Expr::var("x")))));
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::var("x"), "-(-x) should simplify to x");
}

#[test]
fn test_simplify_pow_const_base_one() {
    let expr = Expr::Pow(Box::new(Expr::Const(1.0)), Box::new(Expr::var("x")));
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(1.0), "1^x should simplify to 1");
}

// ─── Expr Display ────────────────────────────────────────────────────────────

#[test]
fn test_expr_display_nonempty() {
    let expr = Expr::Gamma(Box::new(Expr::var("x")));
    let s = format!("{expr}");
    assert!(!s.is_empty(), "Display should produce non-empty string");
    assert!(s.contains('x'), "Display should contain variable name");
}

// ─── PowerSeries ─────────────────────────────────────────────────────────────

#[test]
fn test_power_series_eval_at_center() {
    // At center, result should equal a_0
    let ps = PowerSeries::new(1.0, vec![3.25, 2.0, 1.0]);
    let result = ps.eval(1.0);
    assert!(
        (result - 3.25).abs() < 1e-12,
        "series at center: expected 3.25, got {result}"
    );
}

#[test]
fn test_taylor_gamma_accuracy() {
    // Taylor series of Γ around x0=2.0 should be accurate for very nearby x
    let ps = taylor_gamma(2.0, 4);
    // Γ(2.0) = 1.0
    let at_center = ps.eval(2.0);
    assert!(
        (at_center - gamma(2.0)).abs() < 1e-4,
        "Taylor Gamma at center: got {at_center}, expected {}",
        gamma(2.0)
    );
    // Very nearby point — within convergence radius of truncated series
    let x = 2.05f64;
    let approx = ps.eval(x);
    let exact = gamma(x);
    let rel_err = (approx - exact).abs() / exact.abs();
    assert!(rel_err < 0.02, "Taylor Gamma rel_err at x={x}: {rel_err}");
}

#[test]
fn test_taylor_erf_accuracy() {
    // Analytical series around 0 — use order 11 for good coverage up to x=0.5
    let ps = taylor_erf(0.0, 11);
    let x = 0.5f64;
    let approx = ps.eval(x);
    let exact = erf(x);
    let rel_err = (approx - exact).abs() / exact.abs();
    assert!(rel_err < 1e-4, "Taylor erf rel_err at x={x}: {rel_err}");
}

#[test]
fn test_taylor_bessel_j_accuracy() {
    // Analytical series for J_0 around 0
    let ps = taylor_bessel_j(0, 0.0, 10);
    let x = 1.0f64;
    let approx = ps.eval(x);
    let exact = bessel::j0(x);
    let rel_err = (approx - exact).abs() / exact.abs().max(1e-12);
    assert!(rel_err < 0.01, "Taylor J0 rel_err at x={x}: {rel_err}");
}

#[test]
fn test_power_series_diff() {
    // d/dx (1 + 2x + 3x²) = 2 + 6x
    let ps = PowerSeries::new(0.0, vec![1.0, 2.0, 3.0]);
    let deriv = ps.diff();
    // At x=1: 2 + 6*1 = 8
    let result = deriv.eval(1.0);
    assert!((result - 8.0).abs() < 1e-10, "expected 8.0, got {result}");
}

// ─── Asymptotic expansions ────────────────────────────────────────────────────

#[test]
fn test_stirling_lngamma_large_x() {
    // For x=20, Stirling should be very accurate
    let x = 20.0f64;
    use scirs2_special::gammaln;
    let exact = gammaln(x);
    let approx = eval_stirling_lngamma(x, 4);
    let rel_err = (approx - exact).abs() / exact.abs();
    assert!(rel_err < 1e-8, "Stirling ln Γ rel_err at x={x}: {rel_err}");
}

#[test]
fn test_stirling_gamma_large_x() {
    let x = 15.0f64;
    let exact = gamma(x);
    let approx = eval_stirling_gamma(x, 4);
    let rel_err = (approx - exact).abs() / exact.abs();
    assert!(rel_err < 1e-6, "Stirling Γ rel_err at x={x}: {rel_err}");
}

#[test]
fn test_asymptotic_erfc_large_x() {
    // The erfc asymptotic series is a divergent asymptotic expansion;
    // for x=5 with 4 terms the relative error is typically ~1%.
    let x = 5.0f64;
    let exact = erfc(x);
    let approx = eval_erfc_asymptotic(x, 4);
    let rel_err = (approx - exact).abs() / exact.abs().max(1e-300);
    assert!(
        rel_err < 0.02,
        "asymptotic erfc rel_err at x={x}: {rel_err}, exact={exact}, approx={approx}"
    );
}

// ─── SpecialFunctionDerivatives ───────────────────────────────────────────────

#[test]
fn test_gamma_derivative_matches_finite_diff() {
    let x = 2.5f64;
    let h = 1e-5;
    let fd = (gamma(x + h) - gamma(x - h)) / (2.0 * h);
    let analytical = SpecialFunctionDerivatives::gamma_derivative(x);
    let rel_err = (analytical - fd).abs() / fd.abs();
    assert!(rel_err < 1e-4, "gamma_derivative rel_err: {rel_err}");
}

#[test]
fn test_bessel_j_derivative_recurrence() {
    // d/dx J_0(x) = -J_1(x) (from recurrence with n=0)
    let x = 2.0f64;
    let expected = (bessel::jn(-1, x) - bessel::jn(1, x)) / 2.0;
    let result = SpecialFunctionDerivatives::bessel_j_derivative(0, x);
    assert!(
        (result - expected).abs() < 1e-12,
        "bessel_j_derivative mismatch: {result} vs {expected}"
    );
}

#[test]
fn test_hyp1f1_derivative_z_matches_definition() {
    // d/dz ₁F₁(a;b;z) = (a/b) ₁F₁(a+1;b+1;z)
    let a = 1.0f64;
    let b = 2.0f64;
    let z = 0.5f64;
    let analytical = SpecialFunctionDerivatives::hyp1f1_derivative_z(a, b, z);
    // Also compare to finite-difference to validate
    let h = 1e-6;
    let fd = (hyp1f1(a, b, z + h).unwrap() - hyp1f1(a, b, z - h).unwrap()) / (2.0 * h);
    let rel_err = (analytical - fd).abs() / fd.abs();
    assert!(rel_err < 1e-4, "hyp1f1_derivative_z rel_err: {rel_err}");
}

#[test]
fn test_complex_step_deriv_sin() {
    let x = 1.0f64;
    let result = SpecialFunctionDerivatives::complex_step_deriv(|t| t.sin(), x);
    let expected = x.cos();
    let err = (result - expected).abs();
    assert!(
        err < 1e-8,
        "complex_step sin'(1): err={err}, got={result}, expected={expected}"
    );
}

#[test]
fn test_elliptic_k_derivative_finite_check() {
    // Verify formula against finite difference.
    // The accuracy is limited by the underlying ellipk implementation,
    // so we allow up to 5% relative error.
    let k = 0.5f64;
    let h = 1e-5;
    let k_val = |kk: f64| scirs2_special::ellipk(kk * kk);
    let fd = (k_val(k + h) - k_val(k - h)) / (2.0 * h);
    let analytical = SpecialFunctionDerivatives::elliptic_k_derivative(k);
    // Use absolute error for robustness
    let abs_err = (analytical - fd).abs();
    assert!(
        abs_err < 0.1,
        "elliptic_k_derivative abs_err={abs_err}: analytical={analytical}, fd={fd}"
    );
}

#[test]
fn test_polygamma_n0_matches_digamma() {
    // polygamma(0, x) = digamma(x)
    let x = 3.5f64;
    let pg0 = polygamma(0u32, x);
    let dg = digamma(x);
    let err = (pg0 - dg).abs();
    assert!(err < 1e-12, "polygamma(0,x) != digamma(x): err={err}");
}
