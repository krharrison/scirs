//! Optimization test functions for benchmarking algorithms.
//!
//! # Single-objective functions
//!
//! | Struct | Description |
//! |---|---|
//! | [`AckleyFn`] | Ackley (multimodal, near-flat outer region) |
//! | [`RastriginFn`] | Rastrigin (highly multimodal) |
//! | [`GriewankFn`] | Griewank (product-cosine term) |
//! | [`LevyFn`] | Lévy (valley-shaped) |
//! | [`BraninFn`] | Branin–Hoo (2-D, 3 global minima) |
//! | [`HartmannFn`] | Hartmann-6 (6-D) |
//! | [`RosenbrockFn`] | Rosenbrock banana (narrow curved valley) |
//! | [`SixHumpCamelFn`] | Six-Hump Camel (2 global minima) |
//! | [`Bukin6Fn`] | Bukin N.6 (ridge) |
//! | [`CrossInTrayFn`] | Cross-in-tray (4 global minima) |
//! | [`EggholderFn`] | Eggholder (highly multimodal) |
//!
//! # Multi-objective functions (ZDT family)
//!
//! | Struct | Description |
//! |---|---|
//! | [`ZDT1`] | Convex Pareto front |
//! | [`ZDT2`] | Non-convex Pareto front |
//! | [`ZDT3`] | Disconnected Pareto front |
//! | [`ZDT4`] | Multimodal with many local fronts |
//! | [`ZDT6`] | Non-convex, biased |
//!
//! # Multi-objective functions (DTLZ)
//!
//! | Struct | Description |
//! |---|---|
//! | [`DTLZ1`] | 3-objective, linear Pareto front (hyperplane) |
//!
//! # Multi-objective functions (WFG)
//!
//! | Struct | Description |
//! |---|---|
//! | [`WFG1`] | Convex, mixed Pareto shape |
//! | [`WFG2`] | Disconnected convex Pareto front |
//!
//! All structs implement the [`TestFunction`] trait.

use crate::error::{DatasetsError, Result};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Common trait
// ─────────────────────────────────────────────────────────────────────────────

/// Common interface for optimization test functions.
pub trait TestFunction {
    /// Evaluate the function at point `x`.
    fn evaluate(&self, x: &[f64]) -> Result<f64>;

    /// Return the recommended search bounds `[(lower, upper); n_dims]`.
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// Return the known global minimum `(x*, f*)`, if available.
    fn global_min(&self) -> Option<(Vec<f64>, f64)>;
}

/// Multi-objective variant: evaluate returns `(f1, f2)`.
pub trait MultiObjectiveFunction {
    /// Number of objectives.
    fn n_objectives(&self) -> usize;

    /// Evaluate all objectives at `x`.
    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>>;

    /// Return search bounds.
    fn bounds(&self) -> Vec<(f64, f64)>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// Ackley function (n-dimensional).
///
/// Global minimum: `f(0,…,0) = 0`.
///
/// ```text
/// f(x) = -a·exp(-b·√(Σxᵢ²/n)) - exp(Σcos(c·xᵢ)/n) + a + exp(1)
/// ```
pub struct AckleyFn {
    /// Dimensionality.
    pub n_dims: usize,
    /// Parameter a (default 20).
    pub a: f64,
    /// Parameter b (default 0.2).
    pub b: f64,
    /// Parameter c (default 2π).
    pub c: f64,
}

impl Default for AckleyFn {
    fn default() -> Self {
        Self {
            n_dims: 2,
            a: 20.0,
            b: 0.2,
            c: 2.0 * PI,
        }
    }
}

impl TestFunction for AckleyFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.n_dims {
            return Err(DatasetsError::InvalidFormat(format!(
                "AckleyFn: expected {} dims, got {}",
                self.n_dims,
                x.len()
            )));
        }
        let n = x.len() as f64;
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let sum_cos = x.iter().map(|xi| (self.c * xi).cos()).sum::<f64>();
        let t1 = -self.a * (-self.b * (sum_sq / n).sqrt()).exp();
        let t2 = -(sum_cos / n).exp();
        Ok(t1 + t2 + self.a + std::f64::consts::E)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-32.768, 32.768); self.n_dims]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![0.0; self.n_dims], 0.0))
    }
}

/// Rastrigin function (n-dimensional).
///
/// Global minimum: `f(0,…,0) = 0`.
///
/// ```text
/// f(x) = 10n + Σ(xᵢ² - 10·cos(2π·xᵢ))
/// ```
pub struct RastriginFn {
    /// Dimensionality.
    pub n_dims: usize,
}

impl Default for RastriginFn {
    fn default() -> Self {
        Self { n_dims: 2 }
    }
}

impl TestFunction for RastriginFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.n_dims {
            return Err(DatasetsError::InvalidFormat(format!(
                "RastriginFn: expected {} dims, got {}",
                self.n_dims,
                x.len()
            )));
        }
        let n = x.len() as f64;
        let sum = x
            .iter()
            .map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>();
        Ok(10.0 * n + sum)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-5.12, 5.12); self.n_dims]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![0.0; self.n_dims], 0.0))
    }
}

/// Griewank function (n-dimensional).
///
/// Global minimum: `f(0,…,0) = 0`.
pub struct GriewankFn {
    /// Dimensionality.
    pub n_dims: usize,
}

impl Default for GriewankFn {
    fn default() -> Self {
        Self { n_dims: 2 }
    }
}

impl TestFunction for GriewankFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.n_dims {
            return Err(DatasetsError::InvalidFormat(format!(
                "GriewankFn: expected {} dims, got {}",
                self.n_dims,
                x.len()
            )));
        }
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>() / 4000.0;
        let prod_cos = x
            .iter()
            .enumerate()
            .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product::<f64>();
        Ok(1.0 + sum_sq - prod_cos)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-600.0, 600.0); self.n_dims]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![0.0; self.n_dims], 0.0))
    }
}

/// Lévy function (n-dimensional).
///
/// Global minimum: `f(1,…,1) = 0`.
pub struct LevyFn {
    /// Dimensionality.
    pub n_dims: usize,
}

impl Default for LevyFn {
    fn default() -> Self {
        Self { n_dims: 2 }
    }
}

impl TestFunction for LevyFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.n_dims {
            return Err(DatasetsError::InvalidFormat(format!(
                "LevyFn: expected {} dims, got {}",
                self.n_dims,
                x.len()
            )));
        }
        let w: Vec<f64> = x.iter().map(|xi| 1.0 + (xi - 1.0) / 4.0).collect();
        let n = w.len();
        let term1 = (PI * w[0]).sin().powi(2);
        let term_last = (w[n - 1] - 1.0).powi(2) * (1.0 + (2.0 * PI * w[n - 1]).sin().powi(2));
        let sum = w[..n - 1]
            .iter()
            .map(|wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + PI).sin().powi(2)))
            .sum::<f64>();
        Ok(term1 + sum + term_last)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0); self.n_dims]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![1.0; self.n_dims], 0.0))
    }
}

/// Branin–Hoo function (2-D).
///
/// Three global minima at approximately:
/// `(-π, 12.275)`, `(π, 2.275)`, `(9.42478, 2.475)` with `f* ≈ 0.397887`.
pub struct BraninFn;

impl TestFunction for BraninFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != 2 {
            return Err(DatasetsError::InvalidFormat(
                "BraninFn: requires exactly 2 input dimensions".into(),
            ));
        }
        let (x1, x2) = (x[0], x[1]);
        let a = 1.0;
        let b = 5.1 / (4.0 * PI * PI);
        let c = 5.0 / PI;
        let r = 6.0;
        let s = 10.0;
        let t = 1.0 / (8.0 * PI);
        Ok(a * (x2 - b * x1 * x1 + c * x1 - r).powi(2)
            + s * (1.0 - t) * x1.cos()
            + s)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-5.0, 10.0), (0.0, 15.0)]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        // One of the three global minima.
        Some((vec![PI, 2.275], 0.397_887_357_729_739))
    }
}

/// Hartmann 6-D function.
///
/// Global minimum: `f(x*) ≈ -3.32237` at known location.
pub struct HartmannFn;

// Hartmann 6-D parameters
const HARTMANN_ALPHA: [f64; 4] = [1.0, 1.2, 3.0, 3.2];
#[rustfmt::skip]
const HARTMANN_A: [[f64; 6]; 4] = [
    [10.0, 3.0,  17.0, 3.5,  1.7,  8.0],
    [0.05, 10.0, 17.0, 0.1,  8.0,  14.0],
    [3.0,  3.5,  1.7,  10.0, 17.0, 8.0],
    [17.0, 8.0,  0.05, 10.0, 0.1,  14.0],
];
#[rustfmt::skip]
const HARTMANN_P: [[f64; 6]; 4] = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];

impl TestFunction for HartmannFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != 6 {
            return Err(DatasetsError::InvalidFormat(
                "HartmannFn: requires exactly 6 input dimensions".into(),
            ));
        }
        let val = (0..4)
            .map(|i| {
                let exp_arg = -(0..6)
                    .map(|j| HARTMANN_A[i][j] * (x[j] - HARTMANN_P[i][j]).powi(2))
                    .sum::<f64>();
                HARTMANN_ALPHA[i] * exp_arg.exp()
            })
            .sum::<f64>();
        Ok(-val)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); 6]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        let x_star = vec![
            0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.657300,
        ];
        Some((x_star, -3.32237))
    }
}

/// Rosenbrock banana function (n-dimensional).
///
/// Global minimum: `f(1,…,1) = 0`.
pub struct RosenbrockFn {
    /// Dimensionality (must be ≥ 2).
    pub n_dims: usize,
}

impl Default for RosenbrockFn {
    fn default() -> Self {
        Self { n_dims: 2 }
    }
}

impl TestFunction for RosenbrockFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.n_dims || self.n_dims < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "RosenbrockFn: expected {} dims (>= 2), got {}",
                self.n_dims,
                x.len()
            )));
        }
        let val = x
            .windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum::<f64>();
        Ok(val)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-5.0, 10.0); self.n_dims]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![1.0; self.n_dims], 0.0))
    }
}

/// Six-Hump Camel function (2-D).
///
/// Two global minima: `f(0.0898, -0.7126) ≈ f(-0.0898, 0.7126) ≈ -1.0316`.
pub struct SixHumpCamelFn;

impl TestFunction for SixHumpCamelFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != 2 {
            return Err(DatasetsError::InvalidFormat(
                "SixHumpCamelFn: requires exactly 2 input dimensions".into(),
            ));
        }
        let (x1, x2) = (x[0], x[1]);
        Ok((4.0 - 2.1 * x1 * x1 + x1.powi(4) / 3.0) * x1 * x1
            + x1 * x2
            + (-4.0 + 4.0 * x2 * x2) * x2 * x2)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-3.0, 3.0), (-2.0, 2.0)]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![0.0898, -0.7126], -1.031_628_453_489_877))
    }
}

/// Bukin N.6 function (2-D).
///
/// Global minimum: `f(-10, 1) = 0`.
pub struct Bukin6Fn;

impl TestFunction for Bukin6Fn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != 2 {
            return Err(DatasetsError::InvalidFormat(
                "Bukin6Fn: requires exactly 2 input dimensions".into(),
            ));
        }
        let (x1, x2) = (x[0], x[1]);
        Ok(100.0 * (x2 - 0.01 * x1 * x1).abs().sqrt() + 0.01 * (x1 + 10.0).abs())
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-15.0, -5.0), (-3.0, 3.0)]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![-10.0, 1.0], 0.0))
    }
}

/// Cross-in-tray function (2-D).
///
/// Four global minima at `(±1.3491, ±1.3491)` with `f* ≈ -2.0626`.
pub struct CrossInTrayFn;

impl TestFunction for CrossInTrayFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != 2 {
            return Err(DatasetsError::InvalidFormat(
                "CrossInTrayFn: requires exactly 2 input dimensions".into(),
            ));
        }
        let (x1, x2) = (x[0], x[1]);
        let exp_arg = (100.0 - (x1 * x1 + x2 * x2).sqrt() / PI).abs();
        let inner = (x1.sin() * x2.sin() * exp_arg.exp()).abs() + 1.0;
        Ok(-0.0001 * inner.powf(0.1))
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0), (-10.0, 10.0)]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![1.3491, 1.3491], -2.062_611_870_099_739))
    }
}

/// Eggholder function (2-D).
///
/// Global minimum: `f(512, 404.2319) ≈ -959.6407`.
pub struct EggholderFn;

impl TestFunction for EggholderFn {
    fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != 2 {
            return Err(DatasetsError::InvalidFormat(
                "EggholderFn: requires exactly 2 input dimensions".into(),
            ));
        }
        let (x1, x2) = (x[0], x[1]);
        Ok(-(x2 + 47.0) * (x2 + x1 / 2.0 + 47.0).abs().sqrt().sin()
            - x1 * (x1 - (x2 + 47.0)).abs().sqrt().sin())
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-512.0, 512.0), (-512.0, 512.0)]
    }

    fn global_min(&self) -> Option<(Vec<f64>, f64)> {
        Some((vec![512.0, 404.231_805_123_766], -959.640_662_720_850_7))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZDT multi-objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// ZDT1 — convex Pareto front (Zitzler–Deb–Thiele 2000).
///
/// `f1 = x[0]`, `g = 1 + 9·Σx[1..] / (n-1)`, `h = 1 - √(f1/g)`, `f2 = g·h`.
pub struct ZDT1 {
    /// Number of decision variables (must be ≥ 2).
    pub n_vars: usize,
}

impl Default for ZDT1 {
    fn default() -> Self {
        Self { n_vars: 30 }
    }
}

impl MultiObjectiveFunction for ZDT1 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "ZDT1: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (self.n_vars - 1) as f64;
        let h = 1.0 - (f1 / g).sqrt();
        Ok(vec![f1, g * h])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.n_vars]
    }
}

/// ZDT2 — non-convex Pareto front.
///
/// `h = 1 - (f1/g)²`.
pub struct ZDT2 {
    /// Number of decision variables.
    pub n_vars: usize,
}

impl Default for ZDT2 {
    fn default() -> Self {
        Self { n_vars: 30 }
    }
}

impl MultiObjectiveFunction for ZDT2 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "ZDT2: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (self.n_vars - 1) as f64;
        let h = 1.0 - (f1 / g).powi(2);
        Ok(vec![f1, g * h])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.n_vars]
    }
}

/// ZDT3 — disconnected Pareto front (oscillating h).
pub struct ZDT3 {
    /// Number of decision variables.
    pub n_vars: usize,
}

impl Default for ZDT3 {
    fn default() -> Self {
        Self { n_vars: 30 }
    }
}

impl MultiObjectiveFunction for ZDT3 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "ZDT3: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (self.n_vars - 1) as f64;
        let h = 1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * PI * f1).sin();
        Ok(vec![f1, g * h])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.n_vars]
    }
}

/// ZDT4 — multimodal with many local Pareto fronts.
pub struct ZDT4 {
    /// Number of decision variables (must be ≥ 2).
    pub n_vars: usize,
}

impl Default for ZDT4 {
    fn default() -> Self {
        Self { n_vars: 10 }
    }
}

impl MultiObjectiveFunction for ZDT4 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "ZDT4: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        let f1 = x[0];
        let g = 1.0 + 10.0 * (self.n_vars - 1) as f64
            + x[1..]
                .iter()
                .map(|xi| xi * xi - 10.0 * (4.0 * PI * xi).cos())
                .sum::<f64>();
        let h = 1.0 - (f1 / g).sqrt();
        Ok(vec![f1, g * h])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut b = vec![(0.0, 1.0)];
        b.extend(vec![(-5.0, 5.0); self.n_vars - 1]);
        b
    }
}

/// ZDT6 — non-convex and biased (sparse Pareto front density).
pub struct ZDT6 {
    /// Number of decision variables (must be ≥ 2).
    pub n_vars: usize,
}

impl Default for ZDT6 {
    fn default() -> Self {
        Self { n_vars: 10 }
    }
}

impl MultiObjectiveFunction for ZDT6 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "ZDT6: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        let f1 = 1.0 - (-4.0 * x[0]).exp() * (6.0 * PI * x[0]).sin().powi(6);
        let mean = x[1..].iter().sum::<f64>() / (self.n_vars - 1) as f64;
        let g = 1.0 + 9.0 * mean.powf(0.25);
        let h = 1.0 - (f1 / g).powi(2);
        Ok(vec![f1, g * h])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.n_vars]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DTLZ1 — 3-objective
// ─────────────────────────────────────────────────────────────────────────────

/// DTLZ1 — 3-objective benchmark with a linear Pareto front (hyperplane).
///
/// `n_vars` must be at least `n_obj`.  The number of objectives is fixed to 3.
pub struct DTLZ1 {
    /// Number of decision variables (typically `n_obj + k - 1`, k = 5).
    pub n_vars: usize,
}

impl Default for DTLZ1 {
    fn default() -> Self {
        Self { n_vars: 7 } // 3 objectives + k=5 - 1
    }
}

impl MultiObjectiveFunction for DTLZ1 {
    fn n_objectives(&self) -> usize {
        3
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars {
            return Err(DatasetsError::InvalidFormat(format!(
                "DTLZ1: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        let m = self.n_objectives();
        let k = self.n_vars - m + 1;
        let g: f64 = (self.n_vars - k..self.n_vars)
            .map(|i| (x[i] - 0.5).powi(2) - (20.0 * PI * (x[i] - 0.5)).cos())
            .sum::<f64>()
            + k as f64;
        let g = 100.0 * g;

        let mut f = vec![0.0f64; m];
        for i in 0..m {
            f[i] = 0.5 * (1.0 + g);
            for j in 0..(m - 1 - i) {
                f[i] *= x[j];
            }
            if i > 0 {
                f[i] *= 1.0 - x[m - 1 - i];
            }
        }
        Ok(f)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.n_vars]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WFG test functions
// ─────────────────────────────────────────────────────────────────────────────

/// WFG1 — Walking Fish Group problem 1 (2-objective version).
///
/// The full WFG suite has many transformations; this implements the 2-objective
/// version with a convex, mixed Pareto front.
pub struct WFG1 {
    /// Number of decision variables (must be ≥ 2).
    pub n_vars: usize,
}

impl Default for WFG1 {
    fn default() -> Self {
        Self { n_vars: 8 }
    }
}

impl MultiObjectiveFunction for WFG1 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "WFG1: expected {} vars, got {}",
                self.n_vars,
                x.len()
            )));
        }
        // Normalise to [0,1] from WFG bounds [0, 2i] for variable i.
        let xn: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi / (2.0 * (i + 1) as f64))
            .collect();

        // Simple shape: convex mixed using first variable as parameter t.
        let t = xn[0];
        // g function: average of remaining vars after bias transformation
        let g: f64 = xn[1..]
            .iter()
            .map(|&xi| {
                let t = xi;
                let s = 0.35f64;
                // Power function bias
                if t <= s {
                    0.35 * (t / s).powf(0.02)
                } else {
                    0.35 + 0.65 * ((t - s) / (1.0 - s)).powf(10.0)
                }
            })
            .sum::<f64>()
            / (self.n_vars - 1) as f64;

        // Shape: convex with cosine bump
        let theta = PI / 2.0 * t;
        let f1 = (1.0 + g) * theta.cos();
        let f2 = (1.0 + g) * theta.sin();
        Ok(vec![f1, f2])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        (0..self.n_vars)
            .map(|i| (0.0, 2.0 * (i + 1) as f64))
            .collect()
    }
}

/// WFG2 — Walking Fish Group problem 2 (2-objective, disconnected front).
pub struct WFG2 {
    /// Number of decision variables (must be ≥ 4 and even for this simplified version).
    pub n_vars: usize,
}

impl Default for WFG2 {
    fn default() -> Self {
        Self { n_vars: 8 }
    }
}

impl MultiObjectiveFunction for WFG2 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_vars || self.n_vars < 4 {
            return Err(DatasetsError::InvalidFormat(format!(
                "WFG2: expected {} vars (>= 4), got {}",
                self.n_vars,
                x.len()
            )));
        }
        // Normalise
        let xn: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi / (2.0 * (i + 1) as f64))
            .collect();

        let t = xn[0];
        let g: f64 = xn[1..].iter().sum::<f64>() / (self.n_vars - 1) as f64;

        // Disconnected shape via cos².
        let theta = PI / 2.0 * t;
        let f1 = (1.0 + g) * (theta.cos().powi(2) * (2.0 * theta).cos().powi(2)).sqrt();
        let f2 = (1.0 + g) * theta.sin();
        Ok(vec![f1, f2])
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        (0..self.n_vars)
            .map(|i| (0.0, 2.0 * (i + 1) as f64))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ackley_global_min() {
        let f = AckleyFn::default();
        let v = f.evaluate(&[0.0, 0.0]).expect("valid input");
        assert!(v.abs() < 1e-10, "Ackley at origin should be ~0, got {v}");
    }

    #[test]
    fn test_ackley_dim_mismatch() {
        let f = AckleyFn::default();
        assert!(f.evaluate(&[0.0]).is_err());
    }

    #[test]
    fn test_rastrigin_global_min() {
        let f = RastriginFn::default();
        let v = f.evaluate(&[0.0, 0.0]).expect("valid input");
        assert!(v.abs() < 1e-10, "Rastrigin at origin should be ~0, got {v}");
    }

    #[test]
    fn test_griewank_global_min() {
        let f = GriewankFn::default();
        let v = f.evaluate(&[0.0, 0.0]).expect("valid input");
        assert!(v.abs() < 1e-10, "Griewank at origin should be ~0, got {v}");
    }

    #[test]
    fn test_levy_global_min() {
        let f = LevyFn::default();
        let v = f.evaluate(&[1.0, 1.0]).expect("valid input");
        assert!(v.abs() < 1e-10, "Levy at (1,1) should be ~0, got {v}");
    }

    #[test]
    fn test_branin_global_min() {
        let f = BraninFn;
        let v = f.evaluate(&[PI, 2.275]).expect("valid input");
        assert!(
            (v - 0.397_887).abs() < 1e-3,
            "Branin near global min, got {v}"
        );
    }

    #[test]
    fn test_hartmann_evaluates() {
        let f = HartmannFn;
        let v = f.evaluate(&[0.2, 0.15, 0.477, 0.275, 0.312, 0.657]).expect("valid input");
        // Should be close to -3.322
        assert!(v < -3.0, "Hartmann near global min should be < -3, got {v}");
    }

    #[test]
    fn test_rosenbrock_global_min() {
        let f = RosenbrockFn::default();
        let v = f.evaluate(&[1.0, 1.0]).expect("valid input");
        assert!(v.abs() < 1e-10, "Rosenbrock at (1,1) should be ~0, got {v}");
    }

    #[test]
    fn test_six_hump_camel_evaluates() {
        let f = SixHumpCamelFn;
        let v = f.evaluate(&[0.0898, -0.7126]).expect("valid input");
        assert!(v < -1.0, "Six-Hump Camel near global min, got {v}");
    }

    #[test]
    fn test_bukin6_global_min() {
        let f = Bukin6Fn;
        let v = f.evaluate(&[-10.0, 1.0]).expect("valid input");
        assert!(v.abs() < 1e-10, "Bukin6 global min should be ~0, got {v}");
    }

    #[test]
    fn test_cross_in_tray_evaluates() {
        let f = CrossInTrayFn;
        let v = f.evaluate(&[1.3491, 1.3491]).expect("valid input");
        assert!(v < -2.0, "CrossInTray near global min, got {v}");
    }

    #[test]
    fn test_eggholder_evaluates() {
        let f = EggholderFn;
        let v = f.evaluate(&[512.0, 404.2319]).expect("valid input");
        assert!(v < -959.0, "Eggholder near global min, got {v}");
    }

    #[test]
    fn test_zdt1_pareto_front() {
        let zdt = ZDT1::default();
        // On the Pareto front: x[1..] = 0, g = 1.
        let mut x = vec![0.0f64; 30];
        x[0] = 0.5;
        let f = zdt.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
        // f2 should be 1 - sqrt(f1)
        let expected_f2 = 1.0 - (0.5f64).sqrt();
        assert!((f[1] - expected_f2).abs() < 1e-12, "ZDT1 Pareto: {f:?}");
    }

    #[test]
    fn test_zdt2_evaluates() {
        let zdt = ZDT2::default();
        let x = vec![0.3f64; 30];
        let f = zdt.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_zdt3_evaluates() {
        let zdt = ZDT3::default();
        let x = vec![0.2f64; 30];
        let f = zdt.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_zdt4_evaluates() {
        let zdt = ZDT4::default();
        let mut x = vec![0.0f64; 10];
        x[0] = 0.5;
        let f = zdt.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_zdt6_evaluates() {
        let zdt = ZDT6::default();
        let x = vec![0.5f64; 10];
        let f = zdt.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_dtlz1_evaluates() {
        let dtlz = DTLZ1::default();
        let x = vec![0.5f64; 7];
        let f = dtlz.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 3);
        for &fi in &f {
            assert!(fi.is_finite(), "DTLZ1 produced non-finite: {fi}");
        }
    }

    #[test]
    fn test_wfg1_evaluates() {
        let wfg = WFG1::default();
        let x: Vec<f64> = (0..8).map(|i| (i + 1) as f64).collect(); // midpoint of bounds
        let f = wfg.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_wfg2_evaluates() {
        let wfg = WFG2::default();
        let x: Vec<f64> = (0..8).map(|i| (i + 1) as f64).collect();
        let f = wfg.evaluate(&x).expect("valid input");
        assert_eq!(f.len(), 2);
    }
}
