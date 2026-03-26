//! Linear Structural Equation Models (LinearSEM) with ndarray interface.
//!
//! This module provides [`LinearSEM`] and [`LinearSEMWithIntercepts`] — high-level
//! wrappers over the core SEM machinery that use `Array2<f64>` from ndarray for
//! data I/O, consistent with the rest of scirs2.
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_stats::causal::linear_sem::LinearSEMWithIntercepts;
//! use scirs2_core::ndarray::array;
//!
//! // Chain: X0 → X1 → X2
//! // B[1,0] = 2.0, B[2,1] = 3.0
//! let b = array![
//!     [0.0, 0.0, 0.0],
//!     [2.0, 0.0, 0.0],
//!     [0.0, 3.0, 0.0],
//! ];
//! let sem = LinearSEMWithIntercepts::new(b, vec![1.0, 0.25, 0.25]).unwrap();
//!
//! // Total causal effect of X0 on X2: 2.0 * 3.0 = 6.0
//! let total = sem.total_effect(0, 2);
//! assert!((total - 6.0).abs() < 1e-10);
//!
//! // do-intervention: fix X1 = 5, then simulate
//! let intervened = sem.do_intervention(1, 5.0);
//! let data = intervened.simulate(100).unwrap();
//! ```
//!
//! ## Model
//!
//! The model is:
//!   `X_i = mu_i + Sum_j B[i,j] * X_j + eps_i`, eps\_i ~ N(0, sigma^2\_i)
//!
//! - `B` is the coefficient matrix: `B[i,j]` is the direct effect of X_j on X_i.
//! - The diagonal of B must be zero (no self-loops).
//! - The graph induced by B must be acyclic (DAG).
//! - For the base [`LinearSEMWithIntercepts`], μ is normally zero and set to
//!   the intervention value after `do_intervention()`.
//!
//! ## Total Effects
//!
//! The total effects matrix is `(I - B)^{-1}`, and the total effect of
//! X_cause on X_effect is `[(I-B)^{-1}]_{effect, cause}`.

use std::collections::VecDeque;

use scirs2_core::ndarray::{Array1, Array2};

use super::sem::satisfies_backdoor;
use crate::bayesian_network::approximate_inference::{LcgRng, Rng};
use crate::bayesian_network::dag::DAG;
use crate::StatsError;

// ---------------------------------------------------------------------------
// LinearSEM
// ---------------------------------------------------------------------------

/// Linear Structural Equation Model (ndarray interface, no intercepts).
///
/// The model is:
///   X = B^T X + ε,  ε_i ~ N(0, σ²_i)
///
/// For do-interventions (which require intercepts), use [`LinearSEMWithIntercepts`]
/// or call [`LinearSEM::with_intercepts`].
///
/// # Examples
///
/// ```rust
/// use scirs2_stats::causal::linear_sem::LinearSEM;
/// use scirs2_core::ndarray::array;
///
/// let b = array![[0.0, 0.0], [2.0, 0.0]];
/// let sem = LinearSEM::new(b, vec![1.0, 0.5]).unwrap();
/// assert!((sem.total_effect(0, 1) - 2.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct LinearSEM {
    /// Coefficient matrix B where `B[i][j]` = effect of X\_j on X\_i.
    pub coefficients: Array2<f64>,
    /// Per-variable noise variances: eps\_i ~ N(0, `noise_variances[i]`).
    pub noise_variances: Vec<f64>,
    /// Number of variables.
    pub n_vars: usize,
    /// Topological order (computed from B).
    pub topo_order: Vec<usize>,
}

impl LinearSEM {
    /// Create a new `LinearSEM`.
    ///
    /// # Errors
    /// - If `coefficients` is not square.
    /// - If `noise_variances` length mismatches.
    /// - If any noise variance is negative.
    /// - If the diagonal is non-zero.
    /// - If the graph implied by `B` contains a cycle.
    pub fn new(coefficients: Array2<f64>, noise_variances: Vec<f64>) -> Result<Self, StatsError> {
        let n = validate_coefficient_matrix(&coefficients, &noise_variances)?;
        let topo_order = topo_sort_from_b(&coefficients, n)?;
        Ok(Self {
            coefficients,
            noise_variances,
            n_vars: n,
            topo_order,
        })
    }

    /// Fit a `LinearSEM` from data using OLS, given an adjacency matrix.
    ///
    /// `adjacency[i][j] > 0` means j → i (j is a parent of i).
    ///
    /// # Returns
    /// Fitted `LinearSEM` with OLS coefficients and residual variances.
    pub fn fit(data: &Array2<f64>, adjacency: &Array2<f64>) -> Result<Self, StatsError> {
        let (b, noise_variances) = ols_fit_from_adjacency(data, adjacency)?;
        Self::new(b, noise_variances)
    }

    /// Simulate `n` observations using a fixed LCG seed (42).
    pub fn simulate(&self, n: usize) -> Result<Array2<f64>, StatsError> {
        let mut rng = LcgRng::new(42);
        simulate_inner(
            &self.coefficients,
            &self.noise_variances,
            &[0.0; 0],
            &self.topo_order,
            n,
            &mut rng,
        )
    }

    /// Simulate `n` observations with a provided RNG.
    pub fn simulate_with_rng(
        &self,
        n: usize,
        rng: &mut impl Rng,
    ) -> Result<Array2<f64>, StatsError> {
        simulate_inner(
            &self.coefficients,
            &self.noise_variances,
            &[0.0; 0],
            &self.topo_order,
            n,
            rng,
        )
    }

    /// Apply a do-intervention, returning a [`LinearSEMWithIntercepts`].
    ///
    /// Sets X_var = `value` (constant) by removing all incoming edges to `var`
    /// and fixing its intercept. The returned model correctly simulates from
    /// the interventional distribution P(X | do(X_var = value)).
    pub fn do_intervention(&self, var: usize, value: f64) -> LinearSEMWithIntercepts {
        let intercepts = vec![0.0; self.n_vars];
        let mut sem = LinearSEMWithIntercepts {
            coefficients: self.coefficients.clone(),
            noise_variances: self.noise_variances.clone(),
            intercepts,
            n_vars: self.n_vars,
            topo_order: self.topo_order.clone(),
            intervened: vec![false; self.n_vars],
        };
        sem.apply_intervention(var, value);
        sem
    }

    /// Compute the total causal effect of `cause` on `effect`.
    ///
    /// Returns `[(I-B)^{-1}]_{effect, cause}`.
    /// Returns 0.0 if indices are out of range or cause == effect.
    pub fn total_effect(&self, cause: usize, effect: usize) -> f64 {
        compute_single_total_effect(&self.coefficients, self.n_vars, cause, effect)
    }

    /// Compute the full total effects matrix (I - B)^{-1}.
    pub fn total_effects_matrix(&self) -> Option<Array2<f64>> {
        compute_total_effects_matrix(&self.coefficients, self.n_vars)
    }

    /// Compute the model-implied covariance matrix:
    ///   Cov(X) = (I-B)^{-1} * Σ_ε * (I-B)^{-T}
    pub fn covariance_matrix(&self) -> Option<Array2<f64>> {
        let inv = compute_total_effects_matrix(&self.coefficients, self.n_vars)?;
        Some(build_covariance(&inv, &self.noise_variances, self.n_vars))
    }

    /// Backdoor adjustment estimate of the ATE of `treatment` on `outcome`.
    ///
    /// Regresses `outcome` on `treatment + adjustment_set` in the observational
    /// data, and returns the coefficient of `treatment`.
    pub fn backdoor_adjustment(
        &self,
        treatment: usize,
        outcome: usize,
        adjustment_set: &[usize],
        data: &Array2<f64>,
    ) -> Result<f64, StatsError> {
        backdoor_ols_estimate(treatment, outcome, adjustment_set, data)
    }

    /// Check the backdoor criterion for (`treatment` → `outcome`) given `z`.
    pub fn satisfies_backdoor_criterion(
        &self,
        treatment: usize,
        outcome: usize,
        z: &[usize],
    ) -> bool {
        match self.to_dag() {
            Ok(dag) => satisfies_backdoor(&dag, treatment, outcome, z),
            Err(_) => false,
        }
    }

    /// Convert coefficient matrix to a `DAG`.
    pub fn to_dag(&self) -> Result<DAG, StatsError> {
        b_to_dag(&self.coefficients, self.n_vars)
    }

    /// Convert to a `LinearSEMWithIntercepts` (with zero intercepts).
    pub fn with_intercepts(self) -> LinearSEMWithIntercepts {
        let n = self.n_vars;
        LinearSEMWithIntercepts {
            coefficients: self.coefficients,
            noise_variances: self.noise_variances,
            intercepts: vec![0.0; n],
            n_vars: n,
            topo_order: self.topo_order,
            intervened: vec![false; n],
        }
    }

    /// Mediation analysis: decompose total effect into direct and indirect.
    ///
    /// For `treatment` → `mediator` → `outcome` (and possibly `treatment` → `outcome`):
    /// - Direct effect = B[outcome, treatment]
    /// - Indirect effect = B[mediator, treatment] × B[outcome, mediator]
    /// - Total effect = (I-B)^{-1}[outcome, treatment]
    pub fn mediation_analysis(
        &self,
        treatment: usize,
        mediator: usize,
        outcome: usize,
    ) -> Result<(f64, f64, f64), StatsError> {
        if treatment >= self.n_vars || mediator >= self.n_vars || outcome >= self.n_vars {
            return Err(StatsError::InvalidInput(format!(
                "Indices out of range: n_vars={}",
                self.n_vars
            )));
        }
        let direct = self.coefficients[[outcome, treatment]];
        let a = self.coefficients[[mediator, treatment]];
        let b = self.coefficients[[outcome, mediator]];
        let indirect = a * b;
        let total = self.total_effect(treatment, outcome);
        Ok((direct, indirect, total))
    }
}

// ---------------------------------------------------------------------------
// LinearSEMWithIntercepts
// ---------------------------------------------------------------------------

/// Linear Structural Equation Model with per-variable intercepts.
///
/// The model is:
///   `X_i = mu_i + Sum_j B[i,j] * X_j + eps_i`, eps\_i ~ N(0, sigma^2\_i)
///
/// Intercepts are normally zero. After `do_intervention(var, value)`, the
/// intercept for `var` is set to `value` and its incoming edges + noise are zeroed.
///
/// # Examples
///
/// ```rust
/// use scirs2_stats::causal::linear_sem::LinearSEMWithIntercepts;
/// use scirs2_core::ndarray::array;
///
/// let b = array![
///     [0.0, 0.0, 0.0],
///     [2.0, 0.0, 0.0],
///     [0.0, 3.0, 0.0],
/// ];
/// let sem = LinearSEMWithIntercepts::new(b, vec![1.0, 0.25, 0.25]).unwrap();
/// // do(X1 = 5) → E[X2] = 15
/// let sem_do = sem.do_intervention(1, 5.0);
/// let data = sem_do.simulate(500).unwrap();
/// let mean_x2: f64 = (0..500).map(|s| data[[s, 2]]).sum::<f64>() / 500.0;
/// assert!((mean_x2 - 15.0).abs() < 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct LinearSEMWithIntercepts {
    /// Coefficient matrix B: `B[i][j]` = direct effect of X\_j on X\_i.
    pub coefficients: Array2<f64>,
    /// Per-variable noise variances.
    pub noise_variances: Vec<f64>,
    /// Per-variable intercepts (μ_i).
    pub intercepts: Vec<f64>,
    /// Number of variables.
    pub n_vars: usize,
    /// Topological order (computed from B).
    pub topo_order: Vec<usize>,
    /// Marks which variables have been fixed by intervention (no noise added).
    pub intervened: Vec<bool>,
}

impl LinearSEMWithIntercepts {
    /// Create a new `LinearSEMWithIntercepts` with zero intercepts.
    pub fn new(coefficients: Array2<f64>, noise_variances: Vec<f64>) -> Result<Self, StatsError> {
        let n = validate_coefficient_matrix(&coefficients, &noise_variances)?;
        let topo_order = topo_sort_from_b(&coefficients, n)?;
        Ok(Self {
            coefficients,
            noise_variances,
            intercepts: vec![0.0; n],
            n_vars: n,
            topo_order,
            intervened: vec![false; n],
        })
    }

    /// Fit from observational data and a known adjacency matrix using OLS.
    pub fn fit(data: &Array2<f64>, adjacency: &Array2<f64>) -> Result<Self, StatsError> {
        let (b, noise_variances) = ols_fit_from_adjacency(data, adjacency)?;
        Self::new(b, noise_variances)
    }

    /// Apply a do-intervention: fix X_var = `value`.
    ///
    /// Returns a new `LinearSEMWithIntercepts` where:
    /// - All incoming coefficients for `var` are zeroed.
    /// - Noise variance for `var` is set to zero.
    /// - Intercept for `var` is set to `value`.
    pub fn do_intervention(&self, var: usize, value: f64) -> Self {
        let mut result = self.clone();
        result.apply_intervention(var, value);
        result
    }

    /// Internal: mutably apply an intervention.
    fn apply_intervention(&mut self, var: usize, value: f64) {
        if var >= self.n_vars {
            return;
        }
        for j in 0..self.n_vars {
            self.coefficients[[var, j]] = 0.0;
        }
        self.noise_variances[var] = 0.0;
        self.intercepts[var] = value;
        self.intervened[var] = true;
    }

    /// Simulate `n` observations with a fixed LCG seed (42).
    pub fn simulate(&self, n: usize) -> Result<Array2<f64>, StatsError> {
        let mut rng = LcgRng::new(42);
        self.simulate_with_rng(n, &mut rng)
    }

    /// Simulate `n` observations with a provided RNG.
    pub fn simulate_with_rng(
        &self,
        n: usize,
        rng: &mut impl Rng,
    ) -> Result<Array2<f64>, StatsError> {
        if n == 0 {
            return Err(StatsError::InvalidInput("n must be positive".to_string()));
        }
        let nv = self.n_vars;
        let mut data = Array2::<f64>::zeros((n, nv));
        for s in 0..n {
            for &node in &self.topo_order {
                let intercept = self.intercepts[node];
                let noise = if self.intervened[node] {
                    0.0
                } else {
                    let std = self.noise_variances[node].sqrt();
                    normal_sample(rng, 0.0, std)
                };
                let mut val = intercept + noise;
                for j in 0..nv {
                    let c = self.coefficients[[node, j]];
                    if c.abs() > 1e-15 {
                        val += c * data[[s, j]];
                    }
                }
                data[[s, node]] = val;
            }
        }
        Ok(data)
    }

    /// Compute the total causal effect of `cause` on `effect`.
    pub fn total_effect(&self, cause: usize, effect: usize) -> f64 {
        compute_single_total_effect(&self.coefficients, self.n_vars, cause, effect)
    }

    /// Compute the full total effects matrix (I - B)^{-1}.
    pub fn total_effects_matrix(&self) -> Option<Array2<f64>> {
        compute_total_effects_matrix(&self.coefficients, self.n_vars)
    }

    /// Compute the model-implied covariance matrix.
    pub fn covariance_matrix(&self) -> Option<Array2<f64>> {
        let inv = compute_total_effects_matrix(&self.coefficients, self.n_vars)?;
        Some(build_covariance(&inv, &self.noise_variances, self.n_vars))
    }

    /// Backdoor adjustment OLS estimate of ATE.
    pub fn backdoor_adjustment(
        &self,
        treatment: usize,
        outcome: usize,
        adjustment_set: &[usize],
        data: &Array2<f64>,
    ) -> Result<f64, StatsError> {
        backdoor_ols_estimate(treatment, outcome, adjustment_set, data)
    }

    /// Check the backdoor criterion.
    pub fn satisfies_backdoor_criterion(
        &self,
        treatment: usize,
        outcome: usize,
        z: &[usize],
    ) -> bool {
        match b_to_dag(&self.coefficients, self.n_vars) {
            Ok(dag) => satisfies_backdoor(&dag, treatment, outcome, z),
            Err(_) => false,
        }
    }

    /// Convert to `DAG`.
    pub fn to_dag(&self) -> Result<DAG, StatsError> {
        b_to_dag(&self.coefficients, self.n_vars)
    }

    /// Mediation analysis for `treatment` → `mediator` → `outcome`.
    pub fn mediation_analysis(
        &self,
        treatment: usize,
        mediator: usize,
        outcome: usize,
    ) -> Result<(f64, f64, f64), StatsError> {
        if treatment >= self.n_vars || mediator >= self.n_vars || outcome >= self.n_vars {
            return Err(StatsError::InvalidInput(format!(
                "Indices out of range: n_vars={}",
                self.n_vars
            )));
        }
        let direct = self.coefficients[[outcome, treatment]];
        let a = self.coefficients[[mediator, treatment]];
        let b = self.coefficients[[outcome, mediator]];
        let indirect = a * b;
        let total = self.total_effect(treatment, outcome);
        Ok((direct, indirect, total))
    }

    /// Estimate the Average Causal Effect by simulation.
    ///
    /// ATE = E[Y | do(X = 1)] - E[Y | do(X = 0)]
    pub fn average_causal_effect(
        &self,
        treatment: usize,
        outcome: usize,
        n_samples: usize,
    ) -> Result<f64, StatsError> {
        let sem1 = self.do_intervention(treatment, 1.0);
        let sem0 = self.do_intervention(treatment, 0.0);
        let mut rng = LcgRng::new(12345);
        let data1 = sem1.simulate_with_rng(n_samples, &mut rng)?;
        let data0 = sem0.simulate_with_rng(n_samples, &mut rng)?;
        let mean1 = (0..n_samples).map(|s| data1[[s, outcome]]).sum::<f64>() / n_samples as f64;
        let mean0 = (0..n_samples).map(|s| data0[[s, outcome]]).sum::<f64>() / n_samples as f64;
        Ok(mean1 - mean0)
    }
}

// ---------------------------------------------------------------------------
// Internal shared helpers
// ---------------------------------------------------------------------------

/// Validate B and noise_variances. Returns n on success.
fn validate_coefficient_matrix(
    b: &Array2<f64>,
    noise_variances: &[f64],
) -> Result<usize, StatsError> {
    let shape = b.shape();
    if shape[0] != shape[1] {
        return Err(StatsError::InvalidInput(format!(
            "Coefficient matrix must be square, got {}×{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    if noise_variances.len() != n {
        return Err(StatsError::InvalidInput(format!(
            "noise_variances length {} != n_vars {}",
            noise_variances.len(),
            n
        )));
    }
    for (i, &v) in noise_variances.iter().enumerate() {
        if v < 0.0 {
            return Err(StatsError::InvalidInput(format!(
                "noise_variances[{i}] = {v} is negative"
            )));
        }
    }
    for i in 0..n {
        if b[[i, i]].abs() > 1e-12 {
            return Err(StatsError::InvalidInput(format!(
                "Diagonal B[{i},{i}] = {} must be zero (no self-loops)",
                b[[i, i]]
            )));
        }
    }
    Ok(n)
}

/// Kahn's topological sort from coefficient matrix B.
/// B[i][j] != 0 (i≠j) means edge j → i.
fn topo_sort_from_b(b: &Array2<f64>, n: usize) -> Result<Vec<usize>, StatsError> {
    let mut in_degree = vec![0usize; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n]; // children[j] = {i : B[i,j]!=0}
    for i in 0..n {
        for j in 0..n {
            if i != j && b[[i, j]].abs() > 1e-12 {
                in_degree[i] += 1;
                children[j].push(i);
            }
        }
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &child in &children[node] {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                queue.push_back(child);
            }
        }
    }
    if order.len() != n {
        return Err(StatsError::InvalidInput(
            "Coefficient matrix implies a cycle — not a valid linear DAG".to_string(),
        ));
    }
    Ok(order)
}

/// OLS fitting from data and adjacency matrix. Returns (B, noise_variances).
fn ols_fit_from_adjacency(
    data: &Array2<f64>,
    adjacency: &Array2<f64>,
) -> Result<(Array2<f64>, Vec<f64>), StatsError> {
    let n_samples = data.shape()[0];
    let n_vars = data.shape()[1];
    if n_samples < 2 {
        return Err(StatsError::InvalidInput(
            "Need at least 2 samples".to_string(),
        ));
    }
    if adjacency.shape()[0] != n_vars || adjacency.shape()[1] != n_vars {
        return Err(StatsError::InvalidInput(format!(
            "adjacency must be {}×{}, got {}×{}",
            n_vars,
            n_vars,
            adjacency.shape()[0],
            adjacency.shape()[1]
        )));
    }
    let mut b = Array2::<f64>::zeros((n_vars, n_vars));
    let mut noise_variances = vec![0.0f64; n_vars];

    for i in 0..n_vars {
        let parents: Vec<usize> = (0..n_vars)
            .filter(|&j| adjacency[[i, j]].abs() > 1e-12)
            .collect();
        if parents.is_empty() {
            let col_mean = (0..n_samples).map(|s| data[[s, i]]).sum::<f64>() / n_samples as f64;
            let var = (0..n_samples)
                .map(|s| (data[[s, i]] - col_mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            noise_variances[i] = var;
            continue;
        }
        let p = parents.len();
        // Build Gram matrix XtX (p×p) and Xty (p)
        let mut xtx = vec![vec![0.0f64; p]; p];
        let mut xty = vec![0.0f64; p];
        for s in 0..n_samples {
            for a in 0..p {
                for bb in 0..p {
                    xtx[a][bb] += data[[s, parents[a]]] * data[[s, parents[bb]]];
                }
                xty[a] += data[[s, parents[a]]] * data[[s, i]];
            }
        }
        let coeffs = solve_linear_system(&xtx, &xty).ok_or_else(|| {
            StatsError::ComputationError(format!(
                "OLS singular for variable {i} with parents {parents:?}"
            ))
        })?;
        let mut sse = 0.0f64;
        for s in 0..n_samples {
            let mut y_hat = 0.0f64;
            for (k, &j) in parents.iter().enumerate() {
                b[[i, j]] = coeffs[k];
                y_hat += coeffs[k] * data[[s, j]];
            }
            sse += (data[[s, i]] - y_hat).powi(2);
        }
        noise_variances[i] = sse / n_samples as f64;
    }
    Ok((b, noise_variances))
}

/// Simulate from a linear SEM (shared logic for LinearSEM and LinearSEMWithIntercepts).
fn simulate_inner(
    b: &Array2<f64>,
    noise_variances: &[f64],
    intercepts: &[f64],
    topo_order: &[usize],
    n: usize,
    rng: &mut impl Rng,
) -> Result<Array2<f64>, StatsError> {
    if n == 0 {
        return Err(StatsError::InvalidInput("n must be positive".to_string()));
    }
    let nv = b.shape()[0];
    let has_intercepts = !intercepts.is_empty();
    let mut data = Array2::<f64>::zeros((n, nv));
    for s in 0..n {
        for &node in topo_order {
            let std = noise_variances[node].sqrt();
            let noise = normal_sample(rng, 0.0, std);
            let intercept = if has_intercepts {
                intercepts[node]
            } else {
                0.0
            };
            let mut val = intercept + noise;
            for j in 0..nv {
                let c = b[[node, j]];
                if c.abs() > 1e-15 {
                    val += c * data[[s, j]];
                }
            }
            data[[s, node]] = val;
        }
    }
    Ok(data)
}

/// Compute the total causal effect of `cause` on `effect` via (I-B)^{-1}.
fn compute_single_total_effect(b: &Array2<f64>, n: usize, cause: usize, effect: usize) -> f64 {
    if cause >= n || effect >= n || cause == effect {
        return 0.0;
    }
    match compute_total_effects_matrix(b, n) {
        Some(inv) => inv[[effect, cause]],
        None => 0.0,
    }
}

/// Compute the full (I - B)^{-1} matrix via Gaussian elimination.
fn compute_total_effects_matrix(b: &Array2<f64>, n: usize) -> Option<Array2<f64>> {
    // Augmented matrix [I-B | I], size n × 2n
    let mut aug = vec![vec![0.0f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = if i == j { 1.0 - b[[i, j]] } else { -b[[i, j]] };
        }
        aug[i][n + i] = 1.0;
    }
    // Gaussian elimination with partial pivoting
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&a, &bb| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[bb][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot_row);
        let pv = aug[col][col];
        if pv.abs() < 1e-14 {
            return None;
        }
        for k in 0..2 * n {
            aug[col][k] /= pv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for k in 0..2 * n {
                let v = aug[col][k];
                aug[row][k] -= factor * v;
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[i][n + j];
        }
    }
    Some(inv)
}

/// Build covariance matrix: Cov(X) = inv * diag(noise) * inv^T.
fn build_covariance(inv: &Array2<f64>, noise: &[f64], n: usize) -> Array2<f64> {
    let mut cov = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += inv[[i, k]] * noise[k] * inv[[j, k]];
            }
            cov[[i, j]] = s;
        }
    }
    cov
}

/// Backdoor OLS estimate: regress outcome on treatment + adjustment_set.
fn backdoor_ols_estimate(
    treatment: usize,
    outcome: usize,
    adjustment_set: &[usize],
    data: &Array2<f64>,
) -> Result<f64, StatsError> {
    let n_samples = data.shape()[0];
    let n_vars = data.shape()[1];
    if treatment >= n_vars {
        return Err(StatsError::InvalidInput(format!(
            "treatment={treatment} >= n_vars={n_vars}"
        )));
    }
    if outcome >= n_vars {
        return Err(StatsError::InvalidInput(format!(
            "outcome={outcome} >= n_vars={n_vars}"
        )));
    }
    if n_samples < 2 {
        return Err(StatsError::InvalidInput(
            "Need at least 2 samples".to_string(),
        ));
    }
    // Predictors: treatment first, then non-duplicate adjustment variables
    let mut predictors = vec![treatment];
    for &z in adjustment_set {
        if z != treatment && z != outcome && !predictors.contains(&z) {
            predictors.push(z);
        }
    }
    let p = predictors.len();
    let mut xtx = vec![vec![0.0f64; p]; p];
    let mut xty = vec![0.0f64; p];
    for s in 0..n_samples {
        for a in 0..p {
            for bb in 0..p {
                xtx[a][bb] += data[[s, predictors[a]]] * data[[s, predictors[bb]]];
            }
            xty[a] += data[[s, predictors[a]]] * data[[s, outcome]];
        }
    }
    let coeffs = solve_linear_system(&xtx, &xty).ok_or_else(|| {
        StatsError::ComputationError(
            "Backdoor adjustment: OLS singular (predictors are collinear)".to_string(),
        )
    })?;
    Ok(coeffs[0])
}

/// Build `DAG` from coefficient matrix B.
fn b_to_dag(b: &Array2<f64>, n: usize) -> Result<DAG, StatsError> {
    let mut dag = DAG::new(n);
    for i in 0..n {
        for j in 0..n {
            if b[[i, j]].abs() > 1e-12 {
                dag.add_edge(j, i)?;
            }
        }
    }
    Ok(dag)
}

/// Gaussian elimination with partial pivoting. Returns x such that Ax = b.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| {
            aug[i][col]
                .abs()
                .partial_cmp(&aug[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot);
        let pv = aug[col][col];
        if pv.abs() < 1e-15 {
            return None;
        }
        for k in col..=n {
            aug[col][k] /= pv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for k in col..=n {
                let v = aug[col][k];
                aug[row][k] -= factor * v;
            }
        }
    }
    Some(aug.iter().map(|row| row[n]).collect())
}

/// Box-Muller normal sample.
fn normal_sample(rng: &mut impl Rng, mean: f64, std: f64) -> f64 {
    if std < 1e-15 {
        return mean;
    }
    let u1 = rng.next_f64().max(1e-15);
    let u2 = rng.next_f64();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std * z
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn chain_b() -> Array2<f64> {
        array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0],]
    }

    #[test]
    fn test_new_valid_chain() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        assert_eq!(sem.n_vars, 3);
        assert_eq!(sem.topo_order[0], 0, "X0 must come first in topo order");
    }

    #[test]
    fn test_new_cycle_rejected() {
        let b = array![[0.0, 1.0], [1.0, 0.0]];
        assert!(LinearSEM::new(b, vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn test_new_diagonal_rejected() {
        let b = array![[1.0, 0.0], [0.0, 0.0]];
        assert!(LinearSEM::new(b, vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn test_new_negative_variance_rejected() {
        let b = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(LinearSEM::new(b, vec![1.0, -0.1]).is_err());
    }

    #[test]
    fn test_total_effect_chain() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        assert!(
            (sem.total_effect(0, 1) - 2.0).abs() < 1e-10,
            "Effect 0→1 should be 2.0"
        );
        assert!(
            (sem.total_effect(1, 2) - 3.0).abs() < 1e-10,
            "Effect 1→2 should be 3.0"
        );
        assert!(
            (sem.total_effect(0, 2) - 6.0).abs() < 1e-10,
            "Effect 0→2 should be 6.0"
        );
        assert!(
            sem.total_effect(2, 0).abs() < 1e-10,
            "Reverse has no effect"
        );
        assert!(sem.total_effect(1, 0).abs() < 1e-10, "No effect 1→0");
    }

    #[test]
    fn test_total_effect_fork() {
        // Z → X, Z → Y (Z=0, X=1, Y=2)
        let b = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],];
        let sem = LinearSEM::new(b, vec![1.0, 1.0, 1.0]).unwrap();
        assert!((sem.total_effect(0, 1) - 2.0).abs() < 1e-10);
        assert!((sem.total_effect(0, 2) - 3.0).abs() < 1e-10);
        assert!(
            sem.total_effect(1, 2).abs() < 1e-10,
            "X has no causal effect on Y"
        );
    }

    #[test]
    fn test_total_effects_matrix() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        let te = sem.total_effects_matrix().unwrap();
        // Identity entries
        assert!((te[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((te[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((te[[2, 2]] - 1.0).abs() < 1e-10);
        // Off-diagonal effects
        assert!((te[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((te[[2, 1]] - 3.0).abs() < 1e-10);
        assert!((te[[2, 0]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_matrix() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        let cov = sem.covariance_matrix().unwrap();
        assert_eq!(cov.shape(), &[3, 3]);
        // Symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((cov[[i, j]] - cov[[j, i]]).abs() < 1e-10);
            }
        }
        // Var(X0) = 1.0
        assert!((cov[[0, 0]] - 1.0).abs() < 1e-10);
        // Var(X1) = 4*1 + 0.25 = 4.25
        assert!((cov[[1, 1]] - 4.25).abs() < 1e-10);
        // Var(X2) = 9 * Var(X1) + 0.25 = 9*4.25 + 0.25 = 38.5
        assert!((cov[[2, 2]] - 38.5).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_mean_zero() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        let data = sem.simulate(3000).unwrap();
        assert_eq!(data.shape(), &[3000, 3]);
        let mean_x0: f64 = (0..3000).map(|s| data[[s, 0]]).sum::<f64>() / 3000.0;
        assert!(mean_x0.abs() < 0.1, "E[X0] ≈ 0, got {mean_x0}");
    }

    #[test]
    fn test_simulate_variance() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        let data = sem.simulate(5000).unwrap();
        let m0 = (0..5000).map(|s| data[[s, 0]]).sum::<f64>() / 5000.0;
        let v0 = (0..5000).map(|s| (data[[s, 0]] - m0).powi(2)).sum::<f64>() / 5000.0;
        assert!((v0 - 1.0).abs() < 0.12, "Var(X0) ≈ 1.0, got {v0}");
        let m1 = (0..5000).map(|s| data[[s, 1]]).sum::<f64>() / 5000.0;
        let v1 = (0..5000).map(|s| (data[[s, 1]] - m1).powi(2)).sum::<f64>() / 5000.0;
        assert!((v1 - 4.25).abs() < 0.4, "Var(X1) ≈ 4.25, got {v1}");
    }

    #[test]
    fn test_do_intervention_with_intercepts() {
        let sem = LinearSEMWithIntercepts::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        // do(X1 = 5) → E[X2] = 3 * 5 = 15
        let sem_do = sem.do_intervention(1, 5.0);
        let data = sem_do.simulate(2000).unwrap();
        // All X1 values must be exactly 5
        for s in 0..2000 {
            assert!(
                (data[[s, 1]] - 5.0).abs() < 1e-10,
                "X1[{s}] = {} != 5.0",
                data[[s, 1]]
            );
        }
        let mean_x2 = (0..2000).map(|s| data[[s, 2]]).sum::<f64>() / 2000.0;
        assert!(
            (mean_x2 - 15.0).abs() < 0.4,
            "E[X2|do(X1=5)] ≈ 15, got {mean_x2}"
        );
    }

    #[test]
    fn test_do_intervention_from_base_linear_sem() {
        let sem = LinearSEM::new(chain_b(), vec![1.0, 0.25, 0.25]).unwrap();
        let sem_do = sem.do_intervention(1, 5.0);
        let data = sem_do.simulate(2000).unwrap();
        for s in 0..2000 {
            assert!((data[[s, 1]] - 5.0).abs() < 1e-10);
        }
        let mean_x2 = (0..2000).map(|s| data[[s, 2]]).sum::<f64>() / 2000.0;
        assert!((mean_x2 - 15.0).abs() < 0.4, "E[X2] ≈ 15, got {mean_x2}");
    }

    #[test]
    fn test_backdoor_adjustment() {
        // Z → X → Y, Z → Y (Z is confounder)
        // B[X, Z] = 1.0, B[Y, Z] = 1.5, B[Y, X] = 2.0
        let b = array![
            [0.0, 0.0, 0.0], // Z (0)
            [1.0, 0.0, 0.0], // X (1): X = Z + eps_X
            [1.5, 2.0, 0.0], // Y (2): Y = 1.5*Z + 2.0*X + eps_Y
        ];
        let sem = LinearSEM::new(b, vec![1.0, 1.0, 1.0]).unwrap();
        let data = sem.simulate(5000).unwrap();
        // Adjustment for confounder Z gives true ATE = B[Y,X] = 2.0
        let ate = sem.backdoor_adjustment(1, 2, &[0], &data).unwrap();
        assert!((ate - 2.0).abs() < 0.15, "Backdoor ATE ≈ 2.0, got {ate}");
    }

    #[test]
    fn test_satisfies_backdoor_criterion() {
        let b = array![
            [0.0, 0.0, 0.0], // Z (0)
            [1.0, 0.0, 0.0], // X (1)
            [1.5, 2.0, 0.0], // Y (2)
        ];
        let sem = LinearSEM::new(b, vec![1.0, 1.0, 1.0]).unwrap();
        // Z adjusts for confounding
        assert!(sem.satisfies_backdoor_criterion(1, 2, &[0]));
        // Empty set fails (Z is unblocked confounder)
        assert!(!sem.satisfies_backdoor_criterion(1, 2, &[]));
    }

    #[test]
    fn test_fit_recovers_true_coefficients() {
        let b_true = array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.5, 0.0],];
        let sem = LinearSEM::new(b_true.clone(), vec![1.0, 0.25, 0.25]).unwrap();
        let data = sem.simulate(5000).unwrap();
        let adj = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],];
        let fitted = LinearSEM::fit(&data, &adj).unwrap();
        assert!(
            (fitted.coefficients[[1, 0]] - 1.5).abs() < 0.1,
            "B[1,0] ≈ 1.5, got {}",
            fitted.coefficients[[1, 0]]
        );
        assert!(
            (fitted.coefficients[[2, 1]] - 2.5).abs() < 0.1,
            "B[2,1] ≈ 2.5, got {}",
            fitted.coefficients[[2, 1]]
        );
    }

    #[test]
    fn test_mediation_analysis() {
        // X → M → Y, X → Y directly
        // B[M, X] = 2.0, B[Y, M] = 3.0, B[Y, X] = 1.0
        let b = array![
            [0.0, 0.0, 0.0], // X (0)
            [2.0, 0.0, 0.0], // M (1)
            [1.0, 3.0, 0.0], // Y (2)
        ];
        let sem = LinearSEM::new(b, vec![1.0, 1.0, 1.0]).unwrap();
        let (direct, indirect, total) = sem.mediation_analysis(0, 1, 2).unwrap();
        assert!((direct - 1.0).abs() < 1e-10, "Direct = 1.0, got {direct}");
        assert!(
            (indirect - 6.0).abs() < 1e-10,
            "Indirect = 2*3 = 6.0, got {indirect}"
        );
        assert!((total - 7.0).abs() < 1e-10, "Total = 7.0, got {total}");
    }

    #[test]
    fn test_ace_by_simulation() {
        let b = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0],];
        let sem = LinearSEMWithIntercepts::new(b, vec![1.0, 0.25, 0.25]).unwrap();
        // ACE of X0 on X1 = 2.0
        let ace = sem.average_causal_effect(0, 1, 5000).unwrap();
        assert!((ace - 2.0).abs() < 0.2, "ACE(X0→X1) ≈ 2.0, got {ace}");
    }
}
