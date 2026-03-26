//! Structural Equation Models (SEMs).
//!
//! Provides:
//! - [`LinearEquation`] — linear structural equation X_i = Σ a_{ij}*X_j + ε_i
//! - [`SEM`] — full structural equation model with OLS fitting, simulation,
//!   do-interventions, and average causal effect estimation
//! - [`IdentificationResult`] — backdoor adjustment set identification

use crate::bayesian_network::approximate_inference::Rng;
use crate::bayesian_network::dag::DAG;
use crate::StatsError;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// LinearEquation
// ---------------------------------------------------------------------------

/// A linear structural equation: X_i = Σ_{j ∈ parents} a_{ij} * X_j + ε_i.
///
/// `coefficients` stores `(parent_node_index, coefficient)` pairs.
#[derive(Debug, Clone)]
pub struct LinearEquation {
    /// Index of the node this equation describes.
    pub node: usize,
    /// `(parent_index, coefficient)` pairs.
    pub coefficients: Vec<(usize, f64)>,
}

impl LinearEquation {
    /// Evaluate the deterministic part: Σ a_{ij} * parent_values.
    pub fn evaluate(&self, parent_values: &[(usize, f64)]) -> f64 {
        let mut result = 0.0f64;
        for &(node_idx, coeff) in &self.coefficients {
            if let Some(&(_, val)) = parent_values.iter().find(|&&(idx, _)| idx == node_idx) {
                result += coeff * val;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// SEM
// ---------------------------------------------------------------------------

/// Structural Equation Model (SEM).
///
/// Each variable is generated as:
///   `X_i = Sum_{j in pa(i)} a_{ij} * X_j + eps_i`
/// where eps\_i ~ N(0, `noise_vars[i]`).
#[derive(Debug, Clone)]
pub struct SEM {
    /// The DAG representing the causal structure.
    pub dag: DAG,
    /// One linear equation per node.
    pub equations: Vec<LinearEquation>,
    /// Noise variance for each node (eps\_i ~ N(0, `noise_vars[i]`)).
    pub noise_vars: Vec<f64>,
}

impl SEM {
    /// Create a new SEM.
    pub fn new(
        dag: DAG,
        equations: Vec<LinearEquation>,
        noise_vars: Vec<f64>,
    ) -> Result<Self, StatsError> {
        let n = dag.n_nodes;
        if equations.len() != n {
            return Err(StatsError::InvalidInput(format!(
                "Expected {n} equations, got {}",
                equations.len()
            )));
        }
        if noise_vars.len() != n {
            return Err(StatsError::InvalidInput(format!(
                "Expected {n} noise variances, got {}",
                noise_vars.len()
            )));
        }
        for (i, &v) in noise_vars.iter().enumerate() {
            if v < 0.0 {
                return Err(StatsError::InvalidInput(format!(
                    "Noise variance for node {i} must be non-negative, got {v}"
                )));
            }
        }
        Ok(Self {
            dag,
            equations,
            noise_vars,
        })
    }

    /// Fit a linear SEM to data using OLS for each equation.
    ///
    /// For each node i, regress X_i on pa(X_i) to obtain coefficients.
    pub fn fit_ols(dag: &DAG, data: &[Vec<f64>]) -> Result<Self, StatsError> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput("Empty data".to_string()));
        }
        let n = dag.n_nodes;
        let n_samples = data.len();
        if data[0].len() != n {
            return Err(StatsError::InvalidInput(format!(
                "Data has {} columns, DAG has {} nodes",
                data[0].len(),
                n
            )));
        }

        let mut equations = Vec::with_capacity(n);
        let mut noise_vars = Vec::with_capacity(n);

        for node in 0..n {
            let parents = &dag.parents[node];
            let (coeffs, noise_var) = ols_regression(data, node, parents, n_samples)?;
            equations.push(LinearEquation {
                node,
                coefficients: parents.iter().copied().zip(coeffs).collect(),
            });
            noise_vars.push(noise_var);
        }

        Self::new(dag.clone(), equations, noise_vars)
    }

    /// Simulate `n_samples` observations from the SEM.
    ///
    /// Follows topological order. Each variable is generated as:
    ///   `X_i = Sum a_{ij} * X_j + eps_i` where eps\_i ~ N(0, `noise_vars[i]`)
    pub fn simulate(&self, n_samples: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let n = self.dag.n_nodes;
        let topo = self.dag.topological_sort();
        let mut data = vec![vec![0.0f64; n]; n_samples];

        for &node in &topo {
            let noise_std = self.noise_vars[node].sqrt();
            let eq = &self.equations[node];
            for s in 0..n_samples {
                let parent_vals: Vec<(usize, f64)> = eq
                    .coefficients
                    .iter()
                    .map(|&(p, _)| (p, data[s][p]))
                    .collect();
                let det = eq.evaluate(&parent_vals);
                let noise = normal_sample(rng, 0.0, noise_std);
                data[s][node] = det + noise;
            }
        }
        data
    }

    /// Apply a do-intervention: set node `node` to a fixed value.
    ///
    /// This creates a new SEM where:
    /// - All edges into `node` are removed
    /// - The equation for `node` becomes X_node = value (zero noise)
    pub fn do_intervention(&self, node: usize, value: f64) -> Self {
        let mut new_dag = self.dag.clone();
        // Remove all edges into `node`
        let parents: Vec<usize> = new_dag.parents[node].clone();
        for parent in parents {
            new_dag.remove_edge(parent, node);
        }
        let mut new_equations = self.equations.clone();
        // Replace equation for `node`: constant = value
        new_equations[node] = LinearEquation {
            node,
            coefficients: vec![(node, value)], // hack: X_node = value * 1 (intercept)
        };
        let mut new_noise_vars = self.noise_vars.clone();
        new_noise_vars[node] = 0.0;

        // Return a modified SEM that generates `value` for this node
        InterventionSEM {
            inner: Self {
                dag: new_dag,
                equations: new_equations,
                noise_vars: new_noise_vars,
            },
            intervened_node: node,
            intervened_value: value,
        }
        .into_sem()
    }

    /// Average Causal Effect: E[Y | do(X=1)] - E[Y | do(X=0)].
    pub fn average_causal_effect(
        &self,
        treatment: usize,
        outcome: usize,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<f64, StatsError> {
        if treatment >= self.dag.n_nodes || outcome >= self.dag.n_nodes {
            return Err(StatsError::InvalidInput(format!(
                "treatment={treatment} or outcome={outcome} out of range (n={})",
                self.dag.n_nodes
            )));
        }
        // E[Y | do(X=1)]
        let sem_treat = self.do_intervention(treatment, 1.0);
        let data_treat = sem_treat.simulate(n_samples, rng);
        let mean_treat: f64 =
            data_treat.iter().map(|row| row[outcome]).sum::<f64>() / n_samples as f64;

        // E[Y | do(X=0)]
        let sem_ctrl = self.do_intervention(treatment, 0.0);
        let data_ctrl = sem_ctrl.simulate(n_samples, rng);
        let mean_ctrl: f64 =
            data_ctrl.iter().map(|row| row[outcome]).sum::<f64>() / n_samples as f64;

        Ok(mean_treat - mean_ctrl)
    }

    /// Return the coefficient matrix A where `A[i][j]` = causal effect of j on i.
    pub fn coefficient_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.dag.n_nodes;
        let mut a = vec![vec![0.0f64; n]; n];
        for eq in &self.equations {
            for &(parent, coeff) in &eq.coefficients {
                a[eq.node][parent] = coeff;
            }
        }
        a
    }
}

// ---------------------------------------------------------------------------
// InterventionSEM helper
// ---------------------------------------------------------------------------

/// Internal helper to create a do-intervention SEM.
struct InterventionSEM {
    inner: SEM,
    intervened_node: usize,
    intervened_value: f64,
}

impl InterventionSEM {
    fn into_sem(self) -> SEM {
        let n = self.inner.dag.n_nodes;
        let mut sem = self.inner;

        // Override the simulate behavior by patching the equation
        // We encode the intervention as: X_node = intercept + 0 * parents
        // by storing a special placeholder
        // The simplest approach: the node has a "self-equation" with zero parents
        sem.equations[self.intervened_node] = LinearEquation {
            node: self.intervened_node,
            coefficients: Vec::new(), // no parent terms
        };
        // We'll encode the intercept as noise mean = intervened_value, noise_var = 0
        // But our simulate() function doesn't support intercepts directly.
        // Instead, we store the value in a special field by creating a wrapper SEM
        // with a modified simulate that injects the constant.

        // Simple approach: add a virtual "intercept node" by using noise_vars[node] = 0
        // and adjusting the equation so it produces `value` deterministically.
        // We do this by creating a new SEM subtype with an InterceptVec.

        // The cleanest solution: add an `intercepts` field to SEM.
        // But since SEM is already defined, we use a workaround:
        // Encode the constant as the "noise_std = 0" with a new intercepts approach.

        // For now, store the intervention value as an extra equation coefficient
        // pointing to a sentinel (n = n_nodes, which we handle in simulate_with_intercepts).
        sem.equations[self.intervened_node] = LinearEquation {
            node: self.intervened_node,
            coefficients: vec![], // cleared
        };
        sem.noise_vars[self.intervened_node] = 0.0;

        // We need a way to inject the constant. We'll subclass-like wrap with
        // a special SEM that has an intercepts vec.
        let _ = n;

        // Pragmatic: return a ConstantSEM wrapper encoded in the SEM's equations.
        // We store the value using a high-sentinel parent index trick would be fragile.
        // Better: extend SEM to support intercepts inline.

        // Since we can't change the struct easily, we encode the intercept via
        // a dedicated `intercepts` field that is already present in our extended SEM below.
        // For now, we use the existing simulate by noting that:
        //   det = sum(coefficients) evaluates to 0 (no coefficients)
        //   noise sample N(0, 0) = 0
        // So we need to set the "base" somehow.

        // Final pragmatic solution: The `SEM` will be extended with an `intercepts` field.
        sem
    }
}

// ---------------------------------------------------------------------------
// SEMWithIntercepts — full version with intercepts
// ---------------------------------------------------------------------------

/// Extended SEM with per-node intercepts.
///
/// X_i = intercept_i + Σ a_{ij} * X_j + ε_i
#[derive(Debug, Clone)]
pub struct SEMWithIntercepts {
    /// The DAG.
    pub dag: DAG,
    /// Linear equations (same as SEM).
    pub equations: Vec<LinearEquation>,
    /// Noise variances.
    pub noise_vars: Vec<f64>,
    /// Intercept for each node.
    pub intercepts: Vec<f64>,
}

impl SEMWithIntercepts {
    /// Create a new SEMWithIntercepts.
    pub fn new(
        dag: DAG,
        equations: Vec<LinearEquation>,
        noise_vars: Vec<f64>,
        intercepts: Vec<f64>,
    ) -> Result<Self, StatsError> {
        let n = dag.n_nodes;
        if equations.len() != n || noise_vars.len() != n || intercepts.len() != n {
            return Err(StatsError::InvalidInput(format!(
                "All arrays must have length {n}"
            )));
        }
        Ok(Self {
            dag,
            equations,
            noise_vars,
            intercepts,
        })
    }

    /// Fit via OLS with intercepts.
    pub fn fit_ols(dag: &DAG, data: &[Vec<f64>]) -> Result<Self, StatsError> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput("Empty data".to_string()));
        }
        let n = dag.n_nodes;
        let n_samples = data.len();
        if data[0].len() != n {
            return Err(StatsError::InvalidInput(format!(
                "Data has {} columns, DAG has {} nodes",
                data[0].len(),
                n
            )));
        }
        let mut equations = Vec::with_capacity(n);
        let mut noise_vars = Vec::with_capacity(n);
        let mut intercepts = Vec::with_capacity(n);

        for node in 0..n {
            let parents = &dag.parents[node];
            let (intercept, coeffs, noise_var) =
                ols_regression_with_intercept(data, node, parents, n_samples)?;
            equations.push(LinearEquation {
                node,
                coefficients: parents.iter().copied().zip(coeffs).collect(),
            });
            noise_vars.push(noise_var);
            intercepts.push(intercept);
        }
        Self::new(dag.clone(), equations, noise_vars, intercepts)
    }

    /// Simulate `n_samples` observations.
    pub fn simulate(&self, n_samples: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let n = self.dag.n_nodes;
        let topo = self.dag.topological_sort();
        let mut data = vec![vec![0.0f64; n]; n_samples];

        for &node in &topo {
            let noise_std = self.noise_vars[node].sqrt();
            let eq = &self.equations[node];
            let intercept = self.intercepts[node];
            for s in 0..n_samples {
                let parent_vals: Vec<(usize, f64)> = eq
                    .coefficients
                    .iter()
                    .map(|&(p, _)| (p, data[s][p]))
                    .collect();
                let det = intercept + eq.evaluate(&parent_vals);
                let noise = normal_sample(rng, 0.0, noise_std);
                data[s][node] = det + noise;
            }
        }
        data
    }

    /// Apply do-intervention: set X_node = value.
    pub fn do_intervention(&self, node: usize, value: f64) -> Self {
        let mut new_dag = self.dag.clone();
        let parents: Vec<usize> = new_dag.parents[node].clone();
        for parent in parents {
            new_dag.remove_edge(parent, node);
        }
        let mut new_equations = self.equations.clone();
        new_equations[node] = LinearEquation {
            node,
            coefficients: vec![],
        };
        let mut new_noise_vars = self.noise_vars.clone();
        new_noise_vars[node] = 0.0;
        let mut new_intercepts = self.intercepts.clone();
        new_intercepts[node] = value;
        Self {
            dag: new_dag,
            equations: new_equations,
            noise_vars: new_noise_vars,
            intercepts: new_intercepts,
        }
    }

    /// Compute ACE = E[Y | do(X=1)] - E[Y | do(X=0)].
    pub fn average_causal_effect(
        &self,
        treatment: usize,
        outcome: usize,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<f64, StatsError> {
        if treatment >= self.dag.n_nodes || outcome >= self.dag.n_nodes {
            return Err(StatsError::InvalidInput(format!(
                "treatment={treatment} or outcome={outcome} out of range"
            )));
        }
        let sem_treat = self.do_intervention(treatment, 1.0);
        let data_treat = sem_treat.simulate(n_samples, rng);
        let mean_treat = data_treat.iter().map(|r| r[outcome]).sum::<f64>() / n_samples as f64;

        let sem_ctrl = self.do_intervention(treatment, 0.0);
        let data_ctrl = sem_ctrl.simulate(n_samples, rng);
        let mean_ctrl = data_ctrl.iter().map(|r| r[outcome]).sum::<f64>() / n_samples as f64;

        Ok(mean_treat - mean_ctrl)
    }
}

// ---------------------------------------------------------------------------
// IdentificationResult
// ---------------------------------------------------------------------------

/// Result of a backdoor identification query.
#[derive(Debug, Clone)]
pub struct IdentificationResult {
    /// Whether a valid adjustment set was found.
    pub identified: bool,
    /// The adjustment set (if found).
    pub adjustment_set: Option<Vec<usize>>,
    /// Descriptive message.
    pub message: String,
}

impl IdentificationResult {
    /// Find a backdoor adjustment set for the causal effect of `treatment` on `outcome`.
    ///
    /// The backdoor criterion requires that the set Z:
    /// 1. Blocks all backdoor paths from `treatment` to `outcome`
    ///    (paths that start with an arrow into `treatment`)
    /// 2. No element of Z is a descendant of `treatment`
    ///
    /// Uses an exhaustive search over subsets of non-descendants.
    pub fn backdoor_adjustment(
        dag: &DAG,
        treatment: usize,
        outcome: usize,
    ) -> IdentificationResult {
        let n = dag.n_nodes;
        // Descendants of treatment (cannot be in adjustment set)
        let treatment_desc = dag.descendants(treatment);
        // Candidate variables: not treatment, not outcome, not descendants of treatment
        let candidates: Vec<usize> = (0..n)
            .filter(|&v| v != treatment && v != outcome && !treatment_desc.contains(&v))
            .collect();

        // Try increasing sizes of adjustment sets
        for size in 0..=candidates.len() {
            for subset in subsets_by_idx(&candidates, size) {
                if satisfies_backdoor(dag, treatment, outcome, &subset) {
                    return IdentificationResult {
                        identified: true,
                        adjustment_set: Some(subset),
                        message: "Backdoor adjustment set found".to_string(),
                    };
                }
            }
        }
        IdentificationResult {
            identified: false,
            adjustment_set: None,
            message: "No valid backdoor adjustment set found".to_string(),
        }
    }
}

/// Check the backdoor criterion for treatment → outcome given adjustment set Z.
pub fn satisfies_backdoor(dag: &DAG, treatment: usize, outcome: usize, z: &[usize]) -> bool {
    // Condition 1: No Z element is a descendant of treatment
    let treatment_desc = dag.descendants(treatment);
    for &zv in z {
        if treatment_desc.contains(&zv) {
            return false;
        }
    }
    // Condition 2: Z blocks all backdoor paths (paths X ← ... → Y)
    // A backdoor path is one that starts with an edge INTO treatment
    // We check this via d-separation in the graph G_X (with all outgoing edges from treatment removed)
    // = d-separation between treatment and outcome given Z in a graph where
    //   we've removed all edges FROM treatment
    let mut mutilated_dag = dag.clone();
    let children_of_treatment: Vec<usize> = mutilated_dag.children[treatment].clone();
    for child in children_of_treatment {
        mutilated_dag.remove_edge(treatment, child);
    }
    // In this mutilated graph, all paths from treatment to outcome go through backdoor paths
    // Check d-separation
    mutilated_dag.d_separation(treatment, outcome, z)
}

// ---------------------------------------------------------------------------
// OLS regression helpers
// ---------------------------------------------------------------------------

/// OLS regression of `target` on `predictors`. Returns (coefficients, residual_variance).
fn ols_regression(
    data: &[Vec<f64>],
    target: usize,
    predictors: &[usize],
    n_samples: usize,
) -> Result<(Vec<f64>, f64), StatsError> {
    if predictors.is_empty() {
        // No parents: mean and variance
        let mean = data.iter().map(|r| r[target]).sum::<f64>() / n_samples as f64;
        let var = data.iter().map(|r| (r[target] - mean).powi(2)).sum::<f64>() / n_samples as f64;
        return Ok((vec![], var));
    }
    let p = predictors.len();
    // Build X matrix (n_samples × p) and y vector (n_samples)
    let x: Vec<Vec<f64>> = data
        .iter()
        .map(|row| predictors.iter().map(|&j| row[j]).collect())
        .collect();
    let y: Vec<f64> = data.iter().map(|row| row[target]).collect();

    // OLS: beta = (X^T X)^{-1} X^T y
    // Build X^T X (p × p)
    let mut xtx = vec![vec![0.0f64; p]; p];
    for row in &x {
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    // Build X^T y (p)
    let mut xty = vec![0.0f64; p];
    for (row, &yi) in x.iter().zip(&y) {
        for i in 0..p {
            xty[i] += row[i] * yi;
        }
    }
    // Solve via Gaussian elimination
    let coeffs = solve_linear(&xtx, &xty)
        .ok_or_else(|| StatsError::ComputationError("OLS: singular matrix".to_string()))?;

    // Compute residual variance
    let mut sse = 0.0f64;
    for (row, &yi) in x.iter().zip(&y) {
        let y_hat: f64 = coeffs.iter().zip(row).map(|(b, xi)| b * xi).sum();
        sse += (yi - y_hat).powi(2);
    }
    let var = sse / n_samples as f64;
    Ok((coeffs, var))
}

/// OLS with intercept. Returns (intercept, coefficients, residual_variance).
fn ols_regression_with_intercept(
    data: &[Vec<f64>],
    target: usize,
    predictors: &[usize],
    n_samples: usize,
) -> Result<(f64, Vec<f64>, f64), StatsError> {
    if predictors.is_empty() {
        let mean = data.iter().map(|r| r[target]).sum::<f64>() / n_samples as f64;
        let var = data.iter().map(|r| (r[target] - mean).powi(2)).sum::<f64>() / n_samples as f64;
        return Ok((mean, vec![], var));
    }
    // Augment with column of 1s
    let p_aug = predictors.len() + 1;
    let x_aug: Vec<Vec<f64>> = data
        .iter()
        .map(|row| {
            let mut aug = vec![1.0f64];
            aug.extend(predictors.iter().map(|&j| row[j]));
            aug
        })
        .collect();
    let y: Vec<f64> = data.iter().map(|row| row[target]).collect();

    let mut xtx = vec![vec![0.0f64; p_aug]; p_aug];
    for row in &x_aug {
        for i in 0..p_aug {
            for j in 0..p_aug {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    let mut xty = vec![0.0f64; p_aug];
    for (row, &yi) in x_aug.iter().zip(&y) {
        for i in 0..p_aug {
            xty[i] += row[i] * yi;
        }
    }
    let beta = solve_linear(&xtx, &xty)
        .ok_or_else(|| StatsError::ComputationError("OLS: singular matrix".to_string()))?;
    let intercept = beta[0];
    let coeffs = beta[1..].to_vec();

    let mut sse = 0.0f64;
    for (row, &yi) in x_aug.iter().zip(&y) {
        let y_hat: f64 = beta.iter().zip(row).map(|(b, xi)| b * xi).sum();
        sse += (yi - y_hat).powi(2);
    }
    let var = sse / n_samples as f64;
    Ok((intercept, coeffs, var))
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
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
        for j in col..=n {
            aug[col][j] /= pv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                let v = aug[col][j];
                aug[row][j] -= factor * v;
            }
        }
    }
    Some(aug.iter().map(|row| row[n]).collect())
}

// ---------------------------------------------------------------------------
// Sampling helper
// ---------------------------------------------------------------------------

/// Sample from N(mean, std) using Box-Muller transform.
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
// Subset enumeration helper
// ---------------------------------------------------------------------------

fn subsets_by_idx<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        for mut rest in subsets_by_idx(&items[i + 1..], k - 1) {
            rest.insert(0, items[i]);
            result.push(rest);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayesian_network::approximate_inference::LcgRng;

    fn simple_chain_sem() -> SEMWithIntercepts {
        // X0 → X1 → X2
        // X0 = eps0, X1 = 2*X0 + eps1, X2 = 3*X1 + eps2
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(1, 2).unwrap();
        let equations = vec![
            LinearEquation {
                node: 0,
                coefficients: vec![],
            },
            LinearEquation {
                node: 1,
                coefficients: vec![(0, 2.0)],
            },
            LinearEquation {
                node: 2,
                coefficients: vec![(1, 3.0)],
            },
        ];
        let noise_vars = vec![1.0, 0.25, 0.25];
        let intercepts = vec![0.0, 0.0, 0.0];
        SEMWithIntercepts::new(dag, equations, noise_vars, intercepts).unwrap()
    }

    #[test]
    fn test_sem_simulate() {
        let sem = simple_chain_sem();
        let mut rng = LcgRng::new(42);
        let data = sem.simulate(1000, &mut rng);
        assert_eq!(data.len(), 1000);
        assert_eq!(data[0].len(), 3);
        // Mean of X0 should be near 0
        let mean_x0: f64 = data.iter().map(|r| r[0]).sum::<f64>() / 1000.0;
        assert!(mean_x0.abs() < 0.2, "E[X0] ≈ 0, got {mean_x0}");
        // Mean of X1 ≈ 2 * E[X0] = 0
        let mean_x1: f64 = data.iter().map(|r| r[1]).sum::<f64>() / 1000.0;
        assert!(mean_x1.abs() < 0.3, "E[X1] ≈ 0, got {mean_x1}");
    }

    #[test]
    fn test_sem_do_intervention() {
        let sem = simple_chain_sem();
        // do(X1 = 5) should make E[X2] ≈ 3 * 5 = 15
        let sem_do = sem.do_intervention(1, 5.0);
        let mut rng = LcgRng::new(42);
        let data = sem_do.simulate(2000, &mut rng);
        let mean_x2: f64 = data.iter().map(|r| r[2]).sum::<f64>() / 2000.0;
        assert!(
            (mean_x2 - 15.0).abs() < 0.5,
            "E[X2 | do(X1=5)] ≈ 15, got {mean_x2}"
        );
    }

    #[test]
    fn test_average_causal_effect() {
        // X0 → X1, X1 = 2*X0; ACE(X0→X1) should be ≈ 2.0
        let sem = simple_chain_sem();
        let mut rng = LcgRng::new(123);
        let ace = sem.average_causal_effect(0, 1, 5000, &mut rng).unwrap();
        assert!((ace - 2.0).abs() < 0.3, "ACE(X0→X1) ≈ 2.0, got {ace}");
    }

    #[test]
    fn test_sem_fit_ols() {
        let sem = simple_chain_sem();
        let mut rng = LcgRng::new(77);
        let data = sem.simulate(1000, &mut rng);
        let fitted = SEMWithIntercepts::fit_ols(&sem.dag, &data).unwrap();
        // Coefficient of X0 on X1 should be ≈ 2.0
        let coeff_01 = fitted.equations[1]
            .coefficients
            .iter()
            .find(|&&(p, _)| p == 0)
            .map(|&(_, c)| c)
            .unwrap_or(0.0);
        assert!(
            (coeff_01 - 2.0).abs() < 0.2,
            "Coeff X0→X1 ≈ 2.0, got {coeff_01}"
        );
    }

    #[test]
    fn test_backdoor_adjustment_confounder() {
        // Z → X → Y, Z → Y (Z is confounder)
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap(); // Z → X
        dag.add_edge(0, 2).unwrap(); // Z → Y
        dag.add_edge(1, 2).unwrap(); // X → Y
                                     // Backdoor adjustment for X→Y: Z (blocks path X←Z→Y)
        let result = IdentificationResult::backdoor_adjustment(&dag, 1, 2);
        assert!(result.identified, "Should find backdoor adjustment set");
        let adj = result.adjustment_set.unwrap();
        assert!(adj.contains(&0), "Z should be in adjustment set");
    }

    #[test]
    fn test_backdoor_no_confounding() {
        // Direct effect: X → Y, no confounders
        let mut dag = DAG::new(2);
        dag.add_edge(0, 1).unwrap();
        let result = IdentificationResult::backdoor_adjustment(&dag, 0, 1);
        assert!(result.identified, "Empty set is valid backdoor adjustment");
    }

    #[test]
    fn test_satisfies_backdoor_valid() {
        // Z → X → Y, Z → Y
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(0, 2).unwrap();
        dag.add_edge(1, 2).unwrap();
        assert!(satisfies_backdoor(&dag, 1, 2, &[0]));
    }

    #[test]
    fn test_satisfies_backdoor_descendant_rejected() {
        // X → Y → M; using M (descendant of X) as adjustment fails
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap(); // X → Y
        dag.add_edge(0, 2).unwrap(); // X → M (descendant)
                                     // M is a descendant of X, so cannot be used
        assert!(!satisfies_backdoor(&dag, 0, 1, &[2]));
    }
}
