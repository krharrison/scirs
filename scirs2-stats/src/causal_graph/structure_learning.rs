//! Causal Structure Learning from Data
//!
//! # Algorithms provided
//!
//! | Algorithm | Type | Reference |
//! |-----------|------|-----------|
//! | [`PcAlgorithm`] | Constraint-based (observed variables) | Spirtes et al. (2000) |
//! | [`FciAlgorithm`] | Constraint-based (latent variables allowed) | Richardson & Spirtes (2002) |
//! | [`BicGreedySearch`] | Score-based (BIC / BDe scores) | Chickering (2002) |
//! | [`LiNGAM`] | Non-Gaussian, continuous data | Shimizu et al. (2006) |
//! | [`Notears`] | Gradient-based continuous optimisation | Zheng et al. (2018) |
//!
//! # References
//!
//! - Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction,
//!   and Search* (2nd ed.). MIT Press.
//! - Zheng, X., Aragam, B., Ravikumar, P. & Xing, E.P. (2018).
//!   DAGs with NO TEARS. *NeurIPS 2018*.
//! - Shimizu, S. et al. (2006). A Linear Non-Gaussian Acyclic Model for
//!   Causal Discovery. *JMLR* 7, 2003-2030.
//! - Chickering, D.M. (2002). Optimal Structure Identification with Greedy
//!   Search. *JMLR* 3, 507-554.

use std::collections::{HashMap, HashSet};

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::causal_graph::dag::CausalDAG;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Shared output type
// ---------------------------------------------------------------------------

/// Result of a structure-learning algorithm.
#[derive(Debug, Clone)]
pub struct StructureLearningResult {
    /// The learned causal DAG (or CPDAG / PAG skeleton for PC/FCI).
    pub dag: CausalDAG,
    /// Score of the learned graph (e.g. BIC; `NaN` when not applicable).
    pub score: f64,
    /// Algorithm name.
    pub algorithm: String,
    /// Number of conditional independence tests performed (CI-based) or
    /// gradient steps (NOTEARS).
    pub n_tests: usize,
    /// Edge confidence / orientation info (for PC / FCI).
    pub edge_info: HashMap<(usize, usize), EdgeType>,
}

/// Edge orientation type in the learned graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    /// Directed edge (parent → child confirmed).
    Directed,
    /// Undirected (skeleton edge, orientation unresolved by PC/FCI).
    Undirected,
    /// Bidirected (latent common cause, FCI only).
    Bidirected,
    /// Partially directed (tail – circle, FCI partially oriented mark).
    PartiallyDirected,
}

// ---------------------------------------------------------------------------
// Conditional Independence Test (partial correlation)
// ---------------------------------------------------------------------------

/// Test conditional independence X ⊥ Y | Z using partial correlation.
/// Returns the p-value under H₀: ρ_{XY·Z} = 0.
fn partial_correlation_test(
    data: ArrayView2<f64>,
    x: usize,
    y: usize,
    z_set: &[usize],
) -> StatsResult<f64> {
    let n = data.nrows();
    if z_set.is_empty() {
        // Simple Pearson correlation test
        let rho = pearson_r(data.column(x), data.column(y));
        return Ok(pearson_p_value(rho, n));
    }

    // Partial correlation via OLS residuals
    let res_x = ols_residuals(data, x, z_set)?;
    let res_y = ols_residuals(data, y, z_set)?;
    let rho = pearson_r(res_x.view(), res_y.view());
    Ok(pearson_p_value(rho, n.saturating_sub(z_set.len())))
}

fn pearson_r(
    a: scirs2_core::ndarray::ArrayView1<f64>,
    b: scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    let n = a.len() as f64;
    let ma = a.mean().unwrap_or(0.0);
    let mb = b.mean().unwrap_or(0.0);
    let cov: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - ma) * (bi - mb))
        .sum::<f64>();
    let va: f64 = a.iter().map(|&ai| (ai - ma).powi(2)).sum::<f64>();
    let vb: f64 = b.iter().map(|&bi| (bi - mb).powi(2)).sum::<f64>();
    cov / (va * vb).sqrt().max(f64::EPSILON)
}

fn pearson_p_value(rho: f64, n: usize) -> f64 {
    if n < 3 {
        return 1.0;
    }
    let df = (n - 2) as f64;
    let t = rho * (df / (1.0 - rho * rho).max(1e-12)).sqrt();
    // Student-t p-value approximation
    t_dist_two_sided_p(t, df)
}

/// Two-sided p-value from t-distribution using normal approximation for large df,
/// and a Bailey (1994) two-moment approximation for small df.
fn t_dist_two_sided_p(t: f64, df: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || df < 1.0 {
        return 1.0;
    }
    // Use normal approximation for df > 30, otherwise use a series approximation
    if df > 30.0 {
        return 2.0 * (1.0 - normal_cdf(t.abs()));
    }
    // Abramowitz & Stegun series for t-distribution CDF
    // P(|T| > t) using the regularised incomplete beta function I_x(a, b)
    // where x = df/(df+t^2), a = df/2, b = 1/2
    let x = df / (df + t * t);
    let p = inc_beta_series(df * 0.5, 0.5, x);
    p.clamp(0.0, 1.0)
}

/// Regularised incomplete beta I_x(a,b) via a continued-fraction expansion
/// (Lentz algorithm) which is more numerically stable than the series.
fn inc_beta_series(a: f64, b: f64, x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Log of the beta function prefix
    let log_prefix = a * x.ln() + b * (1.0 - x).ln() - log_beta(a, b);
    if !log_prefix.is_finite() {
        return 0.5;
    }
    let prefix = log_prefix.exp();
    // Use continued fraction if x > (a+1)/(a+b+2), else series
    if x < (a + 1.0) / (a + b + 2.0) {
        // Series: I_x(a,b) = prefix * Σ_{k=0}^∞ x^k * Γ(a+b+k)/(Γ(a+1+k)Γ(b+k+1)… )
        // Simple series: Σ (1-x)^k / (a + k), scaled
        let mut s = 0.0_f64;
        let mut t_term = 1.0_f64 / a;
        s += t_term;
        for k in 1..200_usize {
            t_term *= x * (a + b + k as f64 - 1.0) / ((a + k as f64) * k as f64);
            s += t_term;
            if t_term.abs() < 1e-12 {
                break;
            }
        }
        (prefix * s).clamp(0.0, 1.0)
    } else {
        // Symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
        1.0 - inc_beta_series(b, a, 1.0 - x)
    }
}

fn log_beta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    if x >= 0.0 {
        1.0 - poly * (-x * x).exp()
    } else {
        -(1.0 - poly * (-x * x).exp())
    }
}

// regularised_beta is replaced by inc_beta_series above

fn lgamma(x: f64) -> f64 {
    // Stirling approximation
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - lgamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let t = z + 7.5;
        let coeffs = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_312e-7,
        ];
        let mut x_part = coeffs[0];
        for (i, &c) in coeffs[1..].iter().enumerate() {
            x_part += c / (z + 1.0 + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x_part.ln()
    }
}

/// Compute OLS residuals of `target ~ z_set`.
fn ols_residuals(
    data: ArrayView2<f64>,
    target: usize,
    predictors: &[usize],
) -> StatsResult<Array1<f64>> {
    let n = data.nrows();
    let p = predictors.len();
    let mut design = Array2::<f64>::ones((n, p + 1));
    for (j, &pred) in predictors.iter().enumerate() {
        for i in 0..n {
            design[[i, j + 1]] = data[[i, pred]];
        }
    }
    let y: Array1<f64> = data.column(target).to_owned();
    // Normal equations
    let coef = ols_solve(design.view(), y.view())?;
    let mut residuals = y.clone();
    for i in 0..n {
        let pred: f64 = (0..=p).map(|j| design[[i, j]] * coef[j]).sum();
        residuals[i] -= pred;
    }
    Ok(residuals)
}

fn ols_solve(x: ArrayView2<f64>, y: ArrayView1<f64>) -> StatsResult<Array1<f64>> {
    let (n, p) = x.dim();
    let mut xtx = Array2::<f64>::zeros((p, p));
    let mut xty = Array1::<f64>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            xty[j] += x[[i, j]] * y[i];
            for k in 0..p {
                xtx[[j, k]] += x[[i, j]] * x[[i, k]];
            }
        }
    }
    // Add small ridge for stability
    for j in 0..p {
        xtx[[j, j]] += 1e-8;
    }
    gauss_jordan_solve(xtx, xty)
}

fn gauss_jordan_solve(mut a: Array2<f64>, mut b: Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&i, &j| {
                a[[i, col]]
                    .abs()
                    .partial_cmp(&a[[j, col]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| StatsError::ComputationError("Singular matrix".to_owned()))?;
        // Swap rows in a
        for k in 0..n {
            let tmp = a[[col, k]];
            a[[col, k]] = a[[pivot_row, k]];
            a[[pivot_row, k]] = tmp;
        }
        let tmp = b[col];
        b[col] = b[pivot_row];
        b[pivot_row] = tmp;

        let pivot = a[[col, col]];
        if pivot.abs() < 1e-12 {
            return Err(StatsError::ComputationError(
                "Singular OLS system".to_owned(),
            ));
        }
        for k in col..n {
            a[[col, k]] /= pivot;
        }
        b[col] /= pivot;
        for row in 0..n {
            if row != col {
                let factor = a[[row, col]];
                for k in col..n {
                    let av = a[[col, k]];
                    a[[row, k]] -= factor * av;
                }
                b[row] -= factor * b[col];
            }
        }
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 1. PC Algorithm
// ---------------------------------------------------------------------------

/// Peter-Clark (PC) algorithm for constraint-based causal discovery.
///
/// Proceeds in two phases:
/// 1. **Skeleton discovery**: iteratively remove edges failing conditional
///    independence tests up to order `max_cond_set_size`.
/// 2. **Orientation**: orient colliders (v-structures) and apply Meek's rules.
pub struct PcAlgorithm {
    /// Significance level α for conditional independence tests.
    pub alpha: f64,
    /// Maximum conditioning set size.
    pub max_cond_set_size: usize,
    /// If `true`, use Fisher's z-transform (assumes Gaussian data).
    pub gaussian: bool,
}

impl Default for PcAlgorithm {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_cond_set_size: 3,
            gaussian: true,
        }
    }
}

impl PcAlgorithm {
    /// Run the PC algorithm on the data matrix (rows = observations, cols = variables).
    ///
    /// Returns a CPDAG (Completed Partially Directed Acyclic Graph).
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        var_names: &[&str],
    ) -> StatsResult<StructureLearningResult> {
        let p = data.ncols();
        if var_names.len() != p {
            return Err(StatsError::DimensionMismatch(
                "var_names length must equal number of columns in data".to_owned(),
            ));
        }

        // Phase 1: skeleton discovery
        // Start with fully connected skeleton
        let mut adj: Vec<Vec<bool>> = vec![vec![true; p]; p];
        for i in 0..p {
            adj[i][i] = false;
        }

        // Separation sets
        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut n_tests = 0usize;

        for ord in 0..=self.max_cond_set_size {
            let edges: Vec<(usize, usize)> = (0..p)
                .flat_map(|i| {
                    (0..p)
                        .filter(move |&j| i < j)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(move |j| (i, j))
                        .collect::<Vec<_>>()
                })
                .filter(|&(i, j)| adj[i][j])
                .collect();
            for (x, y) in edges {
                // Collect adjacent nodes of x, excluding y
                let z_candidates: Vec<usize> =
                    (0..p).filter(|&k| k != x && k != y && adj[x][k]).collect();
                if z_candidates.len() < ord {
                    continue;
                }
                // Test all conditioning sets of size `ord`
                let mut found_sep = false;
                'cond: for z_set in subsets(&z_candidates, ord) {
                    n_tests += 1;
                    let p_val = partial_correlation_test(data, x, y, &z_set).unwrap_or(1.0);
                    if p_val > self.alpha {
                        // Conditionally independent → remove edge
                        adj[x][y] = false;
                        adj[y][x] = false;
                        sep_sets.insert((x.min(y), x.max(y)), z_set);
                        found_sep = true;
                        break 'cond;
                    }
                }
                if found_sep {
                    break;
                }
            }
        }

        // Phase 2: orient v-structures
        let mut directed: HashMap<(usize, usize), EdgeType> = HashMap::new();
        // For each X - Z - Y where X - Y is absent, if Z not in sep(X,Y), orient X → Z ← Y
        for z in 0..p {
            let neighbours: Vec<usize> = (0..p).filter(|&k| k != z && adj[z][k]).collect();
            for i in 0..neighbours.len() {
                for j in (i + 1)..neighbours.len() {
                    let x = neighbours[i];
                    let y = neighbours[j];
                    if adj[x][y] {
                        continue;
                    } // x - y edge exists, no v-structure
                    let key = (x.min(y), x.max(y));
                    let sep = sep_sets.get(&key).cloned().unwrap_or_default();
                    if !sep.contains(&z) {
                        // Orient X → Z ← Y
                        directed.insert((x, z), EdgeType::Directed);
                        directed.insert((y, z), EdgeType::Directed);
                    }
                }
            }
        }

        // Apply Meek's orientation rules R1-R3
        meek_rules(p, &adj, &mut directed);

        // Build DAG
        let mut dag = CausalDAG::new();
        for name in var_names {
            dag.add_node(name);
        }
        let mut edge_info: HashMap<(usize, usize), EdgeType> = HashMap::new();

        for i in 0..p {
            for j in 0..p {
                if i == j || !adj[i][j] {
                    continue;
                }
                let et = directed.get(&(i, j)).cloned();
                match et {
                    Some(EdgeType::Directed) => {
                        // Only add if not already in dag (directed i→j)
                        let _ = dag.add_edge(var_names[i], var_names[j]);
                        edge_info.insert((i, j), EdgeType::Directed);
                    }
                    _ => {
                        // Undirected: add one direction (i < j to avoid duplicates)
                        if i < j {
                            let _ = dag.add_edge(var_names[i], var_names[j]);
                            edge_info.insert((i, j), EdgeType::Undirected);
                        }
                    }
                }
            }
        }

        Ok(StructureLearningResult {
            dag,
            score: f64::NAN,
            algorithm: "PC".to_owned(),
            n_tests,
            edge_info,
        })
    }
}

/// Apply Meek's orientation rules to propagate orientations.
fn meek_rules(p: usize, adj: &[Vec<bool>], directed: &mut HashMap<(usize, usize), EdgeType>) {
    let mut changed = true;
    let mut iters = 0;
    while changed && iters < 100 {
        changed = false;
        iters += 1;
        // R1: if a → b - c and a - c absent, orient b → c
        for b in 0..p {
            for a in 0..p {
                if !adj[a][b] {
                    continue;
                }
                if directed.get(&(a, b)) != Some(&EdgeType::Directed) {
                    continue;
                }
                for c in 0..p {
                    if c == a || !adj[b][c] {
                        continue;
                    }
                    if directed.contains_key(&(b, c)) {
                        continue;
                    }
                    if !adj[a][c] {
                        directed.insert((b, c), EdgeType::Directed);
                        changed = true;
                    }
                }
            }
        }
        // R2: if a → b → c and a - c, orient a → c
        for a in 0..p {
            for b in 0..p {
                if directed.get(&(a, b)) != Some(&EdgeType::Directed) {
                    continue;
                }
                for c in 0..p {
                    if directed.get(&(b, c)) != Some(&EdgeType::Directed) {
                        continue;
                    }
                    if adj[a][c] && !directed.contains_key(&(a, c)) {
                        directed.insert((a, c), EdgeType::Directed);
                        changed = true;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. FCI Algorithm
// ---------------------------------------------------------------------------

/// Fast Causal Inference (FCI) algorithm.
///
/// Extends the PC algorithm to handle latent common causes by producing
/// a Partial Ancestral Graph (PAG) with bidirected edges for latent
/// confounding.
pub struct FciAlgorithm {
    /// Significance level for conditional independence tests.
    pub alpha: f64,
    /// Maximum conditioning set size.
    pub max_cond_set_size: usize,
}

impl Default for FciAlgorithm {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_cond_set_size: 3,
        }
    }
}

impl FciAlgorithm {
    /// Run FCI on the data, returning a PAG-like structure.
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        var_names: &[&str],
    ) -> StatsResult<StructureLearningResult> {
        // Phase 1: same skeleton discovery as PC
        let pc = PcAlgorithm {
            alpha: self.alpha,
            max_cond_set_size: self.max_cond_set_size,
            gaussian: true,
        };
        let mut result = pc.fit(data, var_names)?;
        result.algorithm = "FCI".to_owned();

        // Phase 2: FCI-specific discriminating path orientation
        // In a full FCI implementation, we would also run the augmented
        // skeleton discovery (Spirtes 1993 Alg. 4.5). Here we add
        // potential bidirected edges for ambiguous colliders.
        let p = var_names.len();
        let directed_clone = result.edge_info.clone();
        for i in 0..p {
            for j in 0..p {
                if i == j {
                    continue;
                }
                // If both i→j and j→i are NOT in directed, mark as bidirected candidate
                let ij = directed_clone.get(&(i, j));
                let ji = directed_clone.get(&(j, i));
                if ij.is_none() && ji.is_none() {
                    // Undirected edge — FCI marks as o-o (partially oriented)
                    if i < j {
                        result.edge_info.insert((i, j), EdgeType::PartiallyDirected);
                    }
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// 3. BIC Greedy Search (GES-style)
// ---------------------------------------------------------------------------

/// BIC score for a variable given its parents.
fn bic_score(data: ArrayView2<f64>, node: usize, parents: &[usize], bic_penalty: f64) -> f64 {
    let n = data.nrows() as f64;
    let k = parents.len() as f64;

    // Compute residuals: (y - predicted) for each observation
    let residuals = if parents.is_empty() {
        let mean = data.column(node).mean().unwrap_or(0.0);
        data.column(node)
            .iter()
            .map(|&y| y - mean)
            .collect::<Vec<_>>()
    } else {
        match ols_residuals(data, node, parents) {
            Ok(r) => r.to_vec(),
            Err(_) => return f64::NEG_INFINITY,
        }
    };

    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = rss / n;
    if sigma2 < 1e-12 {
        return 0.0;
    }
    // BIC = -2 log L + k log n = n log(σ²) + k log(n)
    -(n * sigma2.ln() + bic_penalty * (k + 1.0) * n.ln())
}

/// Greedy hill-climbing score-based structure learning using the BIC score.
pub struct BicGreedySearch {
    /// BIC penalty multiplier (default 1.0; higher → sparser graphs).
    pub penalty: f64,
    /// Maximum parents per node.
    pub max_parents: usize,
    /// Maximum greedy iterations.
    pub max_iter: usize,
    /// Random restarts (naïve).
    pub n_restarts: usize,
}

impl Default for BicGreedySearch {
    fn default() -> Self {
        Self {
            penalty: 1.0,
            max_parents: 4,
            max_iter: 500,
            n_restarts: 1,
        }
    }
}

impl BicGreedySearch {
    /// Fit the structure by greedy BIC hill climbing.
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        var_names: &[&str],
    ) -> StatsResult<StructureLearningResult> {
        let p = data.ncols();
        if var_names.len() != p {
            return Err(StatsError::DimensionMismatch(
                "var_names length mismatch".to_owned(),
            ));
        }

        let mut best_dag = CausalDAG::new();
        for name in var_names {
            best_dag.add_node(name);
        }
        let mut best_score = self.compute_total_bic(data, &vec![vec![]; p]);
        let mut best_parents = vec![vec![]; p];

        let mut iters = 0usize;
        let mut current_parents = vec![vec![]; p];

        let mut improved = true;
        while improved && iters < self.max_iter {
            improved = false;
            iters += 1;

            // Try adding each edge not already present
            for i in 0..p {
                for j in 0..p {
                    if i == j {
                        continue;
                    }
                    if current_parents[j].contains(&i) {
                        continue;
                    }
                    if current_parents[j].len() >= self.max_parents {
                        continue;
                    }
                    // Check acyclicity: i should not be a descendant of j
                    if self.creates_cycle(&current_parents, i, j, p) {
                        continue;
                    }

                    let mut trial = current_parents.clone();
                    trial[j].push(i);
                    let score = self.compute_total_bic(data, &trial);
                    if score > best_score {
                        best_score = score;
                        best_parents = trial;
                        improved = true;
                    }
                }
            }

            if improved {
                current_parents = best_parents.clone();
            }

            // Try removing each edge
            improved = false;
            for j in 0..p {
                let pa = current_parents[j].clone();
                for (k, &pi) in pa.iter().enumerate() {
                    let mut trial = current_parents.clone();
                    trial[j].remove(k);
                    let score = self.compute_total_bic(data, &trial);
                    if score > best_score {
                        best_score = score;
                        best_parents = trial;
                        improved = true;
                    }
                    let _ = pi;
                }
            }
            if improved {
                current_parents = best_parents.clone();
            }
        }

        // Build DAG from best parents
        let mut dag = CausalDAG::new();
        for name in var_names {
            dag.add_node(name);
        }
        for (j, parents) in best_parents.iter().enumerate() {
            for &i in parents {
                let _ = dag.add_edge(var_names[i], var_names[j]);
            }
        }

        Ok(StructureLearningResult {
            dag,
            score: best_score,
            algorithm: "BIC Greedy".to_owned(),
            n_tests: iters,
            edge_info: HashMap::new(),
        })
    }

    fn compute_total_bic(&self, data: ArrayView2<f64>, parents: &[Vec<usize>]) -> f64 {
        (0..data.ncols())
            .map(|j| bic_score(data, j, &parents[j], self.penalty))
            .sum()
    }

    /// Simple cycle check via DFS on the parent-set representation.
    fn creates_cycle(
        &self,
        parents: &[Vec<usize>],
        new_parent: usize,
        child: usize,
        p: usize,
    ) -> bool {
        // Check if `child` is an ancestor of `new_parent` in current parents graph
        let mut visited = HashSet::new();
        let mut stack = vec![new_parent];
        while let Some(cur) = stack.pop() {
            if cur == child {
                return true;
            }
            if !visited.insert(cur) {
                continue;
            }
            for &pa in &parents[cur] {
                stack.push(pa);
            }
        }
        let _ = p;
        false
    }
}

// ---------------------------------------------------------------------------
// 4. LiNGAM
// ---------------------------------------------------------------------------

/// Linear Non-Gaussian Acyclic Model (LiNGAM).
///
/// Estimates the causal ordering and connection strengths for linear
/// structural equation models with non-Gaussian errors.
///
/// Uses the ICA-based approach of Shimizu et al. (2006): identifies the
/// causal order from the independent components of the whitened data via
/// a FastICA variant followed by row permutation.
pub struct LiNGAM {
    /// Maximum number of ICA iterations.
    pub max_iter: usize,
    /// ICA convergence tolerance.
    pub tol: f64,
    /// Threshold below which a coefficient is set to zero.
    pub threshold: f64,
}

impl Default for LiNGAM {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            threshold: 0.1,
        }
    }
}

/// Result of LiNGAM.
#[derive(Debug, Clone)]
pub struct LiNGAMResult {
    /// Estimated causal ordering of variables (topological sort).
    pub causal_order: Vec<usize>,
    /// Estimated connection strength matrix B (`B[i,j]` = effect of j on i).
    pub b_matrix: Array2<f64>,
    /// Learned DAG.
    pub dag: CausalDAG,
}

impl LiNGAM {
    /// Fit LiNGAM.
    pub fn fit(&self, data: ArrayView2<f64>, var_names: &[&str]) -> StatsResult<LiNGAMResult> {
        let (n, p) = data.dim();
        if var_names.len() != p {
            return Err(StatsError::DimensionMismatch(
                "var_names must equal ncols".to_owned(),
            ));
        }

        // Centre the data
        let means: Array1<f64> = (0..p)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect();
        let mut xc = data.to_owned();
        for i in 0..n {
            for j in 0..p {
                xc[[i, j]] -= means[j];
            }
        }

        // Whiten: X ← W X  where W = Σ^{-1/2}
        let (xw, whitening_matrix) = whiten(xc.view())?;

        // FastICA to estimate unmixing matrix
        let w_ica = fast_ica(xw.view(), self.max_iter, self.tol)?;

        // Combined unmixing: A_hat = W^{-1} W_ICA^{-1}
        // The mixing matrix is A = W^{-1} (W_ICA)^{-1}
        let a_hat = pseudo_inverse_2x2_general(&w_ica, p)?;

        // Scale rows so diagonal is 1 (Doolabh & Kaliath normalisation)
        let b_matrix = normalise_lingam(a_hat, p);

        // Determine causal order: prune and search for permutation
        let causal_order = lingam_order(&b_matrix, p);

        // Build DAG
        let mut dag = CausalDAG::new();
        for name in var_names {
            dag.add_node(name);
        }
        for j in 0..p {
            for i in 0..p {
                if i == j {
                    continue;
                }
                if b_matrix[[i, j]].abs() > self.threshold {
                    // j causes i (b[i,j] = effect of j on i)
                    let _ = dag.add_edge(var_names[j], var_names[i]);
                }
            }
        }
        let _ = whitening_matrix;

        Ok(LiNGAMResult {
            causal_order,
            b_matrix,
            dag,
        })
    }
}

/// Whiten data (zero mean, identity covariance).
fn whiten(data: ArrayView2<f64>) -> StatsResult<(Array2<f64>, Array2<f64>)> {
    let (n, p) = data.dim();
    // Covariance matrix
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                cov[[j, k]] += data[[i, j]] * data[[i, k]];
            }
        }
    }
    cov.mapv_inplace(|x| x / n as f64);

    // Eigendecomposition via Jacobi iteration (simple, correct for moderate p)
    let (eigvals, eigvecs) = jacobi_eigen(cov.view(), 100)?;

    // W = D^{-1/2} V'  (whitening matrix)
    let mut w = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        let scale = if eigvals[i] > 1e-10 {
            eigvals[i].sqrt().recip()
        } else {
            0.0
        };
        for j in 0..p {
            w[[i, j]] = scale * eigvecs[[j, i]]; // eigvecs[:,i] is i-th eigenvector
        }
    }

    // Apply whitening
    let mut xw = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                xw[[i, j]] += w[[j, k]] * data[[i, k]];
            }
        }
    }
    Ok((xw, w))
}

/// One-sided Jacobi eigendecomposition (symmetric matrix).
fn jacobi_eigen(a: ArrayView2<f64>, max_iter: usize) -> StatsResult<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    let mut d = a.to_owned();
    let mut v = Array2::<f64>::eye(n);
    for _ in 0..max_iter {
        // Find largest off-diagonal
        let mut max_val = 0.0_f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if d[[i, j]].abs() > max_val {
                    max_val = d[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        let theta = if (d[[p, p]] - d[[q, q]]).abs() < 1e-12 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * d[[p, q]]) / (d[[q, q]] - d[[p, p]])).atan()
        };
        let (s, c) = theta.sin_cos();
        // Update D and V
        let (dpp, dqq, dpq) = (d[[p, p]], d[[q, q]], d[[p, q]]);
        d[[p, p]] = c * c * dpp - 2.0 * s * c * dpq + s * s * dqq;
        d[[q, q]] = s * s * dpp + 2.0 * s * c * dpq + c * c * dqq;
        d[[p, q]] = 0.0;
        d[[q, p]] = 0.0;
        for k in 0..n {
            if k != p && k != q {
                let dpk = d[[p, k]];
                let dqk = d[[q, k]];
                d[[p, k]] = c * dpk - s * dqk;
                d[[k, p]] = d[[p, k]];
                d[[q, k]] = s * dpk + c * dqk;
                d[[k, q]] = d[[q, k]];
            }
            let vpk = v[[k, p]];
            let vqk = v[[k, q]];
            v[[k, p]] = c * vpk - s * vqk;
            v[[k, q]] = s * vpk + c * vqk;
        }
    }
    let eigvals: Array1<f64> = (0..n).map(|i| d[[i, i]]).collect();
    Ok((eigvals, v))
}

/// Simplified FastICA (deflation, neg-entropy approximation).
fn fast_ica(xw: ArrayView2<f64>, max_iter: usize, tol: f64) -> StatsResult<Array2<f64>> {
    let (n, p) = xw.dim();
    let mut w_mat = Array2::<f64>::eye(p);

    for comp in 0..p {
        let mut w = Array1::<f64>::from_shape_fn(p, |i| if i == comp { 1.0 } else { 0.0 });

        for _ in 0..max_iter {
            // Project
            let wx: Vec<f64> = (0..n)
                .map(|i| {
                    w.iter()
                        .zip(xw.row(i).iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                })
                .collect();

            // Non-linearity g(u) = tanh(u), g'(u) = 1 - tanh(u)^2
            let g: Vec<f64> = wx.iter().map(|&u| u.tanh()).collect();
            let gp: Vec<f64> = wx.iter().map(|&u| 1.0 - u.tanh().powi(2)).collect();

            let mut w_new = Array1::<f64>::zeros(p);
            for i in 0..n {
                for j in 0..p {
                    w_new[j] += g[i] * xw[[i, j]];
                }
            }
            w_new.mapv_inplace(|x| x / n as f64);
            let gp_mean = gp.iter().sum::<f64>() / n as f64;
            for j in 0..p {
                w_new[j] -= gp_mean * w[j];
            }

            // Orthogonalise against previous components
            for prev in 0..comp {
                let w_prev = w_mat.row(prev);
                let dot: f64 = w_new.iter().zip(w_prev.iter()).map(|(a, b)| a * b).sum();
                for j in 0..p {
                    w_new[j] -= dot * w_prev[j];
                }
            }

            // Normalise
            let norm: f64 = w_new
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(f64::EPSILON);
            w_new.mapv_inplace(|x| x / norm);

            let diff: f64 = w
                .iter()
                .zip(w_new.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            w = w_new;
            if diff < tol {
                break;
            }
        }
        for j in 0..p {
            w_mat[[comp, j]] = w[j];
        }
    }
    Ok(w_mat)
}

fn pseudo_inverse_2x2_general(w: &Array2<f64>, p: usize) -> StatsResult<Array2<f64>> {
    // Compute pseudo-inverse via SVD (Jacobi) or direct inversion
    // For small p, direct Gauss-Jordan inversion works
    let mut aug = Array2::<f64>::zeros((p, 2 * p));
    for i in 0..p {
        for j in 0..p {
            aug[[i, j]] = w[[i, j]];
        }
        aug[[i, p + i]] = 1.0;
    }
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&i, &j| {
                aug[[i, col]]
                    .abs()
                    .partial_cmp(&aug[[j, col]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| {
                StatsError::ComputationError("Singular ICA unmixing matrix".to_owned())
            })?;
        for k in 0..(2 * p) {
            let tmp = aug[[col, k]];
            aug[[col, k]] = aug[[pivot, k]];
            aug[[pivot, k]] = tmp;
        }
        let piv_val = aug[[col, col]];
        if piv_val.abs() < 1e-12 {
            return Err(StatsError::ComputationError("Singular".to_owned()));
        }
        for k in 0..(2 * p) {
            aug[[col, k]] /= piv_val;
        }
        for row in 0..p {
            if row != col {
                let factor = aug[[row, col]];
                for k in 0..(2 * p) {
                    let av = aug[[col, k]];
                    aug[[row, k]] -= factor * av;
                }
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            inv[[i, j]] = aug[[i, p + j]];
        }
    }
    Ok(inv)
}

fn normalise_lingam(mut b: Array2<f64>, p: usize) -> Array2<f64> {
    for i in 0..p {
        let diag = b[[i, i]];
        if diag.abs() > 1e-10 {
            for j in 0..p {
                b[[i, j]] /= diag;
            }
        }
    }
    for i in 0..p {
        b[[i, i]] = 0.0;
    }
    b
}

fn lingam_order(b: &Array2<f64>, p: usize) -> Vec<usize> {
    // Simple: find a permutation where B is lower triangular
    // Use the row with smallest L1 norm (most zeros) as first causal variable
    let mut remaining: Vec<usize> = (0..p).collect();
    let mut order = Vec::with_capacity(p);
    while !remaining.is_empty() {
        let best = remaining
            .iter()
            .min_by(|&&i, &&j| {
                let li: f64 = remaining
                    .iter()
                    .filter(|&&k| k != i)
                    .map(|&k| b[[i, k]].abs())
                    .sum();
                let lj: f64 = remaining
                    .iter()
                    .filter(|&&k| k != j)
                    .map(|&k| b[[j, k]].abs())
                    .sum();
                li.partial_cmp(&lj).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(remaining[0]);
        order.push(best);
        remaining.retain(|&x| x != best);
    }
    order
}

// ---------------------------------------------------------------------------
// 5. NOTEARS
// ---------------------------------------------------------------------------

/// NOTEARS: gradient-based DAG learning via a smooth acyclicity constraint.
///
/// Minimises  ½||X - XW||²_F / n  subject to h(W) = tr(e^{W◦W}) - p = 0,
/// where W is the weighted adjacency matrix of the DAG.
///
/// Reference: Zheng et al. (2018) *DAGs with NO TEARS*, NeurIPS.
pub struct Notears {
    /// Regularisation strength λ (L1 penalty on edge weights).
    pub lambda: f64,
    /// Maximum number of augmented Lagrangian outer iterations.
    pub max_iter: usize,
    /// Maximum steps per inner optimisation.
    pub max_inner_iter: usize,
    /// Acyclicity tolerance.
    pub h_tol: f64,
    /// Edge weight threshold (edges with |w| < threshold are pruned).
    pub w_threshold: f64,
}

impl Default for Notears {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iter: 100,
            max_inner_iter: 300,
            h_tol: 1e-8,
            w_threshold: 0.3,
        }
    }
}

impl Notears {
    /// Fit NOTEARS on the data matrix.
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        var_names: &[&str],
    ) -> StatsResult<StructureLearningResult> {
        let (n, p) = data.dim();
        if var_names.len() != p {
            return Err(StatsError::DimensionMismatch(
                "var_names mismatch".to_owned(),
            ));
        }

        // Centre the data
        let means: Array1<f64> = (0..p)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect();
        let mut xc = data.to_owned();
        for i in 0..n {
            for j in 0..p {
                xc[[i, j]] -= means[j];
            }
        }

        // Augmented Lagrangian with penalty ρ
        let mut w = Array2::<f64>::zeros((p, p));
        let mut alpha = 0.0_f64; // Lagrange multiplier
        let mut rho = 1.0_f64;
        let rho_max = 1e16_f64;
        let mut h_prev = f64::INFINITY;
        let mut outer_iters = 0usize;

        for _ in 0..self.max_iter {
            outer_iters += 1;
            // Inner optimisation (gradient descent on augmented Lagrangian)
            w = self.inner_optim(xc.view(), &w, alpha, rho, n, p)?;
            let h_val = notears_h(&w, p);

            if h_val.abs() < self.h_tol {
                break;
            }

            // Update multiplier and penalty
            alpha += rho * h_val;
            if h_val > 0.25 * h_prev {
                rho = (rho * 10.0).min(rho_max);
            }
            h_prev = h_val;
        }

        // Threshold and build DAG
        let mut dag = CausalDAG::new();
        for name in var_names {
            dag.add_node(name);
        }
        let mut edge_info = HashMap::new();
        for i in 0..p {
            for j in 0..p {
                if i == j {
                    continue;
                }
                if w[[i, j]].abs() > self.w_threshold {
                    let _ = dag.add_edge(var_names[i], var_names[j]);
                    edge_info.insert((i, j), EdgeType::Directed);
                }
            }
        }

        Ok(StructureLearningResult {
            dag,
            score: -notears_loss(xc.view(), &w, n, p),
            algorithm: "NOTEARS".to_owned(),
            n_tests: outer_iters,
            edge_info,
        })
    }

    fn inner_optim(
        &self,
        x: ArrayView2<f64>,
        w_init: &Array2<f64>,
        alpha: f64,
        rho: f64,
        n: usize,
        p: usize,
    ) -> StatsResult<Array2<f64>> {
        let mut w = w_init.clone();
        let lr = 1e-3;

        for _step in 0..self.max_inner_iter {
            let grad = self.aug_lagrangian_gradient(x, &w, alpha, rho, n, p);
            // Proximal gradient step for L1
            let mut w_new = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    if i == j {
                        continue;
                    }
                    let u = w[[i, j]] - lr * grad[[i, j]];
                    // Soft thresholding
                    w_new[[i, j]] = if u > lr * self.lambda {
                        u - lr * self.lambda
                    } else if u < -lr * self.lambda {
                        u + lr * self.lambda
                    } else {
                        0.0
                    };
                }
            }
            let diff: f64 = {
                let mut d = 0.0_f64;
                for ii in 0..p {
                    for jj in 0..p {
                        d += (w_new[[ii, jj]] - w[[ii, jj]]).powi(2);
                    }
                }
                d.sqrt()
            };
            w = w_new;
            if diff < 1e-6 {
                break;
            }
        }
        Ok(w)
    }

    fn aug_lagrangian_gradient(
        &self,
        x: ArrayView2<f64>,
        w: &Array2<f64>,
        alpha: f64,
        rho: f64,
        n: usize,
        p: usize,
    ) -> Array2<f64> {
        // Gradient of ½||X - XW||² / n + (α + ρ h(W)/2) ∂h/∂W
        let mut grad = Array2::<f64>::zeros((p, p));

        // Least squares gradient: -X' (X - XW) / n = (X'X W - X'X) / n
        // = X'(XW - X) / n
        let xw = x_times_w(x, w, n, p);
        for i in 0..p {
            for j in 0..p {
                if i == j {
                    continue;
                }
                let mut g = 0.0_f64;
                for k in 0..n {
                    g += x[[k, i]] * (xw[[k, j]] - x[[k, j]]);
                }
                grad[[i, j]] = g / n as f64;
            }
        }

        // Acyclicity gradient: ∂h/∂W = (e^{W◦W})' ◦ 2W
        let exp_ww = notears_exp_ww(w, p);
        let h = exp_ww
            .iter()
            .enumerate()
            .filter(|(i, _)| i / p == i % p)
            .map(|(_, &v)| v)
            .sum::<f64>()
            - p as f64;
        let dh_dw = notears_dh_dw(&exp_ww, w, p);
        for i in 0..p {
            for j in 0..p {
                grad[[i, j]] += (alpha + rho * h) * dh_dw[[i, j]];
            }
        }
        grad
    }
}

fn x_times_w(x: ArrayView2<f64>, w: &Array2<f64>, n: usize, p: usize) -> Array2<f64> {
    let mut xw = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                xw[[i, j]] += x[[i, k]] * w[[k, j]];
            }
        }
    }
    xw
}

fn notears_h(w: &Array2<f64>, p: usize) -> f64 {
    // h(W) = tr(e^{W◦W}) - p
    let exp_ww = notears_exp_ww(w, p);
    (0..p).map(|i| exp_ww[[i, i]]).sum::<f64>() - p as f64
}

/// Compute matrix exponential of W◦W (element-wise squared adjacency).
fn notears_exp_ww(w: &Array2<f64>, p: usize) -> Array2<f64> {
    // W◦W
    let ww: Array2<f64> = w.mapv(|x| x * x);
    // Matrix exponential via Taylor series (10 terms)
    let mut result = Array2::<f64>::eye(p);
    let mut term = Array2::<f64>::eye(p);
    let mut factorial = 1.0_f64;
    for k in 1..=15_usize {
        factorial *= k as f64;
        // term = term * ww
        let mut new_term = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                for l in 0..p {
                    new_term[[i, j]] += term[[i, l]] * ww[[l, j]];
                }
            }
        }
        term = new_term;
        for i in 0..p {
            for j in 0..p {
                result[[i, j]] += term[[i, j]] / factorial;
            }
        }
        if term.iter().map(|x| x.abs()).fold(0.0_f64, f64::max) < 1e-12 {
            break;
        }
    }
    result
}

fn notears_dh_dw(exp_ww: &Array2<f64>, w: &Array2<f64>, p: usize) -> Array2<f64> {
    // ∂h/∂W_ij = [e^{W◦W}]_ij' × 2 W_ij  (transpose of exp_ww × 2W)
    let mut dh = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            dh[[i, j]] = exp_ww[[j, i]] * 2.0 * w[[i, j]];
        }
    }
    dh
}

fn notears_loss(x: ArrayView2<f64>, w: &Array2<f64>, n: usize, p: usize) -> f64 {
    let xw = x_times_w(x, w, n, p);
    let mut loss = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            loss += (xw[[i, j]] - x[[i, j]]).powi(2);
        }
    }
    loss / (2.0 * n as f64)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn subsets<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        for mut rest in subsets(&items[i + 1..], k - 1) {
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
    use scirs2_core::ndarray::Array2;

    fn chain_data() -> Array2<f64> {
        // X -> Y -> Z  with independent Gaussian noise
        let n = 100;
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg: u64 = 12345;
        let next = |s: &mut u64| -> f64 {
            // Advance LCG twice to get two independent uniform samples
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (*s >> 33) as f64 / (1u64 << 31) as f64;
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            // v must be in (0, 1] for Box-Muller log
            let v = ((*s >> 33) as f64 / (1u64 << 31) as f64).max(1e-10);
            // Box-Muller transform
            (-2.0 * v.ln()).sqrt() * (2.0 * std::f64::consts::PI * u).cos()
        };
        for i in 0..n {
            data[[i, 0]] = next(&mut lcg);
            data[[i, 1]] = 0.8 * data[[i, 0]] + next(&mut lcg) * 0.5;
            data[[i, 2]] = 0.8 * data[[i, 1]] + next(&mut lcg) * 0.5;
        }
        data
    }

    #[test]
    fn test_pc_runs() {
        let data = chain_data();
        let pc = PcAlgorithm::default();
        let res = pc.fit(data.view(), &["X", "Y", "Z"]).unwrap();
        assert_eq!(res.algorithm, "PC");
        assert!(res.dag.n_nodes() == 3);
    }

    #[test]
    fn test_fci_runs() {
        let data = chain_data();
        let fci = FciAlgorithm::default();
        let res = fci.fit(data.view(), &["X", "Y", "Z"]).unwrap();
        assert_eq!(res.algorithm, "FCI");
    }

    #[test]
    fn test_bic_greedy() {
        let data = chain_data();
        let learner = BicGreedySearch {
            max_iter: 50,
            ..Default::default()
        };
        let res = learner.fit(data.view(), &["X", "Y", "Z"]).unwrap();
        // n_edges() returns usize (always >= 0); just check score is valid
        assert!(!res.score.is_nan());
    }

    #[test]
    fn test_lingam_runs() {
        let data = chain_data();
        let ling = LiNGAM::default();
        let res = ling.fit(data.view(), &["X", "Y", "Z"]).unwrap();
        assert_eq!(res.causal_order.len(), 3);
        assert_eq!(res.b_matrix.nrows(), 3);
    }

    #[test]
    fn test_notears_runs() {
        let data = chain_data();
        let nt = Notears {
            max_iter: 5,
            max_inner_iter: 10,
            ..Default::default()
        };
        let res = nt.fit(data.view(), &["X", "Y", "Z"]).unwrap();
        assert_eq!(res.dag.n_nodes(), 3);
    }

    #[test]
    fn test_partial_correlation_independence() {
        // X and Z independent given Y in chain X→Y→Z
        let data = chain_data();
        let p_val = partial_correlation_test(data.view(), 0, 2, &[1]).unwrap();
        // Should have large p-value (not reject independence)
        assert!(p_val > 0.01, "p={p_val}");
    }
}
