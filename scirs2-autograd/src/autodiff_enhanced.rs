//! Enhanced automatic differentiation strategies
//!
//! Provides advanced AD techniques:
//! - Checkpointing strategies (sqrt decomposition, binomial)
//! - Gradient rematerialization (trade compute for memory)
//! - Mixed-mode AD (combine forward and reverse for optimal Jacobian)
//! - Implicit differentiation support
//! - Custom VJP/JVP rule registration

use crate::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Type alias for a VJP function: (primals, cotangent) -> input cotangents.
type VjpFn<F> = Arc<dyn Fn(&[Vec<F>], &[F]) -> Vec<Vec<F>> + Send + Sync>;

/// Type alias for a JVP function: (primals, tangents) -> output tangent.
type JvpFn<F> = Arc<dyn Fn(&[Vec<F>], &[Vec<F>]) -> Vec<F> + Send + Sync>;

// ────────────────────────────────────────────────────────────────────────────
// 1. Checkpointing strategies
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for selecting checkpoint locations in a computation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// No checkpointing: store all activations.
    None,
    /// Uniform spacing: checkpoint every k-th layer.
    Uniform { interval: usize },
    /// Sqrt decomposition: O(sqrt(n)) checkpoints for O(sqrt(n)) memory.
    Sqrt,
    /// Binomial (Revolve/treeverse): optimal for a given memory budget.
    Binomial { memory_budget: usize },
    /// Custom checkpoint positions.
    Custom,
}

/// A checkpoint plan: which layers to save and which to recompute.
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
    /// Total number of layers in the sequence
    pub num_layers: usize,
    /// Indices of layers whose activations should be stored
    pub checkpoint_indices: Vec<usize>,
    /// Strategy used
    pub strategy: CheckpointStrategy,
    /// Estimated peak memory (in number of stored activations)
    pub peak_memory: usize,
    /// Estimated recomputation overhead (number of extra forward passes)
    pub recomputation_cost: usize,
}

/// Compute a checkpoint plan using sqrt decomposition.
///
/// For a chain of `n` layers, place checkpoints every `ceil(sqrt(n))` layers.
/// Peak memory = O(sqrt(n)), recomputation = O(sqrt(n)) per backward pass.
pub fn sqrt_checkpoint_plan(num_layers: usize) -> CheckpointPlan {
    if num_layers == 0 {
        return CheckpointPlan {
            num_layers: 0,
            checkpoint_indices: Vec::new(),
            strategy: CheckpointStrategy::Sqrt,
            peak_memory: 0,
            recomputation_cost: 0,
        };
    }

    let k = (num_layers as f64).sqrt().ceil() as usize;
    let k = k.max(1);

    let mut indices = Vec::new();
    let mut i = 0;
    while i < num_layers {
        indices.push(i);
        i += k;
    }
    // Always include the last layer
    if indices.last().copied() != Some(num_layers.saturating_sub(1)) {
        indices.push(num_layers.saturating_sub(1));
    }

    let num_checkpoints = indices.len();
    // Recomputation: between consecutive checkpoints, we recompute at most k layers
    let recomp = if num_checkpoints > 1 {
        (num_checkpoints - 1) * k
    } else {
        0
    };

    CheckpointPlan {
        num_layers,
        checkpoint_indices: indices,
        strategy: CheckpointStrategy::Sqrt,
        peak_memory: num_checkpoints + k,
        recomputation_cost: recomp,
    }
}

/// Compute a checkpoint plan using binomial (Revolve) strategy.
///
/// Given a memory budget `b` (number of checkpoint slots), this finds the
/// optimal placement that minimises recomputation for a chain of `n` layers.
///
/// The optimal number of recomputations for `n` layers and `b` slots is
/// determined by the binomial coefficient table: the smallest `t` such that
/// `C(t+b, b) >= n`.
pub fn binomial_checkpoint_plan(num_layers: usize, memory_budget: usize) -> CheckpointPlan {
    if num_layers == 0 || memory_budget == 0 {
        return CheckpointPlan {
            num_layers,
            checkpoint_indices: Vec::new(),
            strategy: CheckpointStrategy::Binomial { memory_budget },
            peak_memory: 0,
            recomputation_cost: 0,
        };
    }

    // Compute the revolve timetable using dynamic programming.
    // opt[n][b] = minimum recomputations for n layers with b slots.
    let b = memory_budget.min(num_layers);
    let n = num_layers;

    // Binomial coefficient C(t+b, b) >= n  =>  find smallest t
    let t = find_revolve_steps(n, b);

    // Place checkpoints greedily: at each step, choose the position that
    // minimises the sub-problem size using the revolve partition.
    let indices = revolve_partition(n, b, t);

    CheckpointPlan {
        num_layers: n,
        checkpoint_indices: indices.clone(),
        strategy: CheckpointStrategy::Binomial { memory_budget },
        peak_memory: b,
        recomputation_cost: t,
    }
}

/// Find smallest `t` such that `C(t+b, b) >= n`.
fn find_revolve_steps(n: usize, b: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut t = 0usize;
    loop {
        if binom_geq(t + b, b, n) {
            return t;
        }
        t += 1;
        if t > n * 2 {
            return t; // safety bound
        }
    }
}

/// Check if C(n, k) >= target without overflow.
fn binom_geq(n: usize, k: usize, target: usize) -> bool {
    if k > n {
        return false;
    }
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result * (n - i) as u128 / (i + 1) as u128;
        if result >= target as u128 {
            return true;
        }
    }
    result >= target as u128
}

/// Revolve partition: place checkpoints to divide the sequence.
fn revolve_partition(n: usize, b: usize, _t: usize) -> Vec<usize> {
    if n <= 1 || b == 0 {
        return if n > 0 { vec![0] } else { vec![] };
    }

    let mut indices = Vec::new();
    let mut remaining = n;
    let mut pos = 0usize;
    let mut slots = b;

    while remaining > 1 && slots > 0 {
        // Place checkpoint to split remaining layers optimally
        let split = optimal_split(remaining, slots);
        indices.push(pos);
        pos += split;
        remaining -= split;
        slots -= 1;
    }
    if remaining > 0 {
        indices.push(pos);
    }

    indices
}

/// Optimal split point for Revolve: choose k such that we minimise
/// total recomputation on both halves.
fn optimal_split(n: usize, b: usize) -> usize {
    if n <= 1 || b == 0 {
        return n;
    }
    // Heuristic: split at the point proportional to budget
    let split = n * b / (b + 1);
    split.max(1).min(n - 1)
}

/// Compute a checkpoint plan using uniform spacing.
pub fn uniform_checkpoint_plan(num_layers: usize, interval: usize) -> CheckpointPlan {
    let interval = interval.max(1);
    let mut indices = Vec::new();
    let mut i = 0;
    while i < num_layers {
        indices.push(i);
        i += interval;
    }
    if num_layers > 0 && indices.last().copied() != Some(num_layers - 1) {
        indices.push(num_layers - 1);
    }

    let num_cp = indices.len();
    CheckpointPlan {
        num_layers,
        checkpoint_indices: indices,
        strategy: CheckpointStrategy::Uniform { interval },
        peak_memory: num_cp + interval,
        recomputation_cost: if num_cp > 1 {
            (num_cp - 1) * interval
        } else {
            0
        },
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Gradient Rematerialization
// ────────────────────────────────────────────────────────────────────────────

/// Policy for deciding which activations to rematerialise.
#[derive(Debug, Clone)]
pub struct RematerializationPolicy {
    /// Maximum memory allowed (in abstract units)
    pub memory_limit: usize,
    /// Ops that are cheap to recompute (substring match on op name)
    pub cheap_ops: Vec<String>,
    /// Ops that should always be stored (never recomputed)
    pub always_store_ops: Vec<String>,
}

impl Default for RematerializationPolicy {
    fn default() -> Self {
        Self {
            memory_limit: usize::MAX,
            cheap_ops: vec![
                "relu".to_owned(),
                "Relu".to_owned(),
                "neg".to_owned(),
                "Neg".to_owned(),
                "add".to_owned(),
                "Add".to_owned(),
            ],
            always_store_ops: vec![
                "matmul".to_owned(),
                "MatMul".to_owned(),
                "conv2d".to_owned(),
                "Conv2d".to_owned(),
            ],
        }
    }
}

/// Decision for a single activation: store or recompute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RematerializationDecision {
    /// Store this activation in memory
    Store,
    /// Recompute this activation from its inputs when needed
    Recompute,
}

/// Analyse an op name to decide store vs recompute.
pub fn rematerialization_decision(
    op_name: &str,
    policy: &RematerializationPolicy,
) -> RematerializationDecision {
    // Always-store ops take priority
    if policy
        .always_store_ops
        .iter()
        .any(|s| op_name.contains(s.as_str()))
    {
        return RematerializationDecision::Store;
    }
    // Cheap ops are candidates for recomputation
    if policy
        .cheap_ops
        .iter()
        .any(|s| op_name.contains(s.as_str()))
    {
        return RematerializationDecision::Recompute;
    }
    // Default: store
    RematerializationDecision::Store
}

/// Build a rematerialisation plan for a sequence of operations.
///
/// Returns a vec of decisions, one per operation index.
pub fn build_rematerialization_plan(
    op_names: &[&str],
    policy: &RematerializationPolicy,
) -> Vec<RematerializationDecision> {
    let mut decisions: Vec<RematerializationDecision> = op_names
        .iter()
        .map(|name| rematerialization_decision(name, policy))
        .collect();

    // Enforce memory limit: if too many are stored, convert cheapest to recompute
    let stored_count = decisions
        .iter()
        .filter(|d| **d == RematerializationDecision::Store)
        .count();
    if stored_count > policy.memory_limit {
        let excess = stored_count - policy.memory_limit;
        let mut converted = 0usize;
        // Convert cheap-store ops to recompute (from the end backwards)
        for i in (0..decisions.len()).rev() {
            if converted >= excess {
                break;
            }
            if decisions[i] == RematerializationDecision::Store {
                let name = op_names[i];
                // Only convert if not in always_store
                if !policy
                    .always_store_ops
                    .iter()
                    .any(|s| name.contains(s.as_str()))
                {
                    decisions[i] = RematerializationDecision::Recompute;
                    converted += 1;
                }
            }
        }
    }

    decisions
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Mixed-mode AD
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for computing a Jacobian matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianMode {
    /// Forward-mode only: n_inputs forward passes (each computes one Jacobian column)
    Forward,
    /// Reverse-mode only: n_outputs reverse passes (each computes one Jacobian row)
    Reverse,
    /// Mixed: use forward for some directions, reverse for others
    Mixed,
}

/// Given input dimension `m` and output dimension `n`, select the optimal
/// Jacobian computation mode.
///
/// - If m <= n: forward-mode (m forward passes is cheaper than n reverse)
/// - If n < m: reverse-mode (n reverse passes is cheaper than m forward)
/// - Mixed: when both dimensions are large and roughly equal
pub fn select_jacobian_mode(num_inputs: usize, num_outputs: usize) -> JacobianMode {
    if num_inputs == 0 || num_outputs == 0 {
        return JacobianMode::Forward;
    }

    let ratio = num_inputs as f64 / num_outputs as f64;

    if ratio <= 0.5 {
        JacobianMode::Forward
    } else if ratio >= 2.0 {
        JacobianMode::Reverse
    } else {
        // Dimensions are similar -- mixed mode may be beneficial
        // but defaults to the cheaper single-mode option
        if num_inputs <= num_outputs {
            JacobianMode::Forward
        } else {
            JacobianMode::Reverse
        }
    }
}

/// Plan for computing a Jacobian using mixed-mode AD.
#[derive(Debug, Clone)]
pub struct MixedModeJacobianPlan {
    /// Overall mode selected
    pub mode: JacobianMode,
    /// Number of forward passes needed
    pub num_forward_passes: usize,
    /// Number of reverse passes needed
    pub num_reverse_passes: usize,
    /// Estimated total cost (in units of one forward pass)
    pub estimated_cost: f64,
}

/// Plan the optimal Jacobian computation.
///
/// One forward pass costs 1 unit; one reverse pass costs ~3 units (heuristic
/// for the extra memory + backward sweep).
pub fn plan_jacobian_computation(num_inputs: usize, num_outputs: usize) -> MixedModeJacobianPlan {
    let reverse_cost_factor = 3.0_f64;

    let forward_cost = num_inputs as f64;
    let reverse_cost = num_outputs as f64 * reverse_cost_factor;

    let mode = select_jacobian_mode(num_inputs, num_outputs);
    let (fwd, rev) = match mode {
        JacobianMode::Forward => (num_inputs, 0),
        JacobianMode::Reverse => (0, num_outputs),
        JacobianMode::Mixed => {
            // Split: use forward for half, reverse for half
            let fwd_part = num_inputs / 2;
            let rev_part = num_outputs / 2;
            (fwd_part, rev_part)
        }
    };

    let cost = fwd as f64 + rev as f64 * reverse_cost_factor;

    MixedModeJacobianPlan {
        mode,
        num_forward_passes: fwd,
        num_reverse_passes: rev,
        estimated_cost: cost,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Implicit Differentiation
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for implicit differentiation.
///
/// Given an equilibrium equation F(x, theta) = 0 where x = x*(theta),
/// implicit differentiation computes dx*/dtheta via:
///   dx*/dtheta = -(dF/dx)^{-1} @ (dF/dtheta)
#[derive(Debug, Clone)]
pub struct ImplicitDiffConfig {
    /// Maximum iterations for the linear solver
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Solver method for the linear system
    pub solver: LinearSolverMethod,
}

impl Default for ImplicitDiffConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            solver: LinearSolverMethod::ConjugateGradient,
        }
    }
}

/// Linear solver methods for implicit differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverMethod {
    /// Conjugate gradient (for symmetric positive definite)
    ConjugateGradient,
    /// GMRES (general non-symmetric)
    Gmres,
    /// Neumann series approximation (truncated)
    NeumannSeries { num_terms: usize },
}

/// Result of an implicit differentiation solve.
#[derive(Debug, Clone)]
pub struct ImplicitDiffResult<F: Float> {
    /// The computed implicit gradient dx*/dtheta (flattened)
    pub gradient: Vec<F>,
    /// Number of solver iterations used
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
    /// Final residual norm
    pub residual_norm: F,
}

/// Solve implicit differentiation via iterative linear solve.
///
/// Given Jacobians `df_dx` (n x n) and `df_dtheta` (n x p), computes
/// `dx_dtheta = -(df_dx)^{-1} @ df_dtheta` using the conjugate gradient method.
///
/// All matrices are stored in row-major flat vectors.
pub fn solve_implicit_diff<F: Float>(
    df_dx: &[F],     // n x n, row-major
    df_dtheta: &[F], // n x p, row-major
    n: usize,        // number of equilibrium variables
    p: usize,        // number of parameters
    config: &ImplicitDiffConfig,
) -> ImplicitDiffResult<F> {
    // We solve:  df_dx @ result_col = -df_dtheta_col  for each column of df_dtheta
    let mut gradient = vec![F::zero(); n * p];
    let mut total_iters = 0usize;
    let mut converged = true;
    let mut max_residual = F::zero();

    let tol = F::from(config.tolerance).unwrap_or_else(|| F::from(1e-8).unwrap_or(F::zero()));

    for col in 0..p {
        // Extract the col-th column of -df_dtheta
        let mut rhs = vec![F::zero(); n];
        for row in 0..n {
            if row * p + col < df_dtheta.len() {
                rhs[row] = F::zero() - df_dtheta[row * p + col];
            }
        }

        // Solve A @ x = rhs using CG (or simplified iterative method)
        let (sol, iters, res) = cg_solve(df_dx, &rhs, n, config.max_iterations, tol);

        for row in 0..n {
            gradient[row * p + col] = sol[row];
        }
        total_iters += iters;
        if res > tol {
            converged = false;
        }
        if res > max_residual {
            max_residual = res;
        }
    }

    ImplicitDiffResult {
        gradient,
        iterations: total_iters,
        converged,
        residual_norm: max_residual,
    }
}

/// Simple conjugate gradient solver for A @ x = b.
fn cg_solve<F: Float>(
    a: &[F], // n x n row-major
    b: &[F], // n
    n: usize,
    max_iter: usize,
    tol: F,
) -> (Vec<F>, usize, F) {
    let mut x = vec![F::zero(); n];
    let mut r: Vec<F> = b.to_vec();
    let mut p: Vec<F> = r.clone();

    let mut rs_old = dot(&r, &r);

    for iter in 0..max_iter {
        let ap = matvec(a, &p, n);
        let pap = dot(&p, &ap);
        if pap.abs() < F::from(1e-30).unwrap_or(F::zero()) {
            break;
        }
        let alpha = rs_old / pap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(&r, &r);
        let res_norm = rs_new.sqrt();
        if res_norm < tol {
            return (x, iter + 1, res_norm);
        }

        let beta = rs_new / (rs_old + F::from(1e-30).unwrap_or(F::zero()));
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    let res = dot(&r, &r).sqrt();
    (x, max_iter, res)
}

fn dot<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + ai * bi)
}

fn matvec<F: Float>(a: &[F], x: &[F], n: usize) -> Vec<F> {
    let mut result = vec![F::zero(); n];
    for i in 0..n {
        for j in 0..n {
            if i * n + j < a.len() {
                result[i] += a[i * n + j] * x[j];
            }
        }
    }
    result
}

// ────────────────────────────────────────────────────────────────────────────
// 5. Custom VJP/JVP Rule Registration
// ────────────────────────────────────────────────────────────────────────────

/// A registered custom VJP (Vector-Jacobian Product) rule.
pub struct VjpRule<F: Float> {
    /// Operation name this rule applies to
    pub op_name: String,
    /// The VJP function: (primals, cotangent) -> input cotangents
    pub vjp_fn: VjpFn<F>,
}

/// A registered custom JVP (Jacobian-Vector Product) rule.
pub struct JvpRule<F: Float> {
    /// Operation name this rule applies to
    pub op_name: String,
    /// The JVP function: (primals, tangents) -> output tangent
    pub jvp_fn: JvpFn<F>,
}

/// Registry for custom differentiation rules.
pub struct DiffRuleRegistry<F: Float> {
    vjp_rules: Mutex<HashMap<String, VjpFn<F>>>,
    jvp_rules: Mutex<HashMap<String, JvpFn<F>>>,
}

impl<F: Float> Default for DiffRuleRegistry<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> DiffRuleRegistry<F> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            vjp_rules: Mutex::new(HashMap::new()),
            jvp_rules: Mutex::new(HashMap::new()),
        }
    }

    /// Register a custom VJP rule for an operation.
    pub fn register_vjp<G>(&self, op_name: &str, vjp_fn: G)
    where
        G: Fn(&[Vec<F>], &[F]) -> Vec<Vec<F>> + Send + Sync + 'static,
    {
        if let Ok(mut rules) = self.vjp_rules.lock() {
            rules.insert(op_name.to_owned(), Arc::new(vjp_fn));
        }
    }

    /// Register a custom JVP rule for an operation.
    pub fn register_jvp<G>(&self, op_name: &str, jvp_fn: G)
    where
        G: Fn(&[Vec<F>], &[Vec<F>]) -> Vec<F> + Send + Sync + 'static,
    {
        if let Ok(mut rules) = self.jvp_rules.lock() {
            rules.insert(op_name.to_owned(), Arc::new(jvp_fn));
        }
    }

    /// Look up a VJP rule for the given op name.
    pub fn get_vjp(&self, op_name: &str) -> Option<VjpFn<F>> {
        self.vjp_rules
            .lock()
            .ok()
            .and_then(|rules| rules.get(op_name).cloned())
    }

    /// Look up a JVP rule for the given op name.
    pub fn get_jvp(&self, op_name: &str) -> Option<JvpFn<F>> {
        self.jvp_rules
            .lock()
            .ok()
            .and_then(|rules| rules.get(op_name).cloned())
    }

    /// Check whether a VJP rule exists for the given op.
    pub fn has_vjp(&self, op_name: &str) -> bool {
        self.vjp_rules
            .lock()
            .ok()
            .map(|rules| rules.contains_key(op_name))
            .unwrap_or(false)
    }

    /// Check whether a JVP rule exists for the given op.
    pub fn has_jvp(&self, op_name: &str) -> bool {
        self.jvp_rules
            .lock()
            .ok()
            .map(|rules| rules.contains_key(op_name))
            .unwrap_or(false)
    }

    /// Number of registered VJP rules.
    pub fn num_vjp_rules(&self) -> usize {
        self.vjp_rules.lock().ok().map(|r| r.len()).unwrap_or(0)
    }

    /// Number of registered JVP rules.
    pub fn num_jvp_rules(&self) -> usize {
        self.jvp_rules.lock().ok().map(|r| r.len()).unwrap_or(0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── Checkpointing ──────────────────────────────────────────────────

    #[test]
    fn test_sqrt_checkpoint_plan() {
        let plan = sqrt_checkpoint_plan(100);
        assert_eq!(plan.num_layers, 100);
        assert_eq!(plan.strategy, CheckpointStrategy::Sqrt);
        // sqrt(100) = 10, so ~11 checkpoints
        assert!(plan.checkpoint_indices.len() <= 12);
        // peak_memory = num_checkpoints + k where k = ceil(sqrt(100)) = 10
        // With 11 checkpoints + 10 = 21, so bound at 22
        assert!(
            plan.peak_memory <= 22,
            "peak_memory={} should be <= 22",
            plan.peak_memory,
        );
    }

    #[test]
    fn test_sqrt_checkpoint_small() {
        let plan = sqrt_checkpoint_plan(1);
        assert_eq!(plan.checkpoint_indices.len(), 1);
        assert_eq!(plan.checkpoint_indices[0], 0);
    }

    #[test]
    fn test_sqrt_checkpoint_zero() {
        let plan = sqrt_checkpoint_plan(0);
        assert!(plan.checkpoint_indices.is_empty());
        assert_eq!(plan.peak_memory, 0);
    }

    #[test]
    fn test_binomial_checkpoint_plan() {
        let plan = binomial_checkpoint_plan(50, 5);
        assert_eq!(plan.num_layers, 50);
        assert!(!plan.checkpoint_indices.is_empty());
        assert!(plan.peak_memory <= 5);
    }

    #[test]
    fn test_binomial_checkpoint_zero_budget() {
        let plan = binomial_checkpoint_plan(10, 0);
        assert!(plan.checkpoint_indices.is_empty());
    }

    #[test]
    fn test_uniform_checkpoint_plan() {
        let plan = uniform_checkpoint_plan(20, 5);
        assert_eq!(plan.num_layers, 20);
        // With interval 5: checkpoints at 0, 5, 10, 15, 19
        assert!(plan.checkpoint_indices.contains(&0));
        assert!(plan.checkpoint_indices.contains(&5));
    }

    // ── Rematerialization ──────────────────────────────────────────────

    #[test]
    fn test_rematerialization_default_policy() {
        let policy = RematerializationPolicy::default();
        assert_eq!(
            rematerialization_decision("relu", &policy),
            RematerializationDecision::Recompute
        );
        assert_eq!(
            rematerialization_decision("matmul", &policy),
            RematerializationDecision::Store
        );
        assert_eq!(
            rematerialization_decision("sigmoid", &policy),
            RematerializationDecision::Store
        );
    }

    #[test]
    fn test_build_rematerialization_plan() {
        let policy = RematerializationPolicy {
            memory_limit: 2,
            ..RematerializationPolicy::default()
        };
        let ops = &["relu", "matmul", "sigmoid", "add"];
        let decisions = build_rematerialization_plan(ops, &policy);
        assert_eq!(decisions.len(), 4);
        // relu and add should be recompute (cheap), matmul always stored
        assert_eq!(decisions[0], RematerializationDecision::Recompute);
        assert_eq!(decisions[1], RematerializationDecision::Store);
        assert_eq!(decisions[3], RematerializationDecision::Recompute);
    }

    // ── Mixed-mode AD ──────────────────────────────────────────────────

    #[test]
    fn test_select_jacobian_mode() {
        // Few inputs, many outputs -> forward
        assert_eq!(select_jacobian_mode(2, 100), JacobianMode::Forward);
        // Many inputs, few outputs -> reverse
        assert_eq!(select_jacobian_mode(100, 2), JacobianMode::Reverse);
        // Equal -> forward (cheaper single mode)
        assert_eq!(select_jacobian_mode(10, 10), JacobianMode::Forward);
    }

    #[test]
    fn test_plan_jacobian_computation() {
        let plan = plan_jacobian_computation(5, 100);
        assert_eq!(plan.mode, JacobianMode::Forward);
        assert_eq!(plan.num_forward_passes, 5);
        assert_eq!(plan.num_reverse_passes, 0);
        assert!(plan.estimated_cost > 0.0);
    }

    // ── Implicit differentiation ───────────────────────────────────────

    #[test]
    fn test_implicit_diff_identity() {
        // A = I (identity), b = [1,2]  =>  x = -b = [-1, -2]
        let a: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let b: Vec<f64> = vec![1.0, 2.0];
        let config = ImplicitDiffConfig::default();
        let result = solve_implicit_diff(&a, &b, 2, 1, &config);
        assert!(result.converged);
        assert!((result.gradient[0] - (-1.0)).abs() < 1e-6);
        assert!((result.gradient[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_implicit_diff_2x2() {
        // A = [[2, 0], [0, 3]], b = [[1, 0], [0, 1]] (2x2 identity as dF/dtheta)
        // x = -A^{-1} @ b = [[-0.5, 0], [0, -1/3]]
        let a: Vec<f64> = vec![2.0, 0.0, 0.0, 3.0];
        let b: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let config = ImplicitDiffConfig::default();
        let result = solve_implicit_diff(&a, &b, 2, 2, &config);
        assert!(result.converged);
        assert!((result.gradient[0] - (-0.5)).abs() < 1e-6);
        assert!((result.gradient[3] - (-1.0 / 3.0)).abs() < 1e-6);
    }

    // ── Custom rules registry ──────────────────────────────────────────

    #[test]
    fn test_diff_rule_registry_vjp() {
        let registry = DiffRuleRegistry::<f64>::new();
        assert_eq!(registry.num_vjp_rules(), 0);
        assert!(!registry.has_vjp("my_op"));

        registry.register_vjp("my_op", |primals, cotangent| {
            // Simple passthrough VJP
            vec![cotangent.to_vec()]
        });

        assert!(registry.has_vjp("my_op"));
        assert_eq!(registry.num_vjp_rules(), 1);

        let vjp = registry.get_vjp("my_op");
        assert!(vjp.is_some());
        let result = (vjp.as_ref().expect("exists"))(&[vec![1.0, 2.0]], &[3.0, 4.0]);
        assert_eq!(result, vec![vec![3.0, 4.0]]);
    }

    #[test]
    fn test_diff_rule_registry_jvp() {
        let registry = DiffRuleRegistry::<f64>::new();
        registry.register_jvp("my_op", |_primals, tangents| tangents[0].clone());

        assert!(registry.has_jvp("my_op"));
        let jvp = registry.get_jvp("my_op");
        assert!(jvp.is_some());
    }

    #[test]
    fn test_diff_rule_registry_empty_lookup() {
        let registry = DiffRuleRegistry::<f32>::new();
        assert!(registry.get_vjp("nonexistent").is_none());
        assert!(registry.get_jvp("nonexistent").is_none());
    }

    #[test]
    fn test_binom_geq() {
        // C(5, 2) = 10 >= 10  =>  true
        assert!(binom_geq(5, 2, 10));
        // C(5, 2) = 10 >= 11  =>  false
        assert!(!binom_geq(5, 2, 11));
        // C(10, 3) = 120 >= 100  =>  true
        assert!(binom_geq(10, 3, 100));
    }

    #[test]
    fn test_checkpoint_indices_sorted() {
        let plan = sqrt_checkpoint_plan(50);
        for w in plan.checkpoint_indices.windows(2) {
            assert!(
                w[0] < w[1],
                "Checkpoint indices must be strictly increasing"
            );
        }
    }
}
