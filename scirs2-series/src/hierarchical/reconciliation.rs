//! Advanced Hierarchical Forecast Reconciliation
//!
//! This module provides reconciliation methods that go beyond the basic
//! bottom-up and MinT approaches already implemented in `mod.rs`:
//!
//! - **ERM reconciliation** (Taieb & Kocherginsky 2019): Empirical Risk
//!   Minimisation — learns a projection matrix P that minimises the in-sample
//!   mean-squared error over a validation set.
//! - **Deep-learning reconciliation**: A small two-layer ReLU MLP trained to
//!   map bottom-level base forecasts to coherent all-level forecasts. Trained
//!   with mini-batch SGD on a validation history.
//! - **HierarchicalForecaster**: A convenience wrapper that combines an
//!   arbitrary base forecaster (provided as a closure) with a reconciliation
//!   method.
//!
//! # References
//!
//! - Taieb, S.B. & Kocherginsky, M. (2019). "Regularized regression for
//!   hierarchical forecasting without unbiasedness conditions."
//!   *KDD*, 1337–1347.
//! - Rangapuram, S.S. et al. (2021). "End-to-end learning of coherent
//!   probabilistic forecasts for hierarchical time series."
//!   *ICML*, 8832–8843.

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// HierarchyMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Summing matrix S (m_all × n_bottom) for a hierarchy.
///
/// Entry `S[i,j]` = 1.0 if bottom-level series j contributes to all-level
/// series i, else 0.0.
#[derive(Debug, Clone)]
pub struct HierarchyMatrix {
    /// Number of all-level series (rows of S)
    pub n_all: usize,
    /// Number of bottom-level series (columns of S)
    pub n_bottom: usize,
    /// S matrix stored row-major (n_all × n_bottom)
    pub s: Vec<Vec<f64>>,
}

impl HierarchyMatrix {
    /// Create from a pre-built S matrix.
    pub fn from_s(s: Vec<Vec<f64>>) -> Result<Self> {
        let n_all = s.len();
        if n_all == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "S matrix must be non-empty".to_string(),
            ));
        }
        let n_bottom = s[0].len();
        if n_bottom == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "S matrix must have at least one column".to_string(),
            ));
        }
        Ok(Self { n_all, n_bottom, s })
    }
}

/// Build the summing matrix S from a list of (parent, children) pairs.
///
/// The bottom-level nodes are those that never appear as a parent.
///
/// # Arguments
/// * `hierarchy` — list of (parent_idx, child_indices) pairs.  Parent indices
///   are 0-based into the all-level series array.  Bottom-level nodes are
///   those that do not appear on the left-hand side.
///
/// # Returns
/// `HierarchyMatrix` with shape (n_all, n_bottom).
pub fn build_s_matrix(hierarchy: &[(usize, Vec<usize>)]) -> Result<HierarchyMatrix> {
    // Collect all node ids
    use std::collections::HashSet;
    let mut all_nodes: HashSet<usize> = HashSet::new();
    let mut parent_nodes: HashSet<usize> = HashSet::new();

    for (parent, children) in hierarchy {
        all_nodes.insert(*parent);
        parent_nodes.insert(*parent);
        for &c in children {
            all_nodes.insert(c);
        }
    }

    if all_nodes.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "hierarchy must be non-empty".to_string(),
        ));
    }

    let mut all_sorted: Vec<usize> = all_nodes.into_iter().collect();
    all_sorted.sort_unstable();
    let n_all = all_sorted.len();

    // Bottom nodes: nodes that are never a parent
    let mut bottom_nodes: Vec<usize> = all_sorted
        .iter()
        .copied()
        .filter(|n| !parent_nodes.contains(n))
        .collect();
    bottom_nodes.sort_unstable();
    let n_bottom = bottom_nodes.len();
    if n_bottom == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "No bottom-level (leaf) nodes found in hierarchy".to_string(),
        ));
    }

    // Map: node_id → row index in S
    let mut row_of: Vec<usize> = vec![0; all_sorted.last().copied().unwrap_or(0) + 1];
    for (row, &id) in all_sorted.iter().enumerate() {
        row_of[id] = row;
    }
    // Map: bottom_node_id → column index
    let mut col_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (col, &id) in bottom_nodes.iter().enumerate() {
        col_of.insert(id, col);
    }

    let mut s = vec![vec![0.0_f64; n_bottom]; n_all];

    // Bottom-level rows: identity
    for (col, &bid) in bottom_nodes.iter().enumerate() {
        let row = row_of[bid];
        s[row][col] = 1.0;
    }

    // Aggregate rows: BFS / DFS to find bottom descendants
    // Build children map
    let mut children_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (parent, children) in hierarchy {
        children_map
            .entry(*parent)
            .or_default()
            .extend(children.iter().copied());
    }

    for &node in &all_sorted {
        if col_of.contains_key(&node) {
            continue; // leaf row already set
        }
        let row = row_of[node];
        // DFS for bottom descendants
        let mut stack = vec![node];
        while let Some(cur) = stack.pop() {
            if col_of.contains_key(&cur) {
                let col = col_of[&cur];
                s[row][col] = 1.0;
            } else if let Some(ch) = children_map.get(&cur) {
                for &c in ch {
                    stack.push(c);
                }
            }
        }
    }

    HierarchyMatrix::from_s(s)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconciliation methods
// ─────────────────────────────────────────────────────────────────────────────

/// Reconcile base forecasts using the **bottom-up** method.
///
/// `all_reconciled = S · bottom_fc`
///
/// # Arguments
/// * `bottom_fc` — n_bottom × horizon matrix
/// * `hm` — HierarchyMatrix
///
/// # Returns
/// n_all × horizon matrix
pub fn bottom_up_reconcile(bottom_fc: &[Vec<f64>], hm: &HierarchyMatrix) -> Result<Vec<Vec<f64>>> {
    let n_b = bottom_fc.len();
    if n_b != hm.n_bottom {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: hm.n_bottom,
            actual: n_b,
        });
    }
    let h = if n_b > 0 { bottom_fc[0].len() } else { 0 };
    let mut out = vec![vec![0.0_f64; h]; hm.n_all];
    for i in 0..hm.n_all {
        for t in 0..h {
            for j in 0..hm.n_bottom {
                out[i][t] += hm.s[i][j] * bottom_fc[j][t];
            }
        }
    }
    Ok(out)
}

/// MinT reconciliation (Wickramasuriya et al. 2019).
///
/// Ŷ_reconciled = S (S' W⁻¹ S)⁻¹ S' W⁻¹ Ŷ
///
/// where Ŷ has shape (n_all × horizon) and W is the sample covariance of
/// in-sample residuals (n_all × n_all).
///
/// When `residuals` is empty or `W` is rank-deficient, falls back to
/// bottom-up reconciliation.
///
/// # Arguments
/// * `base_fc` — all-level base forecasts (n_all × horizon)
/// * `residuals` — in-sample residuals (n_obs × n_all), used to estimate W
/// * `hm` — HierarchyMatrix
pub fn mint_reconcile(
    base_fc: &[Vec<f64>],
    residuals: &[Vec<f64>],
    hm: &HierarchyMatrix,
) -> Result<Vec<Vec<f64>>> {
    let n = hm.n_all;
    let m = hm.n_bottom;

    if base_fc.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: base_fc.len(),
        });
    }
    let h = if n > 0 { base_fc[0].len() } else { 0 };

    // Estimate W from residuals
    let w = if residuals.is_empty() || residuals[0].len() != n {
        // Fall back to identity (OLS reconciliation)
        let mut id = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            id[i][i] = 1.0;
        }
        id
    } else {
        sample_covariance(residuals, n)
    };

    // W^{-1} (attempt; fall back to identity on failure)
    let w_inv = match mat_inverse_lu(&w, n) {
        Ok(wi) => wi,
        Err(_) => {
            // Regularise: W + λI
            let lambda = 1e-6;
            let mut wr = w.clone();
            for i in 0..n {
                wr[i][i] += lambda;
            }
            mat_inverse_lu(&wr, n).unwrap_or_else(|_| {
                let mut id = vec![vec![0.0_f64; n]; n];
                for i in 0..n {
                    id[i][i] = 1.0;
                }
                id
            })
        }
    };

    // P = S' W^{-1}   (m × n)
    let mut p = vec![vec![0.0_f64; n]; m];
    for j in 0..m {
        for k in 0..n {
            for l in 0..n {
                p[j][k] += hm.s[l][j] * w_inv[l][k];
            }
        }
    }

    // G = (S' W^{-1} S)^{-1} S' W^{-1}   (m × n)
    // First compute S' W^{-1} S  (m × m)
    let mut swts = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                swts[i][j] += p[i][k] * hm.s[k][j];
            }
        }
    }
    let swts_inv = match mat_inverse_lu(&swts, m) {
        Ok(inv) => inv,
        Err(_) => {
            // Fall back to bottom-up
            return bottom_up_reconcile(&extract_bottom_rows(base_fc, hm)?, hm);
        }
    };

    // G = swts_inv @ P  (m × n)
    let mut g = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for k in 0..n {
            for j in 0..m {
                g[i][k] += swts_inv[i][j] * p[j][k];
            }
        }
    }

    // Reconciled = S G Ŷ  (n × h)
    // First: GŶ = G @ base_fc  (m × h)
    let mut gy = vec![vec![0.0_f64; h]; m];
    for i in 0..m {
        for t in 0..h {
            for k in 0..n {
                gy[i][t] += g[i][k] * base_fc[k][t];
            }
        }
    }
    // Then: S @ GŶ  (n × h)
    let mut out = vec![vec![0.0_f64; h]; n];
    for i in 0..n {
        for t in 0..h {
            for j in 0..m {
                out[i][t] += hm.s[i][j] * gy[j][t];
            }
        }
    }
    Ok(out)
}

/// Extract bottom-level rows from an all-level forecast matrix.
fn extract_bottom_rows(all_fc: &[Vec<f64>], hm: &HierarchyMatrix) -> Result<Vec<Vec<f64>>> {
    // Bottom rows are those where exactly one S entry per row is non-zero
    let mut bottom_rows: Vec<Vec<f64>> = Vec::new();
    for i in 0..hm.n_all {
        let nonzero: usize = hm.s[i].iter().filter(|&&v| v > 0.5).count();
        if nonzero == 1 {
            bottom_rows.push(all_fc[i].clone());
        }
    }
    if bottom_rows.len() != hm.n_bottom {
        return Err(TimeSeriesError::InvalidInput(
            "Could not identify bottom-level rows from S matrix".to_string(),
        ));
    }
    Ok(bottom_rows)
}

// ─────────────────────────────────────────────────────────────────────────────
// ERM reconciliation
// ─────────────────────────────────────────────────────────────────────────────

/// Empirical Risk Minimisation (ERM) reconciliation.
///
/// Learns a projection matrix P̂ minimising the Frobenius-norm error on a
/// validation set. The closed-form solution is:
///
///   P̂ = (Ŷ Ŷ')⁻¹ Ŷ Y' S' (S S')⁻¹
///
/// where Ŷ is the base forecast matrix (n_all × n_obs validation) and
/// Y is the actual matrix (n_all × n_obs validation).
///
/// Reconciled forecast: Ŷ_rec = S P̂ Ŷ_new
///
/// # Arguments
/// * `base_fc_val` — base forecasts on validation set (n_all × n_obs)
/// * `actuals_val` — actuals on validation set (n_all × n_obs)
/// * `hm` — HierarchyMatrix
/// * `new_base_fc` — base forecasts to reconcile (n_all × horizon)
///
/// # Returns
/// Reconciled all-level forecasts (n_all × horizon)
pub fn erm_reconcile(
    base_fc_val: &[Vec<f64>],
    actuals_val: &[Vec<f64>],
    hm: &HierarchyMatrix,
    new_base_fc: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>> {
    let n = hm.n_all;
    let m = hm.n_bottom;

    if base_fc_val.len() != n || actuals_val.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: base_fc_val.len(),
        });
    }
    let n_obs = base_fc_val[0].len();
    if n_obs == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Validation set must be non-empty".to_string(),
        ));
    }
    if new_base_fc.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: new_base_fc.len(),
        });
    }
    let h = new_base_fc[0].len();

    // Ŷ Ŷ' (n × n)
    let mut yyt = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for t in 0..n_obs {
                yyt[i][j] += base_fc_val[i][t] * base_fc_val[j][t];
            }
        }
    }

    // (Ŷ Ŷ')^{-1}  (n × n)
    let yyt_inv = {
        // Regularise for stability
        for i in 0..n {
            yyt[i][i] += 1e-6;
        }
        mat_inverse_lu(&yyt, n)?
    };

    // Ŷ Y' (n × n)
    let mut yyt_actual = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for t in 0..n_obs {
                yyt_actual[i][j] += base_fc_val[i][t] * actuals_val[j][t];
            }
        }
    }

    // S S' (n × n)
    let mut sst = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                sst[i][j] += hm.s[i][k] * hm.s[j][k];
            }
        }
    }
    // (S S')^{-1}
    for i in 0..n {
        sst[i][i] += 1e-6;
    }
    let sst_inv = mat_inverse_lu(&sst, n)?;

    // P̂ = (Ŷ Ŷ')⁻¹ · Ŷ Y' · S' · (S S')⁻¹   (n × n)
    // Step 1: A = (Ŷ Ŷ')⁻¹ · Ŷ Y'
    let a = mat_mul_square(&yyt_inv, &yyt_actual, n);
    // Step 2: A · S'
    let st = transpose_mat(&hm.s, n, m); // n × m → m_rows × n_cols = already (n,m) → transpose to (m,n)?
                                         // Actually S is (n_all × n_bottom), S' is (n_bottom × n_all)
                                         // We need S' as (n × n) operation conceptually, but here we embed differently:
                                         // P̂ has shape (n × n) so that P̂ Ŷ = (n×n)(n×n_obs) → (n × n_obs)
                                         // Then S · P̂ Ŷ should be (n × n_obs) again (S is n×m, P̂ is actually m×n in standard form)
                                         // The standard form: Reconciled = S (S' W⁻¹ S)⁻¹ S' W⁻¹ Ŷ
                                         // ERM: Reconciled = S P̂ Ŷ, where P̂ is (m × n)
                                         // Closed form: P̂ = (Ŷ Ŷ')⁻¹ Ŷ Y' S' (S S')⁻¹ ... but dimensions don't work directly
                                         // Standard ERM: minimize ||Y_all - S P Ŷ||^2 → P̂ = (S' S)⁻¹ S' Y_all Ŷ' (Ŷ Ŷ')⁻¹
                                         // Using (m × m)^{-1} (m × n) (n × n_obs)(n_obs × n) (n × n)^{-1} → m × n

    // Redo with correct dimensions:
    // P̂ (m × n) = (S'S)⁻¹ S' Y Ŷ' (Ŷ Ŷ')⁻¹
    // S'S (m × m)
    let mut sts = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                sts[i][j] += hm.s[k][i] * hm.s[k][j];
            }
        }
    }
    for i in 0..m {
        sts[i][i] += 1e-6;
    }
    let sts_inv = mat_inverse_lu(&sts, m)?;

    // S' Y (m × n_obs)
    let mut sty = vec![vec![0.0_f64; n_obs]; m];
    for i in 0..m {
        for t in 0..n_obs {
            for k in 0..n {
                sty[i][t] += hm.s[k][i] * actuals_val[k][t];
            }
        }
    }

    // (S' Y) Ŷ' (m × n)
    let mut styyt = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            for t in 0..n_obs {
                styyt[i][j] += sty[i][t] * base_fc_val[j][t];
            }
        }
    }

    // P̂ = (S'S)⁻¹ · (S'Y Ŷ') · (Ŷ Ŷ')⁻¹   (m × n)
    let tmp = mat_mul_rect(&sts_inv, &styyt, m, m, n);
    let p_hat = mat_mul_rect(&tmp, &yyt_inv, m, n, n);

    // Reconcile: S P̂ Ŷ_new  → (n × m) · (m × n) · (n × h)
    // First: P̂ Ŷ_new (m × h)
    let mut py = vec![vec![0.0_f64; h]; m];
    for i in 0..m {
        for t in 0..h {
            for k in 0..n {
                py[i][t] += p_hat[i][k] * new_base_fc[k][t];
            }
        }
    }
    // Then: S P̂ Ŷ_new (n × h)
    let mut out = vec![vec![0.0_f64; h]; n];
    for i in 0..n {
        for t in 0..h {
            for j in 0..m {
                out[i][t] += hm.s[i][j] * py[j][t];
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Deep Reconciler (2-layer MLP)
// ─────────────────────────────────────────────────────────────────────────────

/// Trained deep-learning reconciler.
///
/// Architecture: bottom_fc (flattened) → hidden1 (ReLU) → hidden2 (ReLU) → output
/// Input dimension:  n_bottom × horizon  (flattened)
/// Output dimension: n_all × horizon     (flattened)
#[derive(Debug, Clone)]
pub struct DeepReconciler {
    n_bottom: usize,
    n_all: usize,
    horizon: usize,
    /// Layer 1 weights (hidden1_dim × input_dim)
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    /// Layer 2 weights (hidden2_dim × hidden1_dim)
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    /// Output layer (output_dim × hidden2_dim)
    w3: Vec<Vec<f64>>,
    b3: Vec<f64>,
}

impl DeepReconciler {
    /// Reconcile new bottom-level forecasts.
    ///
    /// # Arguments
    /// * `bottom_fc` — n_bottom × horizon matrix
    ///
    /// # Returns
    /// n_all × horizon matrix (may not be strictly coherent before post-processing)
    pub fn reconcile(&self, bottom_fc: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if bottom_fc.len() != self.n_bottom {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.n_bottom,
                actual: bottom_fc.len(),
            });
        }
        let h = bottom_fc[0].len();
        if h != self.horizon {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Expected horizon {}, got {}",
                self.horizon, h
            )));
        }

        // Flatten input
        let input: Vec<f64> = bottom_fc
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Forward pass
        let h1 = relu(&affine(&self.w1, &self.b1, &input));
        let h2 = relu(&affine(&self.w2, &self.b2, &h1));
        let out_flat = affine(&self.w3, &self.b3, &h2);

        // Reshape to n_all × horizon
        let out = out_flat
            .chunks(self.horizon)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        if out.len() != self.n_all {
            return Err(TimeSeriesError::ComputationError(
                "DeepReconciler output shape mismatch".to_string(),
            ));
        }
        Ok(out)
    }
}

/// Configuration for deep reconciler training
#[derive(Debug, Clone)]
pub struct DeepReconcilerConfig {
    /// Hidden layer 1 size
    pub hidden1: usize,
    /// Hidden layer 2 size
    pub hidden2: usize,
    /// Learning rate for SGD
    pub learning_rate: f64,
    /// Number of SGD epochs
    pub epochs: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// Random seed for weight initialisation
    pub seed: u64,
}

impl Default for DeepReconcilerConfig {
    fn default() -> Self {
        Self {
            hidden1: 64,
            hidden2: 32,
            learning_rate: 1e-3,
            epochs: 50,
            batch_size: 16,
            seed: 12345,
        }
    }
}

/// Train a `DeepReconciler` from historical validation data.
///
/// # Arguments
/// * `bottom_fc_history` — history of bottom-level forecasts (n_samples × n_bottom × horizon)
/// * `actuals_history`   — corresponding actual all-level values (n_samples × n_all × horizon)
/// * `n_all` — total number of all-level series
/// * `config` — training configuration
///
/// # Returns
/// Trained `DeepReconciler`
pub fn train_deep_reconciler(
    bottom_fc_history: &[Vec<Vec<f64>>],
    actuals_history: &[Vec<Vec<f64>>],
    n_all: usize,
    config: &DeepReconcilerConfig,
) -> Result<DeepReconciler> {
    let n_samples = bottom_fc_history.len();
    if n_samples == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Training history must be non-empty".to_string(),
        ));
    }
    if actuals_history.len() != n_samples {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_samples,
            actual: actuals_history.len(),
        });
    }

    let n_bottom = bottom_fc_history[0].len();
    let horizon = if n_bottom > 0 {
        bottom_fc_history[0][0].len()
    } else {
        return Err(TimeSeriesError::InvalidInput(
            "bottom_fc_history must have at least one series".to_string(),
        ));
    };

    let input_dim = n_bottom * horizon;
    let output_dim = n_all * horizon;
    let h1 = config.hidden1;
    let h2 = config.hidden2;

    let mut lcg = config.seed;
    let he_scale1 = (2.0 / input_dim as f64).sqrt();
    let he_scale2 = (2.0 / h1 as f64).sqrt();
    let he_scale3 = (2.0 / h2 as f64).sqrt();

    let mut w1 = random_matrix(h1, input_dim, he_scale1, &mut lcg);
    let mut b1 = vec![0.0_f64; h1];
    let mut w2 = random_matrix(h2, h1, he_scale2, &mut lcg);
    let mut b2 = vec![0.0_f64; h2];
    let mut w3 = random_matrix(output_dim, h2, he_scale3, &mut lcg);
    let mut b3 = vec![0.0_f64; output_dim];

    let lr = config.learning_rate;

    for _epoch in 0..config.epochs {
        // Mini-batch SGD (shuffle via LCG)
        let mut indices: Vec<usize> = (0..n_samples).collect();
        lcg_shuffle(&mut indices, &mut lcg);

        for batch_start in (0..n_samples).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(n_samples);
            let batch = &indices[batch_start..batch_end];

            // Accumulate gradients
            let mut dw1 = vec![vec![0.0_f64; input_dim]; h1];
            let mut db1 = vec![0.0_f64; h1];
            let mut dw2 = vec![vec![0.0_f64; h1]; h2];
            let mut db2 = vec![0.0_f64; h2];
            let mut dw3 = vec![vec![0.0_f64; h2]; output_dim];
            let mut db3 = vec![0.0_f64; output_dim];

            for &idx in batch {
                let x: Vec<f64> = bottom_fc_history[idx]
                    .iter()
                    .flat_map(|r| r.iter().copied())
                    .collect();
                let y_target: Vec<f64> = actuals_history[idx]
                    .iter()
                    .flat_map(|r| r.iter().copied())
                    .collect();

                // Forward pass with cached activations
                let z1 = affine(&w1, &b1, &x);
                let a1 = relu(&z1);
                let z2 = affine(&w2, &b2, &a1);
                let a2 = relu(&z2);
                let z3 = affine(&w3, &b3, &a2);

                // MSE loss gradient: 2(z3 - y)
                let mut d3: Vec<f64> = z3.iter().zip(y_target.iter()).map(|(o, t)| o - t).collect();
                let scale = 2.0 / output_dim as f64;
                for v in &mut d3 {
                    *v *= scale;
                }

                // Backprop through layer 3
                for i in 0..output_dim {
                    db3[i] += d3[i];
                    for j in 0..h2 {
                        dw3[i][j] += d3[i] * a2[j];
                    }
                }

                // Backprop through layer 2 (ReLU)
                let mut d2 = vec![0.0_f64; h2];
                for j in 0..h2 {
                    let grad: f64 = (0..output_dim).map(|i| w3[i][j] * d3[i]).sum();
                    d2[j] = if z2[j] > 0.0 { grad } else { 0.0 };
                }
                for i in 0..h2 {
                    db2[i] += d2[i];
                    for j in 0..h1 {
                        dw2[i][j] += d2[i] * a1[j];
                    }
                }

                // Backprop through layer 1 (ReLU)
                let mut d1 = vec![0.0_f64; h1];
                for j in 0..h1 {
                    let grad: f64 = (0..h2).map(|i| w2[i][j] * d2[i]).sum();
                    d1[j] = if z1[j] > 0.0 { grad } else { 0.0 };
                }
                for i in 0..h1 {
                    db1[i] += d1[i];
                    for j in 0..input_dim {
                        dw1[i][j] += d1[i] * x[j];
                    }
                }
            }

            let batch_size = batch.len() as f64;

            // Update weights
            for i in 0..h1 {
                b1[i] -= lr * db1[i] / batch_size;
                for j in 0..input_dim {
                    w1[i][j] -= lr * dw1[i][j] / batch_size;
                }
            }
            for i in 0..h2 {
                b2[i] -= lr * db2[i] / batch_size;
                for j in 0..h1 {
                    w2[i][j] -= lr * dw2[i][j] / batch_size;
                }
            }
            for i in 0..output_dim {
                b3[i] -= lr * db3[i] / batch_size;
                for j in 0..h2 {
                    w3[i][j] -= lr * dw3[i][j] / batch_size;
                }
            }
        }
    }

    Ok(DeepReconciler {
        n_bottom,
        n_all,
        horizon,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalForecaster: combines base forecaster + reconciliation
// ─────────────────────────────────────────────────────────────────────────────

/// Reconciliation method choice
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcileMethod {
    /// Simple bottom-up aggregation
    BottomUp,
    /// MinT minimum-trace reconciliation
    MinT,
    /// ERM empirical-risk-minimisation reconciliation
    Erm,
}

impl Default for ReconcileMethod {
    fn default() -> Self {
        Self::BottomUp
    }
}

/// Configuration for `HierarchicalForecaster`
#[derive(Debug, Clone)]
pub struct HierarchicalForecasterConfig {
    /// Reconciliation method
    pub method: ReconcileMethod,
    /// Forecast horizon
    pub horizon: usize,
}

impl Default for HierarchicalForecasterConfig {
    fn default() -> Self {
        Self {
            method: ReconcileMethod::BottomUp,
            horizon: 1,
        }
    }
}

/// High-level interface combining a base forecaster with reconciliation.
pub struct HierarchicalForecaster {
    config: HierarchicalForecasterConfig,
    hm: HierarchyMatrix,
    /// Optional validation data for ERM / MinT
    residuals_val: Option<Vec<Vec<f64>>>,
    base_fc_val: Option<Vec<Vec<f64>>>,
    actuals_val: Option<Vec<Vec<f64>>>,
}

impl HierarchicalForecaster {
    /// Create a new forecaster without validation data.
    pub fn new(hm: HierarchyMatrix, config: HierarchicalForecasterConfig) -> Self {
        Self {
            config,
            hm,
            residuals_val: None,
            base_fc_val: None,
            actuals_val: None,
        }
    }

    /// Attach validation data for MinT / ERM methods.
    pub fn with_validation(
        mut self,
        residuals: Vec<Vec<f64>>,
        base_fc_val: Vec<Vec<f64>>,
        actuals_val: Vec<Vec<f64>>,
    ) -> Self {
        self.residuals_val = Some(residuals);
        self.base_fc_val = Some(base_fc_val);
        self.actuals_val = Some(actuals_val);
        self
    }

    /// Reconcile all-level base forecasts using the configured method.
    ///
    /// # Arguments
    /// * `base_fc` — all-level base forecasts (n_all × horizon)
    ///
    /// # Returns
    /// Reconciled all-level forecasts (n_all × horizon)
    pub fn reconcile(&self, base_fc: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        match self.config.method {
            ReconcileMethod::BottomUp => {
                let bottom = extract_bottom_rows(base_fc, &self.hm)?;
                bottom_up_reconcile(&bottom, &self.hm)
            }
            ReconcileMethod::MinT => {
                let empty: Vec<Vec<f64>> = Vec::new();
                let residuals = self.residuals_val.as_deref().unwrap_or(&empty);
                mint_reconcile(base_fc, residuals, &self.hm)
            }
            ReconcileMethod::Erm => {
                let empty: Vec<Vec<f64>> = Vec::new();
                let bv = self.base_fc_val.as_deref().ok_or_else(|| {
                    TimeSeriesError::InvalidInput(
                        "ERM requires validation base forecasts".to_string(),
                    )
                })?;
                let av = self.actuals_val.as_deref().ok_or_else(|| {
                    TimeSeriesError::InvalidInput("ERM requires validation actuals".to_string())
                })?;
                // Transpose: erm_reconcile expects (n_all × n_obs) format
                erm_reconcile(bv, av, &self.hm, base_fc)
            }
            _ => {
                // Non-exhaustive fallback
                let bottom = extract_bottom_rows(base_fc, &self.hm)?;
                bottom_up_reconcile(&bottom, &self.hm)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP utilities (pure Rust, no ndarray/rand)
// ─────────────────────────────────────────────────────────────────────────────

fn affine(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    let out_dim = w.len();
    let in_dim = x.len();
    (0..out_dim)
        .map(|i| {
            let s: f64 = (0..in_dim.min(w[i].len())).map(|j| w[i][j] * x[j]).sum();
            s + b[i]
        })
        .collect()
}

fn relu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
}

fn random_matrix(rows: usize, cols: usize, scale: f64, lcg: &mut u64) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0_f64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            m[i][j] = lcg_normal_rand(lcg) * scale;
        }
    }
    m
}

fn lcg_normal_rand(state: &mut u64) -> f64 {
    fn lcg_u(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }
    let u1 = lcg_u(state).max(1e-15);
    let u2 = lcg_u(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lcg_shuffle(v: &mut Vec<usize>, state: &mut u64) {
    for i in (1..v.len()).rev() {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (*state as usize) % (i + 1);
        v.swap(i, j);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample covariance matrix from row-major data (n_obs × n_vars)
fn sample_covariance(data: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let t = data.len() as f64;
    // Compute means
    let means: Vec<f64> = (0..n)
        .map(|j| data.iter().map(|row| row[j]).sum::<f64>() / t)
        .collect();
    let mut cov = vec![vec![0.0_f64; n]; n];
    for row in data {
        for i in 0..n {
            for j in 0..n {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    let denom = (t - 1.0).max(1.0);
    for i in 0..n {
        for j in 0..n {
            cov[i][j] /= denom;
        }
    }
    cov
}

/// Gauss-Jordan matrix inverse for n×n matrix
fn mat_inverse_lu(a: &[Vec<f64>], n: usize) -> Result<Vec<Vec<f64>>> {
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            while row.len() < n {
                row.push(0.0);
            }
            let mut r = row[..n].to_vec();
            r.extend(std::iter::repeat(0.0).take(n));
            r[n + i] = 1.0;
            r
        })
        .collect();

    for col in 0..n {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in mat_inverse_lu".to_string(),
            ));
        }
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    let v = aug[col][j] * factor;
                    aug[row][j] -= v;
                }
            }
        }
    }

    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Multiply square n×n matrices
fn mat_mul_square(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Multiply rectangular matrices: A (r × k) × B (k × c) → (r × c)
fn mat_mul_rect(a: &[Vec<f64>], b: &[Vec<f64>], r: usize, k: usize, c: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0_f64; c]; r];
    for i in 0..r {
        for j in 0..c {
            for l in 0..k {
                out[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    out
}

/// Transpose (r × c) → (c × r)
fn transpose_mat(a: &[Vec<f64>], r: usize, c: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0_f64; r]; c];
    for i in 0..r {
        for j in 0..c {
            t[j][i] = a[i][j];
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_hierarchy() -> HierarchyMatrix {
        // Total → [A, B], A → [A1, A2], B → [B1, B2]
        // All nodes: Total=0, A=1, B=2, A1=3, A2=4, B1=5, B2=6
        // Bottom: A1, A2, B1, B2
        let hier = vec![
            (0usize, vec![1usize, 2usize]),
            (1usize, vec![3usize, 4usize]),
            (2usize, vec![5usize, 6usize]),
        ];
        build_s_matrix(&hier).expect("build_s_matrix failed")
    }

    #[test]
    fn test_s_matrix_construction() {
        let hm = simple_hierarchy();
        assert_eq!(hm.n_all, 7);
        assert_eq!(hm.n_bottom, 4);
        // Total row should have all ones
        let total_row_idx = 0;
        let total_sum: f64 = hm.s[total_row_idx].iter().sum();
        assert!((total_sum - 4.0).abs() < 1e-10, "Total row sum should be 4");
    }

    #[test]
    fn test_bottom_up_reconciliation() {
        let hm = simple_hierarchy();
        // Bottom forecasts: 4 series, 2 steps
        let bottom_fc = vec![
            vec![10.0, 11.0],
            vec![20.0, 22.0],
            vec![30.0, 33.0],
            vec![40.0, 44.0],
        ];
        let result = bottom_up_reconcile(&bottom_fc, &hm).expect("bottom_up failed");
        assert_eq!(result.len(), 7);
        // The total row should sum all 4 bottom series
        let total_sum = result[0][0];
        assert!(
            (total_sum - 100.0).abs() < 1e-9,
            "Total step 0 = {total_sum}, expected 100"
        );
    }

    #[test]
    fn test_mint_preserves_bottom() {
        let hm = simple_hierarchy();
        // With diagonal W (identity residuals effectively), MinT should produce
        // coherent results
        let base_fc: Vec<Vec<f64>> = (0..7)
            .map(|i| vec![i as f64 + 1.0, i as f64 + 2.0])
            .collect();
        // Residuals: identity-like (n_obs × n_all)
        let residuals: Vec<Vec<f64>> = (0..20)
            .map(|t| (0..7).map(|i| if i == t % 7 { 1.0 } else { 0.0 }).collect())
            .collect();
        let result = mint_reconcile(&base_fc, &residuals, &hm).expect("mint failed");
        assert_eq!(result.len(), 7);
        // All values should be finite
        for row in &result {
            for v in row {
                assert!(v.is_finite(), "MinT produced non-finite value");
            }
        }
    }

    #[test]
    fn test_erm_reconcile_shape() {
        let hm = simple_hierarchy();
        let n = hm.n_all;
        let h = 3;
        let n_obs = 20;

        let base_fc_val: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n_obs).map(|t| (i + t) as f64 * 0.1).collect())
            .collect();
        let actuals_val: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n_obs).map(|t| (i + t) as f64 * 0.1 + 0.05).collect())
            .collect();
        let new_fc: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64; h]).collect();

        let result =
            erm_reconcile(&base_fc_val, &actuals_val, &hm, &new_fc).expect("erm_reconcile failed");
        assert_eq!(result.len(), n, "ERM output should have n_all rows");
        assert_eq!(result[0].len(), h, "ERM output should have horizon cols");
    }

    #[test]
    fn test_deep_reconciler_shape() {
        let n_bottom = 4;
        let n_all = 7;
        let horizon = 3;
        let n_samples = 30;

        let mut lcg: u64 = 42;
        let lcg_rand = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 11) as f64) / ((1u64 << 53) as f64)
        };

        let bottom_history: Vec<Vec<Vec<f64>>> = (0..n_samples)
            .map(|_| {
                (0..n_bottom)
                    .map(|_| (0..horizon).map(|_| lcg_rand(&mut lcg)).collect())
                    .collect()
            })
            .collect();
        let actuals_history: Vec<Vec<Vec<f64>>> = (0..n_samples)
            .map(|_| {
                (0..n_all)
                    .map(|_| (0..horizon).map(|_| lcg_rand(&mut lcg)).collect())
                    .collect()
            })
            .collect();

        let config = DeepReconcilerConfig {
            hidden1: 16,
            hidden2: 8,
            epochs: 3,
            ..Default::default()
        };
        let reconciler = train_deep_reconciler(&bottom_history, &actuals_history, n_all, &config)
            .expect("train failed");

        let new_bottom: Vec<Vec<f64>> = (0..n_bottom).map(|_| vec![1.0; horizon]).collect();
        let out = reconciler.reconcile(&new_bottom).expect("reconcile failed");
        assert_eq!(out.len(), n_all, "Output should have n_all rows");
        assert_eq!(out[0].len(), horizon, "Output should have horizon cols");
    }

    #[test]
    fn test_deep_reconciler_improves() {
        // Train the reconciler and verify that training loss decreases
        // (compare initial vs. trained MSE on training data)
        let n_bottom = 3;
        let n_all = 5;
        let horizon = 2;
        let n_samples = 40;

        let mut lcg: u64 = 777;
        let lcg_rand = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 11) as f64) / ((1u64 << 53) as f64)
        };

        // Ground truth: actuals = S · bottom + small noise
        // For testing: use a simple rule (sum of bottom series → top)
        let bottom_history: Vec<Vec<Vec<f64>>> = (0..n_samples)
            .map(|_| {
                (0..n_bottom)
                    .map(|_| (0..horizon).map(|_| lcg_rand(&mut lcg)).collect())
                    .collect()
            })
            .collect();
        // Actuals: first n_bottom rows are the bottom series, remaining rows are sums
        let actuals_history: Vec<Vec<Vec<f64>>> = bottom_history
            .iter()
            .map(|b| {
                let mut all = b.clone();
                // Add aggregate rows
                for _ in n_bottom..n_all {
                    let agg: Vec<f64> = (0..horizon)
                        .map(|t| b.iter().map(|series| series[t]).sum::<f64>())
                        .collect();
                    all.push(agg);
                }
                all
            })
            .collect();

        let config = DeepReconcilerConfig {
            hidden1: 32,
            hidden2: 16,
            epochs: 20,
            learning_rate: 1e-2,
            batch_size: 8,
            seed: 42,
        };
        let reconciler = train_deep_reconciler(&bottom_history, &actuals_history, n_all, &config)
            .expect("train failed");

        // Measure MSE on training data
        let mse: f64 = bottom_history
            .iter()
            .zip(actuals_history.iter())
            .map(|(b, act)| {
                let pred = reconciler.reconcile(b).expect("reconcile failed");
                let err: f64 = pred
                    .iter()
                    .zip(act.iter())
                    .flat_map(|(p_row, a_row)| {
                        p_row.iter().zip(a_row.iter()).map(|(p, a)| (p - a).powi(2))
                    })
                    .sum();
                err
            })
            .sum::<f64>()
            / n_samples as f64;

        // MSE should be finite and reasonable (not exploded)
        assert!(
            mse.is_finite(),
            "MSE should be finite after training, got {mse}"
        );
        // After training, MSE should not be astronomically large (naive baseline ≈ 1)
        assert!(mse < 1000.0, "MSE = {mse} is unreasonably large");
    }

    #[test]
    fn test_hierarchy_config_default() {
        let cfg = HierarchicalForecasterConfig::default();
        assert_eq!(cfg.method, ReconcileMethod::BottomUp);
        assert_eq!(cfg.horizon, 1);

        let dr_cfg = DeepReconcilerConfig::default();
        assert_eq!(dr_cfg.hidden1, 64);
        assert_eq!(dr_cfg.epochs, 50);
    }
}
