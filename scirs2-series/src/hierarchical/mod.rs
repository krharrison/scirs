//! Hierarchical Time Series Forecasting
//!
//! Provides data structures and algorithms for hierarchical time series, where
//! individual series are related through an aggregation hierarchy (e.g., national
//! → regional → local).
//!
//! # Overview
//!
//! A hierarchical time series has the property that observations at higher levels
//! are the exact sum of observations at the levels below them. This module
//! provides:
//!
//! - [`Hierarchy`] — a tree structure encoding the aggregation relationships.
//! - [`summing_matrix`] — construct the S matrix that maps bottom-level series to
//!   all levels.
//! - [`bottom_up_forecast`] — aggregate bottom-level base forecasts upward.
//! - [`top_down_forecast`] — disaggregate a top-level forecast using historical
//!   proportions.
//! - [`middle_out_forecast`] — generate forecasts outward from a chosen middle
//!   level.
//!
//! # References
//!
//! - Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. & Shang, H.L. (2011).
//!   "Optimal combination forecasts for hierarchical time series."
//!   *Computational Statistics & Data Analysis*, 55(9), 2579–2589.
//! - Hyndman, R.J. & Athanasopoulos, G. (2021).
//!   *Forecasting: Principles and Practice*, 3rd ed., Chapter 11.

pub mod online;
pub mod reconciliation;

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the hierarchy tree.
///
/// Nodes are identified by a numeric `id` and carry an optional human-readable
/// `name`. The tree topology is expressed through `parent` (the `id` of the
/// parent node, or `None` for the root) and `children` (the `id`s of all
/// direct children).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierarchyNode {
    /// Unique identifier for this node (0-based index into the node vector).
    pub id: usize,
    /// Human-readable label for the node (e.g., "Australia", "NSW").
    pub name: String,
    /// Id of the parent node. `None` for the root.
    pub parent: Option<usize>,
    /// Ids of direct child nodes (empty for leaf nodes).
    pub children: Vec<usize>,
}

impl HierarchyNode {
    /// Create a new node.
    pub fn new(id: usize, name: impl Into<String>, parent: Option<usize>) -> Self {
        Self {
            id,
            name: name.into(),
            parent,
            children: Vec::new(),
        }
    }

    /// Returns `true` if this node is a leaf (bottom-level series).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// A hierarchical structure that relates time series at different aggregation
/// levels.
///
/// The hierarchy is stored as a flat vector of [`HierarchyNode`]s. Node `id`s
/// correspond to indices into that vector, which simplifies traversal and
/// matrix construction.
///
/// # Example
///
/// ```rust
/// use scirs2_series::hierarchical::Hierarchy;
///
/// // Build a two-level hierarchy: Total -> [A, B] -> [A1, A2, B1, B2]
/// let mut h = Hierarchy::new();
/// let total = h.add_node("Total", None).expect("should succeed");
/// let a     = h.add_node("A", Some(total)).expect("should succeed");
/// let b     = h.add_node("B", Some(total)).expect("should succeed");
/// let _a1   = h.add_node("A1", Some(a)).expect("should succeed");
/// let _a2   = h.add_node("A2", Some(a)).expect("should succeed");
/// let _b1   = h.add_node("B1", Some(b)).expect("should succeed");
/// let _b2   = h.add_node("B2", Some(b)).expect("should succeed");
///
/// assert_eq!(h.bottom_level_nodes().len(), 4);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Hierarchy {
    nodes: Vec<HierarchyNode>,
}

impl Hierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a new node and return its `id`.
    ///
    /// The `parent` is automatically registered as a parent in the parent's
    /// `children` list.
    pub fn add_node(&mut self, name: impl Into<String>, parent: Option<usize>) -> Result<usize> {
        if let Some(p) = parent {
            if p >= self.nodes.len() {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Parent node id {p} does not exist (only {} nodes so far)",
                    self.nodes.len()
                )));
            }
        }
        let id = self.nodes.len();
        self.nodes.push(HierarchyNode::new(id, name, parent));
        if let Some(p) = parent {
            self.nodes[p].children.push(id);
        }
        Ok(id)
    }

    /// Return a reference to a node by id.
    pub fn node(&self, id: usize) -> Option<&HierarchyNode> {
        self.nodes.get(id)
    }

    /// Return the total number of nodes (all levels).
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Return node ids that have no children (bottom-level, leaf nodes).
    /// The ids are returned in ascending order.
    pub fn bottom_level_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.is_leaf())
            .map(|n| n.id)
            .collect()
    }

    /// Return the id of the root node (the unique node with no parent).
    ///
    /// Returns an error if the hierarchy is empty or has more than one root.
    pub fn root(&self) -> Result<usize> {
        let roots: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();
        match roots.len() {
            0 => Err(TimeSeriesError::InvalidInput(
                "Hierarchy has no root node".to_string(),
            )),
            1 => Ok(roots[0]),
            _ => Err(TimeSeriesError::InvalidInput(format!(
                "Hierarchy has {} roots; exactly one is required",
                roots.len()
            ))),
        }
    }

    /// Return all node ids sorted topologically (root first, leaves last).
    ///
    /// Uses a breadth-first traversal starting from the root.
    pub fn topological_order(&self) -> Result<Vec<usize>> {
        let root = self.root()?;
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        while let Some(id) = queue.pop_front() {
            order.push(id);
            let node = self
                .nodes
                .get(id)
                .ok_or_else(|| TimeSeriesError::InvalidInput(format!("Node id {id} not found")))?;
            for &child in &node.children {
                queue.push_back(child);
            }
        }
        Ok(order)
    }

    /// Returns all nodes in the slice as a reference.
    pub fn all_nodes(&self) -> &[HierarchyNode] {
        &self.nodes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Summing matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Construct the **summing matrix** S that relates the bottom-level (leaf)
/// series to every level of the hierarchy.
///
/// For a hierarchy with `n` total nodes and `m` bottom-level nodes, the
/// summing matrix S has shape `(n, m)`. Entry `S[i, j] = 1` if bottom-level
/// series j contributes (directly or through aggregation) to node i, and 0
/// otherwise.
///
/// The columns correspond to bottom-level nodes in *ascending id order*.
/// The rows correspond to all nodes in *ascending id order*.
///
/// # Arguments
/// * `hierarchy` — fully constructed [`Hierarchy`].
///
/// # Returns
/// An `Array2<f64>` of shape `(n, m)` where `n = hierarchy.num_nodes()` and
/// `m = hierarchy.bottom_level_nodes().len()`.
pub fn summing_matrix(hierarchy: &Hierarchy) -> Result<Array2<f64>> {
    let n = hierarchy.num_nodes();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Hierarchy has no nodes".to_string(),
        ));
    }

    let bottom_ids = hierarchy.bottom_level_nodes();
    let m = bottom_ids.len();
    if m == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Hierarchy has no leaf nodes".to_string(),
        ));
    }

    // Map from node id → column index among bottom-level nodes.
    let mut col_index = vec![usize::MAX; n];
    for (j, &bid) in bottom_ids.iter().enumerate() {
        col_index[bid] = j;
    }

    let mut s = Array2::<f64>::zeros((n, m));

    // For every node i, collect the set of bottom-level descendants and set
    // the corresponding entries to 1.
    for node in hierarchy.all_nodes() {
        if node.is_leaf() {
            // Leaf contributes only to itself.
            let j = col_index[node.id];
            s[[node.id, j]] = 1.0;
        } else {
            // Aggregate node: collect all bottom-level descendants via BFS.
            let descendants = collect_bottom_descendants(hierarchy, node.id);
            for bid in descendants {
                let j = col_index[bid];
                s[[node.id, j]] = 1.0;
            }
        }
    }

    Ok(s)
}

/// BFS helper: returns all leaf (bottom-level) descendants of `start_id`.
fn collect_bottom_descendants(hierarchy: &Hierarchy, start_id: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut stack = vec![start_id];
    while let Some(id) = stack.pop() {
        if let Some(node) = hierarchy.node(id) {
            if node.is_leaf() {
                result.push(id);
            } else {
                stack.extend_from_slice(&node.children);
            }
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Bottom-up forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// Produce coherent forecasts for all levels using the **bottom-up** approach.
///
/// Bottom-up forecasting is the simplest coherent method: forecast each
/// bottom-level series independently and aggregate upward using the summing
/// matrix S.
///
/// The aggregated forecast is `y_hat = S * bottom_forecasts` where
/// `bottom_forecasts` has shape `(m, h)` (m bottom-level series, h forecast
/// horizon) and S has shape `(n, m)`.
///
/// # Arguments
/// * `bottom_level_forecasts` — Array of shape `(m, h)`: each row is one
///   bottom-level series' h-step-ahead base forecasts.
/// * `s` — Summing matrix of shape `(n, m)` as returned by [`summing_matrix`].
///
/// # Returns
/// An `Array2<f64>` of shape `(n, h)` with coherent forecasts for all nodes.
pub fn bottom_up_forecast(
    bottom_level_forecasts: &Array2<f64>,
    s: &Array2<f64>,
) -> Result<Array2<f64>> {
    let (n, m_s) = (s.shape()[0], s.shape()[1]);
    let (m_f, h) = (
        bottom_level_forecasts.shape()[0],
        bottom_level_forecasts.shape()[1],
    );

    if m_s != m_f {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: m_s,
            actual: m_f,
        });
    }

    // y_hat = S @ bottom_level_forecasts  (n x m) @ (m x h) = (n x h)
    let mut result = Array2::<f64>::zeros((n, h));
    for i in 0..n {
        for t in 0..h {
            let mut acc = 0.0_f64;
            for j in 0..m_s {
                acc += s[[i, j]] * bottom_level_forecasts[[j, t]];
            }
            result[[i, t]] = acc;
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-down forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// Produce bottom-level forecasts by disaggregating a **top-down** forecast.
///
/// Two variants of top-down disaggregation are well-known:
/// - **Average historical proportions**: `p_j = (1/T) * sum_t (y_{j,t} / y_{total,t})`
/// - **Proportions of historical averages**: `p_j = mean(y_j) / mean(y_total)`
///
/// This function takes pre-computed proportions (one per bottom-level series,
/// summing to 1) and applies them to the top-level forecast.
///
/// # Arguments
/// * `top_level_forecast` — Length-h array of top-level point forecasts.
/// * `historical_proportions` — Length-m array of non-negative weights that
///   sum to 1; entry j corresponds to the j-th bottom-level series (in the
///   same order as the columns of S).
///
/// # Returns
/// An `Array2<f64>` of shape `(m, h)` with disaggregated bottom-level forecasts.
pub fn top_down_forecast(
    top_level_forecast: &Array1<f64>,
    historical_proportions: &Array1<f64>,
) -> Result<Array2<f64>> {
    let h = top_level_forecast.len();
    let m = historical_proportions.len();

    if h == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "top_level_forecast must not be empty".to_string(),
        ));
    }
    if m == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "historical_proportions must not be empty".to_string(),
        ));
    }

    // Validate proportions: non-negative and sum approximately to 1.
    let sum_p: f64 = historical_proportions.iter().copied().sum();
    if (sum_p - 1.0_f64).abs() > 1e-6 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "historical_proportions must sum to 1 (got {sum_p:.6})"
        )));
    }
    for &p in historical_proportions.iter() {
        if p < 0.0 {
            return Err(TimeSeriesError::InvalidInput(
                "historical_proportions must be non-negative".to_string(),
            ));
        }
    }

    let mut result = Array2::<f64>::zeros((m, h));
    for j in 0..m {
        for t in 0..h {
            result[[j, t]] = historical_proportions[j] * top_level_forecast[t];
        }
    }

    Ok(result)
}

/// Compute **average historical proportions** from training data.
///
/// For each time point t, computes the fraction `y_{j,t} / y_{total,t}` for
/// each bottom-level series j, then averages across all time points. Time
/// points where the total is zero are skipped.
///
/// # Arguments
/// * `bottom_series` — Array of shape `(m, T)`: historical observations for
///   each bottom-level series.
/// * `top_series` — Length-T array of total (top-level) historical values.
///
/// # Returns
/// Length-m proportions array summing to ≤ 1 (exactly 1 when no zero-total
/// periods are present).
pub fn average_historical_proportions(
    bottom_series: &Array2<f64>,
    top_series: &Array1<f64>,
) -> Result<Array1<f64>> {
    let m = bottom_series.shape()[0];
    let t_len = bottom_series.shape()[1];

    if top_series.len() != t_len {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: t_len,
            actual: top_series.len(),
        });
    }
    if m == 0 || t_len == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "bottom_series must be non-empty".to_string(),
        ));
    }

    let mut prop_sums = vec![0.0_f64; m];
    let mut valid_count = 0_usize;

    for t in 0..t_len {
        let total = top_series[t];
        if total.abs() < f64::EPSILON {
            continue;
        }
        valid_count += 1;
        for j in 0..m {
            prop_sums[j] += bottom_series[[j, t]] / total;
        }
    }

    if valid_count == 0 {
        return Err(TimeSeriesError::ComputationError(
            "All top-level values are zero; cannot compute proportions".to_string(),
        ));
    }

    let scale = 1.0 / valid_count as f64;
    let proportions: Vec<f64> = prop_sums.iter().map(|&s| s * scale).collect();
    Ok(Array1::from(proportions))
}

/// Compute **proportions of historical averages** from training data.
///
/// `p_j = mean(y_j) / sum_k mean(y_k)`. When the grand sum is zero an error
/// is returned.
///
/// # Arguments
/// * `bottom_series` — Array of shape `(m, T)`.
///
/// # Returns
/// Length-m proportions array summing to 1.
pub fn proportions_of_historical_averages(bottom_series: &Array2<f64>) -> Result<Array1<f64>> {
    let m = bottom_series.shape()[0];
    let t_len = bottom_series.shape()[1];

    if m == 0 || t_len == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "bottom_series must be non-empty".to_string(),
        ));
    }

    let means: Vec<f64> = (0..m)
        .map(|j| bottom_series.row(j).iter().copied().sum::<f64>() / t_len as f64)
        .collect();

    let grand_mean: f64 = means.iter().copied().sum();
    if grand_mean.abs() < f64::EPSILON {
        return Err(TimeSeriesError::ComputationError(
            "Grand mean is zero; cannot compute proportions of historical averages".to_string(),
        ));
    }

    let proportions: Vec<f64> = means.iter().map(|&m| m / grand_mean).collect();
    Ok(Array1::from(proportions))
}

// ─────────────────────────────────────────────────────────────────────────────
// Middle-out forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// Produce coherent forecasts using the **middle-out** approach.
///
/// Middle-out forecasting is a hybrid method:
/// - Forecast the chosen *middle level* directly (externally provided).
/// - Aggregate upward to all ancestor levels using the summing matrix.
/// - Disaggregate downward to all descendant levels using historical
///   proportions stored in `middle_to_bottom_proportions`.
///
/// # Arguments
/// * `middle_level_forecasts` — Array of shape `(k, h)` where k is the number
///   of nodes at the chosen middle level and h is the forecast horizon.
/// * `middle_node_ids` — Length-k vector of node ids corresponding to the rows
///   of `middle_level_forecasts`.
/// * `hierarchy` — The full hierarchy.
/// * `middle_to_bottom_proportions` — Array of shape `(k, m)` where m is the
///   total number of bottom-level nodes. Entry `[i, j]` gives the proportion
///   that bottom node j represents within middle node i. Rows must sum to 1.
/// * `s` — Summing matrix of shape `(n, m)`.
///
/// # Returns
/// An `Array2<f64>` of shape `(n, h)` with coherent forecasts for all nodes.
pub fn middle_out_forecast(
    middle_level_forecasts: &Array2<f64>,
    middle_node_ids: &[usize],
    hierarchy: &Hierarchy,
    middle_to_bottom_proportions: &Array2<f64>,
    s: &Array2<f64>,
) -> Result<Array2<f64>> {
    let k = middle_level_forecasts.shape()[0];
    let h = middle_level_forecasts.shape()[1];
    let n = s.shape()[0];
    let m = s.shape()[1];

    if middle_node_ids.len() != k {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: k,
            actual: middle_node_ids.len(),
        });
    }
    if middle_to_bottom_proportions.shape()[0] != k || middle_to_bottom_proportions.shape()[1] != m
    {
        return Err(TimeSeriesError::InvalidInput(format!(
            "middle_to_bottom_proportions shape must be ({k}, {m}), got ({}, {})",
            middle_to_bottom_proportions.shape()[0],
            middle_to_bottom_proportions.shape()[1]
        )));
    }

    // Validate that the proportions of each middle node sum to 1.
    for i in 0..k {
        let row_sum: f64 = middle_to_bottom_proportions.row(i).iter().copied().sum();
        if (row_sum - 1.0_f64).abs() > 1e-6 {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Proportions for middle node {i} sum to {row_sum:.6}, expected 1"
            )));
        }
    }

    // Step 1: Disaggregate each middle-level forecast to bottom level.
    // bottom_forecasts[j, t] = sum_i  proportions[i,j] * middle_forecasts[i,t]
    let mut bottom_forecasts = Array2::<f64>::zeros((m, h));
    for i in 0..k {
        for j in 0..m {
            let p = middle_to_bottom_proportions[[i, j]];
            if p == 0.0 {
                continue;
            }
            for t in 0..h {
                bottom_forecasts[[j, t]] += p * middle_level_forecasts[[i, t]];
            }
        }
    }

    // Step 2: Aggregate upward using S.
    let all_forecasts = bottom_up_forecast(&bottom_forecasts, s)?;

    // Step 3: Override rows corresponding to middle-level nodes with the
    // directly-provided forecasts (removes small floating-point rounding errors
    // and respects the user-supplied base forecasts exactly).
    let mut result = all_forecasts;
    for (i, &nid) in middle_node_ids.iter().enumerate() {
        if nid >= n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Middle node id {nid} exceeds hierarchy size {n}"
            )));
        }
        for t in 0..h {
            result[[nid, t]] = middle_level_forecasts[[i, t]];
        }
    }

    // Verify that ancestor rows are re-aggregated correctly from children.
    // (BFS from root, overwriting ancestor values with sums of children.)
    let topo_order = hierarchy.topological_order()?;
    for &id in &topo_order {
        let node = hierarchy
            .node(id)
            .ok_or_else(|| TimeSeriesError::InvalidInput(format!("Node id {id} not found")))?;
        if node.is_leaf() {
            continue;
        }
        // Check whether any child is a middle node; if all children are
        // covered, re-sum the parent from children.
        // Only re-sum if at least one child is NOT in middle_node_ids —
        // otherwise the middle-level override is already consistent.
        let children_all_middle = node.children.iter().all(|c| middle_node_ids.contains(c));
        if children_all_middle {
            // Sum children to produce coherent parent.
            for t in 0..h {
                let mut s_val = 0.0_f64;
                for &c in &node.children {
                    s_val += result[[c, t]];
                }
                result[[id, t]] = s_val;
            }
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn build_two_level() -> Hierarchy {
        let mut h = Hierarchy::new();
        let total = h.add_node("Total", None).expect("failed to create total");
        let a = h.add_node("A", Some(total)).expect("failed to create a");
        let b = h.add_node("B", Some(total)).expect("failed to create b");
        h.add_node("A1", Some(a)).expect("unexpected None or Err");
        h.add_node("A2", Some(a)).expect("unexpected None or Err");
        h.add_node("B1", Some(b)).expect("unexpected None or Err");
        h.add_node("B2", Some(b)).expect("unexpected None or Err");
        h
    }

    #[test]
    fn test_hierarchy_structure() {
        let h = build_two_level();
        assert_eq!(h.num_nodes(), 7);
        assert_eq!(h.bottom_level_nodes(), vec![3, 4, 5, 6]);
        assert_eq!(h.root().expect("unexpected None or Err"), 0);
    }

    #[test]
    fn test_summing_matrix_shape() {
        let h = build_two_level();
        let s = summing_matrix(&h).expect("failed to create s");
        assert_eq!(s.shape(), &[7, 4]);
    }

    #[test]
    fn test_summing_matrix_values() {
        let h = build_two_level();
        let s = summing_matrix(&h).expect("failed to create s");
        // Total row (id=0): all four bottom series
        assert_eq!(s.row(0).to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
        // A row (id=1): A1 and A2
        assert_eq!(s.row(1).to_vec(), vec![1.0, 1.0, 0.0, 0.0]);
        // B row (id=2): B1 and B2
        assert_eq!(s.row(2).to_vec(), vec![0.0, 0.0, 1.0, 1.0]);
        // Leaf A1 (id=3): only itself
        assert_eq!(s.row(3).to_vec(), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bottom_up_forecast() {
        let h = build_two_level();
        let s = summing_matrix(&h).expect("failed to create s");
        // Bottom-level forecasts: 4 series, 2 horizons
        let bottom = array![[10.0, 11.0], [20.0, 22.0], [30.0, 33.0], [40.0, 44.0]];
        let all = bottom_up_forecast(&bottom, &s).expect("failed to create all");
        // Total should be 100 / 110
        assert!((all[[0, 0]] - 100.0).abs() < 1e-9);
        assert!((all[[0, 1]] - 110.0).abs() < 1e-9);
        // A = A1 + A2 = 30 / 33
        assert!((all[[1, 0]] - 30.0).abs() < 1e-9);
        // B = B1 + B2 = 70 / 77
        assert!((all[[2, 0]] - 70.0).abs() < 1e-9);
    }

    #[test]
    fn test_top_down_forecast() {
        let props = array![0.25, 0.25, 0.25, 0.25];
        let top = array![100.0, 120.0];
        let result = top_down_forecast(&top, &props).expect("failed to create result");
        assert_eq!(result.shape(), &[4, 2]);
        assert!((result[[0, 0]] - 25.0).abs() < 1e-9);
        assert!((result[[0, 1]] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_top_down_proportions_must_sum_to_one() {
        let bad_props = array![0.3, 0.3, 0.3, 0.3];
        let top = array![100.0];
        assert!(top_down_forecast(&top, &bad_props).is_err());
    }

    #[test]
    fn test_proportions_of_historical_averages() {
        // Two series: series0 always 3x series1
        let bottom = array![[30.0, 30.0, 30.0], [10.0, 10.0, 10.0]];
        let props = proportions_of_historical_averages(&bottom).expect("failed to create props");
        assert!((props[0] - 0.75).abs() < 1e-9);
        assert!((props[1] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_average_historical_proportions() {
        let bottom = array![[30.0, 30.0], [10.0, 10.0]];
        let top = array![40.0, 40.0];
        let props = average_historical_proportions(&bottom, &top).expect("failed to create props");
        assert!((props[0] - 0.75).abs() < 1e-9);
        assert!((props[1] - 0.25).abs() < 1e-9);
    }
}
