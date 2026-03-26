//! Knapsack Problem Solvers
//!
//! This module provides exact and approximate algorithms for several variants
//! of the classical knapsack problem.
//!
//! # Variants
//!
//! - **0-1 Knapsack** ([`knapsack_dp`]): Each item can be selected at most once.
//!   Solved exactly via dynamic programming in O(n * capacity) pseudo-polynomial time.
//!
//! - **Fractional Knapsack** ([`fractional_knapsack`]): Items can be taken as fractions.
//!   Solved greedily in O(n log n) by sorting on value/weight ratio.
//!
//! - **Bounded Knapsack** ([`bounded_knapsack`]): Each item has a limited supply count.
//!   Reduced to 0-1 knapsack via binary splitting.
//!
//! - **Multi-dimensional Knapsack** ([`multi_dimensional_knapsack`]): Multiple capacity
//!   constraints. Solved via branch-and-bound with LP relaxation.
//!
//! # References
//! - Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer
//!   Implementations*. Wiley.
//! - Pisinger, D. (1995). "Algorithms for Knapsack Problems." PhD thesis, DIKU.

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// 0-1 Knapsack (DP)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the 0-1 knapsack problem with dynamic programming.
///
/// Maximizes `sum(values[i] * x[i])` subject to `sum(weights[i] * x[i]) <= capacity`
/// and `x[i] in {0, 1}`.
///
/// Weights are converted to integer indices by multiplying by a scaling factor
/// so that the DP table has size `n * (scaled_capacity + 1)`.
///
/// # Arguments
/// * `values`   – item values (must be non-negative)
/// * `weights`  – item weights (must be non-negative)
/// * `capacity` – knapsack capacity (must be non-negative)
///
/// # Returns
/// `(optimal_value, selection)` where `selection[i]` is `true` iff item `i` is included.
///
/// # Errors
/// Returns [`OptimizeError::InvalidInput`] if the inputs have different lengths
/// or if any value/weight is negative.
///
/// # Example
/// ```
/// use scirs2_optimize::integer::knapsack::knapsack_dp;
///
/// let values  = vec![4.0, 3.0, 5.0, 2.0, 6.0];
/// let weights = vec![2.0, 3.0, 4.0, 1.0, 5.0];
/// let (val, sel) = knapsack_dp(&values, &weights, 8.0).expect("valid input");
/// assert!((val - 12.0).abs() < 1e-6);
/// assert_eq!(sel, vec![true, false, false, true, true]);
/// ```
pub fn knapsack_dp(
    values: &[f64],
    weights: &[f64],
    capacity: f64,
) -> OptimizeResult<(f64, Vec<bool>)> {
    let n = values.len();
    if weights.len() != n {
        return Err(OptimizeError::InvalidInput(
            "values and weights must have the same length".to_string(),
        ));
    }
    for &v in values {
        if v < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all values must be non-negative".to_string(),
            ));
        }
    }
    for &w in weights {
        if w < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all weights must be non-negative".to_string(),
            ));
        }
    }
    if capacity < 0.0 {
        return Err(OptimizeError::InvalidInput(
            "capacity must be non-negative".to_string(),
        ));
    }
    if n == 0 {
        return Ok((0.0, Vec::new()));
    }

    // Scale weights to integers.  We need a scale factor such that
    // (weight * scale).round() preserves the relative ordering and feasibility.
    //
    // Strategy: find the smallest fractional granularity needed.  If all
    // weights are (near-)integers, scale = 1.  Otherwise we pick a scale
    // that maps the smallest fractional part to at least 1 grid unit, capped
    // to avoid memory blowup.
    let scale = {
        // Collect the fractional decimal digits needed
        let mut best_scale = 1.0_f64;
        for &w in weights.iter().chain(std::iter::once(&capacity)) {
            if w <= 0.0 {
                continue;
            }
            // Find the number of significant decimal places (up to 6)
            let mut s = 1.0_f64;
            for _ in 0..6 {
                if ((w * s).round() - w * s).abs() < 1e-9 {
                    break;
                }
                s *= 10.0;
            }
            if s > best_scale {
                best_scale = s;
            }
        }
        // Cap so that scaled_cap doesn't exceed a reasonable size
        let max_scale = if capacity > 0.0 {
            (20_000_000.0 / capacity).max(1.0)
        } else {
            1e6
        };
        best_scale.min(max_scale)
    };

    let scaled_cap = (capacity * scale).round() as usize;
    let scaled_w: Vec<usize> = weights
        .iter()
        .map(|&w| (w * scale).round() as usize)
        .collect();

    // dp[i][w] = max value using items 0..i with capacity w
    // Space-optimised: use two rows
    let cols = scaled_cap + 1;
    let mut prev = vec![0.0_f64; cols];
    let mut curr = vec![0.0_f64; cols];

    // We also need a back-tracking table: keep[i][w] = true if item i was chosen
    // When n and cols are large, this can be memory-heavy; we cap at a reasonable size.
    let use_backtrack = (n as u64) * (cols as u64) <= 40_000_000;

    let mut keep: Vec<Vec<bool>> = if use_backtrack {
        vec![vec![false; cols]; n]
    } else {
        Vec::new()
    };

    for i in 0..n {
        for w in 0..cols {
            let sw = scaled_w[i];
            if sw > w {
                curr[w] = prev[w];
                if use_backtrack {
                    keep[i][w] = false;
                }
            } else {
                let take = prev[w - sw] + values[i];
                if take > prev[w] {
                    curr[w] = take;
                    if use_backtrack {
                        keep[i][w] = true;
                    }
                } else {
                    curr[w] = prev[w];
                    if use_backtrack {
                        keep[i][w] = false;
                    }
                }
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let opt_val = prev[scaled_cap];

    // Back-track to find selection
    let selection = if use_backtrack {
        let mut sel = vec![false; n];
        let mut w = scaled_cap;
        for i in (0..n).rev() {
            if keep[i][w] {
                sel[i] = true;
                w -= scaled_w[i];
            }
        }
        sel
    } else {
        // If back-tracking is disabled, return a greedy approximation for the selection
        greedy_selection_fallback(values, weights, capacity)
    };

    Ok((opt_val, selection))
}

/// Greedy selection fallback (used when DP backtrack table is too large).
fn greedy_selection_fallback(values: &[f64], weights: &[f64], capacity: f64) -> Vec<bool> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ra = if weights[a] > 0.0 {
            values[a] / weights[a]
        } else {
            f64::INFINITY
        };
        let rb = if weights[b] > 0.0 {
            values[b] / weights[b]
        } else {
            f64::INFINITY
        };
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sel = vec![false; n];
    let mut rem = capacity;
    for i in order {
        if weights[i] <= rem {
            sel[i] = true;
            rem -= weights[i];
        }
    }
    sel
}

// ─────────────────────────────────────────────────────────────────────────────
// Fractional Knapsack (greedy)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the fractional knapsack problem with a greedy algorithm.
///
/// Items can be taken as fractions; the optimal solution is obtained by
/// sorting items by value-to-weight ratio and filling greedily.
///
/// Returns the maximum total value achievable.
///
/// # Arguments
/// * `values`   – item values (non-negative)
/// * `weights`  – item weights (non-negative, items with weight 0 are taken first)
/// * `capacity` – total capacity
///
/// # Errors
/// Returns [`OptimizeError::InvalidInput`] on length mismatch or negative inputs.
///
/// # Example
/// ```
/// use scirs2_optimize::integer::knapsack::fractional_knapsack;
///
/// let values  = vec![60.0, 100.0, 120.0];
/// let weights = vec![10.0,  20.0,  30.0];
/// let val = fractional_knapsack(&values, &weights, 50.0).expect("valid input");
/// assert!((val - 240.0).abs() < 1e-6);
/// ```
pub fn fractional_knapsack(values: &[f64], weights: &[f64], capacity: f64) -> OptimizeResult<f64> {
    let n = values.len();
    if weights.len() != n {
        return Err(OptimizeError::InvalidInput(
            "values and weights must have the same length".to_string(),
        ));
    }
    for &v in values {
        if v < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all values must be non-negative".to_string(),
            ));
        }
    }
    for &w in weights {
        if w < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all weights must be non-negative".to_string(),
            ));
        }
    }
    if capacity < 0.0 {
        return Err(OptimizeError::InvalidInput(
            "capacity must be non-negative".to_string(),
        ));
    }
    if n == 0 {
        return Ok(0.0);
    }

    // Sort by value/weight descending; weight-zero items have infinite ratio
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ra = if weights[a] > 0.0 {
            values[a] / weights[a]
        } else {
            f64::INFINITY
        };
        let rb = if weights[b] > 0.0 {
            values[b] / weights[b]
        } else {
            f64::INFINITY
        };
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut total_value = 0.0;
    let mut remaining = capacity;

    for i in order {
        if remaining <= 0.0 {
            break;
        }
        let w = weights[i];
        if w <= 0.0 {
            // Zero-weight items: take fully
            total_value += values[i];
        } else if w <= remaining {
            total_value += values[i];
            remaining -= w;
        } else {
            // Take fraction
            total_value += values[i] * (remaining / w);
            remaining = 0.0;
        }
    }

    Ok(total_value)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounded Knapsack (binary splitting → 0-1 DP)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the bounded knapsack problem.
///
/// Maximizes `sum(values[i] * x[i])` subject to:
/// - `sum(weights[i] * x[i]) <= capacity`
/// - `0 <= x[i] <= counts[i]`  (integer)
///
/// Uses *binary splitting*: each item `i` with bound `b_i` is split into
/// O(log b_i) virtual 0-1 items representing multiples 1, 2, 4, …, remainder.
/// The resulting 0-1 DP finds the exact optimum.
///
/// # Arguments
/// * `values`   – per-unit values
/// * `weights`  – per-unit weights
/// * `counts`   – maximum number of units of each item
/// * `capacity` – knapsack capacity
///
/// # Returns
/// `(optimal_value, selection)` where `selection[i]` is the number of
/// units of item `i` in the solution.
///
/// # Errors
/// Returns [`OptimizeError::InvalidInput`] on dimension mismatch or negative inputs.
///
/// # Example
/// ```
/// use scirs2_optimize::integer::knapsack::bounded_knapsack;
///
/// let values  = vec![3.0, 4.0, 5.0];
/// let weights = vec![1.0, 2.0, 3.0];
/// let counts  = vec![4usize, 3, 2];
/// let (val, sel) = bounded_knapsack(&values, &weights, &counts, 7.0).expect("valid input");
/// // Optimal is 17: e.g. 4×item0 (v=12,w=4) + 1×item2 (v=5,w=3)
/// let total_w: f64 = weights.iter().zip(sel.iter()).map(|(&w, &c)| w * c as f64).sum();
/// assert!(total_w <= 7.0 + 1e-9);
/// assert!(val >= 17.0 - 1e-6);
/// ```
pub fn bounded_knapsack(
    values: &[f64],
    weights: &[f64],
    counts: &[usize],
    capacity: f64,
) -> OptimizeResult<(f64, Vec<usize>)> {
    let n = values.len();
    if weights.len() != n || counts.len() != n {
        return Err(OptimizeError::InvalidInput(
            "values, weights, and counts must have the same length".to_string(),
        ));
    }
    for &v in values {
        if v < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all values must be non-negative".to_string(),
            ));
        }
    }
    for &w in weights {
        if w < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all weights must be non-negative".to_string(),
            ));
        }
    }
    if capacity < 0.0 {
        return Err(OptimizeError::InvalidInput(
            "capacity must be non-negative".to_string(),
        ));
    }
    if n == 0 {
        return Ok((0.0, Vec::new()));
    }

    // Build virtual 0-1 items via binary splitting
    // Each virtual item carries (value, weight, original_item_index, multiplier)
    let mut virtual_values: Vec<f64> = Vec::new();
    let mut virtual_weights: Vec<f64> = Vec::new();
    let mut virtual_orig: Vec<usize> = Vec::new();
    let mut virtual_mult: Vec<usize> = Vec::new();

    for i in 0..n {
        let mut remaining = counts[i];
        let mut k = 1usize;
        while remaining > 0 {
            let take = k.min(remaining);
            virtual_values.push(values[i] * take as f64);
            virtual_weights.push(weights[i] * take as f64);
            virtual_orig.push(i);
            virtual_mult.push(take);
            remaining -= take;
            k *= 2;
        }
    }

    // Solve 0-1 knapsack on virtual items
    let (opt_val, sel01) = knapsack_dp(&virtual_values, &virtual_weights, capacity)?;

    // Map back: count how many of each original item was selected
    let mut selection = vec![0usize; n];
    for (k, selected) in sel01.iter().enumerate() {
        if *selected {
            selection[virtual_orig[k]] += virtual_mult[k];
        }
    }

    // Clamp to counts just in case of floating-point rounding
    for i in 0..n {
        if selection[i] > counts[i] {
            selection[i] = counts[i];
        }
    }

    Ok((opt_val, selection))
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-dimensional Knapsack (branch-and-bound with LP relaxation)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the multi-dimensional 0-1 knapsack problem.
///
/// Maximizes `sum(values[i] * x[i])` subject to:
/// - For each dimension `d`: `sum(weights[d][i] * x[i]) <= capacities[d]`
/// - `x[i] in {0, 1}`
///
/// Uses a branch-and-bound algorithm with a greedy LP upper bound.
///
/// # Arguments
/// * `values`      – item values (non-negative), length n
/// * `weights`     – `d × n` weight matrix; `weights[d][i]` is the weight of
///                   item `i` in dimension `d`
/// * `capacities`  – length-d capacity vector
///
/// # Returns
/// The maximum total value achievable.
///
/// # Errors
/// Returns [`OptimizeError::InvalidInput`] on invalid dimensions or negative inputs.
///
/// # Example
/// ```
/// use scirs2_optimize::integer::knapsack::multi_dimensional_knapsack;
///
/// // 3 items, 2 dimensions
/// let values  = vec![10.0, 6.0, 5.0];
/// let weights = vec![
///     vec![2.0, 3.0, 1.0],  // dimension 0 weights
///     vec![4.0, 1.0, 2.0],  // dimension 1 weights
/// ];
/// let capacities = vec![5.0, 6.0];
/// let val = multi_dimensional_knapsack(&values, &weights, &capacities).expect("valid input");
/// assert!(val >= 15.0 - 1e-6);
/// ```
pub fn multi_dimensional_knapsack(
    values: &[f64],
    weights: &[Vec<f64>],
    capacities: &[f64],
) -> OptimizeResult<f64> {
    let n = values.len();
    let d = weights.len();

    if d == 0 {
        // No constraints – take everything
        return Ok(values.iter().sum());
    }
    for dim in 0..d {
        if weights[dim].len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "weights[{}] has length {}, expected {}",
                dim,
                weights[dim].len(),
                n
            )));
        }
    }
    if capacities.len() != d {
        return Err(OptimizeError::InvalidInput(format!(
            "capacities length {} != number of weight dimensions {}",
            capacities.len(),
            d
        )));
    }
    for &v in values {
        if v < 0.0 {
            return Err(OptimizeError::InvalidInput(
                "all values must be non-negative".to_string(),
            ));
        }
    }
    for (dim, cap) in capacities.iter().enumerate() {
        if *cap < 0.0 {
            return Err(OptimizeError::InvalidInput(format!(
                "capacity[{}] must be non-negative",
                dim
            )));
        }
    }
    if n == 0 {
        return Ok(0.0);
    }

    // Solve via branch-and-bound with LP relaxation upper bound
    let best = Mkp::new(n, d, values, weights, capacities).branch_and_bound();
    Ok(best)
}

/// Internal helper for multi-dimensional knapsack B&B.
struct Mkp<'a> {
    n: usize,
    d: usize,
    values: &'a [f64],
    weights: &'a [Vec<f64>],
    capacities: &'a [f64],
    /// Item ordering by decreasing value/first-dim-weight ratio
    order: Vec<usize>,
}

impl<'a> Mkp<'a> {
    fn new(
        n: usize,
        d: usize,
        values: &'a [f64],
        weights: &'a [Vec<f64>],
        capacities: &'a [f64],
    ) -> Self {
        let mut order: Vec<usize> = (0..n).collect();
        // Sort by composite ratio: value / sum_of_weights
        order.sort_by(|&a, &b| {
            let sum_a: f64 = (0..d).map(|k| weights[k][a]).sum::<f64>();
            let sum_b: f64 = (0..d).map(|k| weights[k][b]).sum::<f64>();
            let ra = if sum_a > 0.0 {
                values[a] / sum_a
            } else {
                f64::INFINITY
            };
            let rb = if sum_b > 0.0 {
                values[b] / sum_b
            } else {
                f64::INFINITY
            };
            rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
        });
        Mkp {
            n,
            d,
            values,
            weights,
            capacities,
            order,
        }
    }

    /// LP upper bound by relaxing integrality (fractional knapsack per first dim).
    fn lp_upper_bound(&self, sel: &[Option<bool>], rem_cap: &[f64]) -> f64 {
        // Greedy fractional relaxation respecting the remaining capacities
        let mut curr_val: f64 = sel
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|take| if take { self.values[i] } else { 0.0 }))
            .sum();

        let mut remaining: Vec<f64> = rem_cap.to_vec();

        // Items not yet assigned, in order
        for &item in &self.order {
            if sel[item].is_some() {
                continue;
            }
            // Check feasibility: what fraction can we take?
            let mut max_frac = 1.0_f64;
            for dim in 0..self.d {
                let w = self.weights[dim][item];
                if w > 0.0 {
                    let frac = remaining[dim] / w;
                    if frac < max_frac {
                        max_frac = frac;
                    }
                }
            }
            if max_frac <= 0.0 {
                continue;
            }
            let frac = max_frac.min(1.0);
            curr_val += self.values[item] * frac;
            if frac >= 1.0 {
                for dim in 0..self.d {
                    remaining[dim] -= self.weights[dim][item];
                }
            }
        }
        curr_val
    }

    /// Check if adding item i is feasible given remaining capacities.
    fn is_feasible_add(&self, item: usize, rem_cap: &[f64]) -> bool {
        for dim in 0..self.d {
            if self.weights[dim][item] > rem_cap[dim] + 1e-10 {
                return false;
            }
        }
        true
    }

    fn branch_and_bound(&self) -> f64 {
        // Stack-based DFS B&B
        // State: (sel, rem_cap, current_value, next_item_in_order)
        struct State {
            sel: Vec<Option<bool>>,
            rem_cap: Vec<f64>,
            curr_val: f64,
            // index into self.order
            next_idx: usize,
        }

        let init_sel = vec![None; self.n];
        let init_rem = self.capacities.to_vec();
        let init_ub = self.lp_upper_bound(&init_sel, &init_rem);

        let mut best = 0.0_f64;
        let mut stack: Vec<State> = Vec::new();
        stack.push(State {
            sel: init_sel,
            rem_cap: init_rem,
            curr_val: 0.0,
            next_idx: 0,
        });

        // Limit iterations to avoid exponential blow-up on large instances
        let max_nodes = 200_000usize;
        let mut nodes = 0usize;

        while let Some(state) = stack.pop() {
            nodes += 1;
            if nodes > max_nodes {
                break;
            }

            // Compute UB
            let ub = self.lp_upper_bound(&state.sel, &state.rem_cap);
            if ub <= best + 1e-9 {
                continue; // prune
            }

            if state.next_idx >= self.order.len() {
                // Leaf node
                if state.curr_val > best {
                    best = state.curr_val;
                }
                continue;
            }

            let item = self.order[state.next_idx];

            // Branch: don't take item
            {
                let mut new_sel = state.sel.clone();
                new_sel[item] = Some(false);
                stack.push(State {
                    sel: new_sel,
                    rem_cap: state.rem_cap.clone(),
                    curr_val: state.curr_val,
                    next_idx: state.next_idx + 1,
                });
            }

            // Branch: take item (if feasible)
            if self.is_feasible_add(item, &state.rem_cap) {
                let mut new_sel = state.sel.clone();
                new_sel[item] = Some(true);
                let mut new_rem = state.rem_cap.clone();
                for dim in 0..self.d {
                    new_rem[dim] -= self.weights[dim][item];
                }
                let new_val = state.curr_val + self.values[item];
                if new_val > best {
                    best = new_val;
                }
                stack.push(State {
                    sel: new_sel,
                    rem_cap: new_rem,
                    curr_val: new_val,
                    next_idx: state.next_idx + 1,
                });
            }
        }

        // If B&B was cut short, run greedy for a lower bound comparison
        let greedy = self.greedy_solution();
        best.max(greedy)
    }

    /// Greedy lower bound: take items greedily in sorted order
    fn greedy_solution(&self) -> f64 {
        let mut rem: Vec<f64> = self.capacities.to_vec();
        let mut val = 0.0_f64;
        for &item in &self.order {
            if self.is_feasible_add(item, &rem) {
                val += self.values[item];
                for dim in 0..self.d {
                    rem[dim] -= self.weights[dim][item];
                }
            }
        }
        val
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── 0-1 Knapsack ───────────────────────────────────────────────────────

    #[test]
    fn test_knapsack_dp_basic() {
        let values = vec![4.0, 3.0, 5.0, 2.0, 6.0];
        let weights = vec![2.0, 3.0, 4.0, 1.0, 5.0];
        let (val, sel) = knapsack_dp(&values, &weights, 8.0).expect("unexpected None or Err");
        assert_abs_diff_eq!(val, 12.0, epsilon = 1e-6);
        // Verify selection is feasible and achieves the reported value
        let total_w: f64 = weights
            .iter()
            .zip(sel.iter())
            .map(|(&w, &s)| if s { w } else { 0.0 })
            .sum();
        let total_v: f64 = values
            .iter()
            .zip(sel.iter())
            .map(|(&v, &s)| if s { v } else { 0.0 })
            .sum();
        assert!(total_w <= 8.0 + 1e-9, "weight {} > capacity 8", total_w);
        assert_abs_diff_eq!(total_v, val, epsilon = 1e-6);
    }

    #[test]
    fn test_knapsack_dp_empty() {
        let (val, sel) = knapsack_dp(&[], &[], 10.0).expect("unexpected None or Err");
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
        assert!(sel.is_empty());
    }

    #[test]
    fn test_knapsack_dp_none_fit() {
        // All items too heavy
        let values = vec![10.0, 20.0];
        let weights = vec![5.0, 8.0];
        let (val, _sel) = knapsack_dp(&values, &weights, 3.0).expect("unexpected None or Err");
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_knapsack_dp_all_fit() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0, 1.0];
        let (val, sel) = knapsack_dp(&values, &weights, 10.0).expect("unexpected None or Err");
        assert_abs_diff_eq!(val, 6.0, epsilon = 1e-6);
        assert_eq!(sel, vec![true, true, true]);
    }

    #[test]
    fn test_knapsack_dp_error_negative_value() {
        let result = knapsack_dp(&[-1.0, 2.0], &[1.0, 1.0], 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_knapsack_dp_error_length_mismatch() {
        let result = knapsack_dp(&[1.0, 2.0, 3.0], &[1.0, 2.0], 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_knapsack_dp_integer_weights() {
        // Classic textbook example: n=4, cap=5
        // values: 10,40,30,50  weights: 5,4,6,3
        let values = vec![10.0, 40.0, 30.0, 50.0];
        let weights = vec![5.0, 4.0, 6.0, 3.0];
        let (val, sel) = knapsack_dp(&values, &weights, 5.0).expect("unexpected None or Err");
        // Best: item 1 (val=40, w=4) + no second item that fits (remaining=1)
        //       or item 3 (val=50, w=3) + no fit for rest
        // Actually: item 3 (w=3, v=50) leaves 2 cap; no item fits in 2 -> val=50
        // But: item 1 (w=4, v=40) leaves 1 cap -> val=40
        // So optimal = 50
        assert_abs_diff_eq!(val, 50.0, epsilon = 1e-6);
        let total_w: f64 = weights
            .iter()
            .zip(sel.iter())
            .map(|(&w, &s)| if s { w } else { 0.0 })
            .sum();
        assert!(total_w <= 5.0 + 1e-9);
    }

    // ── Fractional Knapsack ────────────────────────────────────────────────

    #[test]
    fn test_fractional_knapsack_basic() {
        // Classic: items with (value, weight) = (60,10),(100,20),(120,30), cap=50
        // Optimal: all of item 0 & 1, then 2/3 of item 2
        // = 60 + 100 + 80 = 240
        let values = vec![60.0, 100.0, 120.0];
        let weights = vec![10.0, 20.0, 30.0];
        let val = fractional_knapsack(&values, &weights, 50.0).expect("failed to create val");
        assert_abs_diff_eq!(val, 240.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fractional_knapsack_empty() {
        let val = fractional_knapsack(&[], &[], 100.0).expect("failed to create val");
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fractional_knapsack_exact_fit() {
        let values = vec![10.0, 20.0];
        let weights = vec![5.0, 10.0];
        let val = fractional_knapsack(&values, &weights, 15.0).expect("failed to create val");
        assert_abs_diff_eq!(val, 30.0, epsilon = 1e-9);
    }

    #[test]
    fn test_fractional_knapsack_zero_weight_item() {
        // Zero-weight item should always be taken fully
        let values = vec![100.0, 5.0];
        let weights = vec![0.0, 1.0];
        let val = fractional_knapsack(&values, &weights, 1.0).expect("failed to create val");
        assert_abs_diff_eq!(val, 105.0, epsilon = 1e-9);
    }

    #[test]
    fn test_fractional_knapsack_error_negative_capacity() {
        let result = fractional_knapsack(&[1.0], &[1.0], -1.0);
        assert!(result.is_err());
    }

    // ── Bounded Knapsack ───────────────────────────────────────────────────

    #[test]
    fn test_bounded_knapsack_basic() {
        let values = vec![3.0, 4.0, 5.0];
        let weights = vec![1.0, 2.0, 3.0];
        let counts = vec![4usize, 3, 2];
        let (val, sel) =
            bounded_knapsack(&values, &weights, &counts, 7.0).expect("unexpected None or Err");
        // Verify feasibility
        let total_w: f64 = weights
            .iter()
            .zip(sel.iter())
            .map(|(&w, &c)| w * c as f64)
            .sum();
        assert!(
            total_w <= 7.0 + 1e-9,
            "weight {} exceeds capacity 7",
            total_w
        );
        // Verify counts respected
        for i in 0..3 {
            assert!(
                sel[i] <= counts[i],
                "sel[{}]={} > counts[{}]={}",
                i,
                sel[i],
                i,
                counts[i]
            );
        }
        // Optimal is 17: e.g. 3×item0 (v=9,w=3) + 2×item1 (v=8,w=4) = 17, w=7
        // or 4×item0 (v=12,w=4) + 1×item2 (v=5,w=3) = 17, w=7
        assert!(val >= 17.0 - 1e-6, "val={} should be >= 17", val);
    }

    #[test]
    fn test_bounded_knapsack_unit_counts() {
        // bounded with counts=1 should match 0-1 knapsack
        let values = vec![4.0, 3.0, 5.0, 2.0];
        let weights = vec![2.0, 3.0, 4.0, 1.0];
        let counts = vec![1usize; 4];
        let (val_b, _) =
            bounded_knapsack(&values, &weights, &counts, 6.0).expect("unexpected None or Err");
        let (val_dp, _) = knapsack_dp(&values, &weights, 6.0).expect("unexpected None or Err");
        assert_abs_diff_eq!(val_b, val_dp, epsilon = 1e-6);
    }

    #[test]
    fn test_bounded_knapsack_error_mismatch() {
        let result = bounded_knapsack(&[1.0, 2.0], &[1.0], &[1, 1], 5.0);
        assert!(result.is_err());
    }

    // ── Multi-dimensional Knapsack ─────────────────────────────────────────

    #[test]
    fn test_multi_dimensional_knapsack_1d() {
        // 1-D should equal the 0-1 DP result
        let values = vec![4.0, 3.0, 5.0, 2.0, 6.0];
        let weights = vec![vec![2.0, 3.0, 4.0, 1.0, 5.0]];
        let caps = vec![8.0];
        let val_md =
            multi_dimensional_knapsack(&values, &weights, &caps).expect("failed to create val_md");
        let (val_dp, _) = knapsack_dp(&values, &weights[0], 8.0).expect("unexpected None or Err");
        assert_abs_diff_eq!(val_md, val_dp, epsilon = 1e-6);
    }

    #[test]
    fn test_multi_dimensional_knapsack_2d() {
        let values = vec![10.0, 6.0, 5.0];
        let weights = vec![vec![2.0, 3.0, 1.0], vec![4.0, 1.0, 2.0]];
        let caps = vec![5.0, 6.0];
        let val =
            multi_dimensional_knapsack(&values, &weights, &caps).expect("failed to create val");
        assert!(val >= 15.0 - 1e-6, "val={} should be >= 15", val);
    }

    #[test]
    fn test_multi_dimensional_knapsack_no_dims() {
        // No constraints → take everything
        let values = vec![1.0, 2.0, 3.0];
        let weights: Vec<Vec<f64>> = Vec::new();
        let caps: Vec<f64> = Vec::new();
        let val =
            multi_dimensional_knapsack(&values, &weights, &caps).expect("failed to create val");
        assert_abs_diff_eq!(val, 6.0, epsilon = 1e-9);
    }

    #[test]
    fn test_multi_dimensional_knapsack_empty_items() {
        let weights: Vec<Vec<f64>> = vec![vec![]];
        let caps = vec![5.0];
        let val = multi_dimensional_knapsack(&[], &weights, &caps).expect("failed to create val");
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_multi_dimensional_knapsack_error_dim_mismatch() {
        let values = vec![1.0, 2.0];
        let weights = vec![vec![1.0]]; // only 1 item weight, but 2 items
        let caps = vec![5.0];
        let result = multi_dimensional_knapsack(&values, &weights, &caps);
        assert!(result.is_err());
    }
}
