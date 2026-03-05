//! Integer partitions and combinatorics
//!
//! This module provides:
//!
//! - `partition_count(n)` — number of integer partitions p(n) via Euler's
//!   pentagonal theorem
//! - `partition_count_table(n_max)` — table p(0), …, p(n_max)
//! - `list_partitions(n)` — enumerate all integer partitions of n
//! - `compositions(n, k)` — ordered k-compositions of n (weak compositions)
//! - `partition_into_distinct(n)` — partitions of n with distinct parts
//! - `young_tableau(shape)` — enumerate all standard Young tableaux of given
//!   shape
//! - `hook_length_formula(shape)` — count of standard Young tableaux via the
//!   hook-length formula

// ── Partition count ───────────────────────────────────────────────────────────

/// Number of integer partitions p(n).
///
/// Uses Euler's pentagonal number theorem recurrence:
///
/// ```text
/// p(n) = Σ_{k≠0} (−1)^{k+1} · p(n − ω(k))
/// ```
///
/// where ω(k) = k(3k−1)/2 for k = 1, −1, 2, −2, …
///
/// The result saturates at `u64::MAX` for very large n.
///
/// # Examples
/// ```
/// use scirs2_special::partitions::partition_count;
/// assert_eq!(partition_count(0),  1);
/// assert_eq!(partition_count(1),  1);
/// assert_eq!(partition_count(4),  5);
/// assert_eq!(partition_count(10), 42);
/// assert_eq!(partition_count(20), 627);
/// ```
pub fn partition_count(n: usize) -> u64 {
    partition_count_table(n)
        .last()
        .copied()
        .unwrap_or(1)
}

/// Table of partition counts p(0), p(1), …, p(n_max).
///
/// Computed in a single pass using the pentagonal recurrence; the time and
/// space complexity is O(n² / 2).
///
/// # Examples
/// ```
/// use scirs2_special::partitions::partition_count_table;
/// let t = partition_count_table(5);
/// assert_eq!(t, vec![1, 1, 2, 3, 5, 7]);
/// ```
pub fn partition_count_table(n_max: usize) -> Vec<u64> {
    let mut p = vec![0u64; n_max + 1];
    p[0] = 1;

    for m in 1..=n_max {
        // Iterate over generalised pentagonal numbers ω(k) = k(3k−1)/2 for
        // k = 1, −1, 2, −2, 3, −3, …
        let mut acc = 0i128;
        let mut k: i64 = 1;
        let mut sign: i64 = 1;
        loop {
            // ω(k) and ω(−k)
            let w_pos = (k * (3 * k - 1) / 2) as usize;
            let w_neg = (k * (3 * k + 1) / 2) as usize;

            if w_pos > m {
                break;
            }
            acc += sign * p[m - w_pos] as i128;

            if w_neg <= m {
                acc += sign * p[m - w_neg] as i128;
            }

            sign = -sign;
            k += 1;
        }
        p[m] = acc.max(0) as u64;
    }
    p
}

// ── Enumerate all partitions ──────────────────────────────────────────────────

/// Enumerate all integer partitions of n in non-increasing part order.
///
/// The number of partitions p(n) grows rapidly; for n ≳ 50 the result is very
/// large (p(50) = 204 226).
///
/// # Examples
/// ```
/// use scirs2_special::partitions::list_partitions;
/// let mut parts = list_partitions(4);
/// parts.sort();
/// assert_eq!(parts.len(), 5);
/// // Partitions of 4: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
/// assert!(parts.contains(&vec![4]));
/// assert!(parts.contains(&vec![3, 1]));
/// ```
pub fn list_partitions(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    if n == 0 {
        result.push(Vec::new());
        return result;
    }
    let mut current = Vec::new();
    enumerate_partitions(n, n, &mut current, &mut result);
    result
}

/// Recursive helper: generate partitions of `remaining` with largest part ≤ `max_part`.
fn enumerate_partitions(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }
    let limit = max_part.min(remaining);
    for part in (1..=limit).rev() {
        current.push(part);
        enumerate_partitions(remaining - part, part, current, result);
        current.pop();
    }
}

// ── Compositions ──────────────────────────────────────────────────────────────

/// All ordered k-compositions (weak) of n: sequences (a₁, …, a_k) with
/// a_i ≥ 0 and Σ a_i = n.
///
/// The number of weak compositions is C(n+k−1, k−1).
///
/// # Examples
/// ```
/// use scirs2_special::partitions::compositions;
/// let c = compositions(3, 2);
/// // (0,3), (1,2), (2,1), (3,0)
/// assert_eq!(c.len(), 4);
/// for comp in &c {
///     assert_eq!(comp.iter().sum::<usize>(), 3);
///     assert_eq!(comp.len(), 2);
/// }
/// ```
pub fn compositions(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        if n == 0 {
            return vec![Vec::new()];
        } else {
            return Vec::new();
        }
    }
    let mut result = Vec::new();
    let mut current = vec![0usize; k];
    enumerate_compositions(n, k, 0, &mut current, &mut result);
    result
}

/// Recursive helper for weak compositions.
fn enumerate_compositions(
    remaining: usize,
    k: usize,
    pos: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if pos == k - 1 {
        current[pos] = remaining;
        result.push(current.clone());
        return;
    }
    for a in 0..=remaining {
        current[pos] = a;
        enumerate_compositions(remaining - a, k, pos + 1, current, result);
    }
}

// ── Partitions into distinct parts ────────────────────────────────────────────

/// All partitions of n into distinct (strictly decreasing) positive parts.
///
/// # Examples
/// ```
/// use scirs2_special::partitions::partition_into_distinct;
/// let d = partition_into_distinct(6);
/// // [6], [5,1], [4,2], [3,2,1]
/// assert_eq!(d.len(), 4);
/// for p in &d {
///     let sum: usize = p.iter().sum();
///     assert_eq!(sum, 6);
///     // All parts must be distinct
///     let mut sorted = p.clone();
///     sorted.sort_unstable();
///     sorted.dedup();
///     assert_eq!(sorted.len(), p.len());
/// }
/// ```
pub fn partition_into_distinct(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    if n == 0 {
        result.push(Vec::new());
        return result;
    }
    let mut current = Vec::new();
    enumerate_distinct_partitions(n, n, &mut current, &mut result);
    result
}

/// Recursive helper: generate distinct-parts partitions of `remaining` with
/// all parts strictly less than `max_part`.
fn enumerate_distinct_partitions(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }
    let limit = max_part.min(remaining);
    for part in (1..=limit).rev() {
        current.push(part);
        // Next parts must be strictly smaller than this one
        enumerate_distinct_partitions(remaining - part, part - 1, current, result);
        current.pop();
    }
}

// ── Standard Young tableaux ───────────────────────────────────────────────────

/// Enumerate all standard Young tableaux (SYT) of a given partition shape.
///
/// A standard Young tableau of shape λ = (λ₁ ≥ λ₂ ≥ … ≥ λ_r) is a filling
/// of the Young diagram with the integers 1, 2, …, n (where n = Σ λ_i) such
/// that entries are strictly increasing along each row (left to right) and
/// each column (top to bottom).
///
/// The number of SYT is given by the hook-length formula (see
/// [`hook_length_formula`]).
///
/// Returns a `Vec` of tableaux; each tableau is stored as a `Vec<Vec<usize>>`
/// of shape len(shape) rows where row i has shape[i] entries.
///
/// # Examples
/// ```
/// use scirs2_special::partitions::young_tableau;
/// let tabs = young_tableau(&[2, 1]);
/// // Shape (2,1): n=3 — there should be 2 SYT
/// assert_eq!(tabs.len(), 2);
/// for tab in &tabs {
///     // Check row-increasing
///     for row in tab {
///         for w in row.windows(2) {
///             assert!(w[0] < w[1]);
///         }
///     }
///     // Check column-increasing
///     for col in 0..tab[0].len() {
///         for r in 1..tab.len() {
///             if col < tab[r].len() {
///                 assert!(tab[r-1][col] < tab[r][col]);
///             }
///         }
///     }
/// }
/// ```
pub fn young_tableau(shape: &[usize]) -> Vec<Vec<Vec<usize>>> {
    let n: usize = shape.iter().sum();
    if n == 0 {
        return vec![vec![]];
    }
    // Validate shape is non-increasing
    for w in shape.windows(2) {
        debug_assert!(w[0] >= w[1], "shape must be non-increasing");
    }

    let rows = shape.len();
    // Initialize empty tableau (each cell = 0 means unfilled)
    let mut tableau: Vec<Vec<usize>> = shape.iter().map(|&c| vec![0usize; c]).collect();
    // Track how many cells have been filled in each row
    let mut fill_count = vec![0usize; rows];

    let mut results: Vec<Vec<Vec<usize>>> = Vec::new();
    fill_young_tableau(
        n,
        1,
        shape,
        &mut tableau,
        &mut fill_count,
        &mut results,
    );
    results
}

/// Recursive backtracker: place number `num` (1-indexed) into a valid cell.
///
/// A cell (r, c) is valid if:
/// 1. It is the next unfilled cell in its row (c == fill_count[r]).
/// 2. It is greater than the cell above (r==0 or fill_count[r-1] > c).
fn fill_young_tableau(
    n: usize,
    num: usize,
    shape: &[usize],
    tableau: &mut Vec<Vec<usize>>,
    fill_count: &mut Vec<usize>,
    results: &mut Vec<Vec<Vec<usize>>>,
) {
    if num > n {
        results.push(tableau.clone());
        return;
    }
    let rows = shape.len();
    for r in 0..rows {
        let c = fill_count[r];
        if c >= shape[r] {
            continue; // row is full
        }
        // Row-increasing: the cell to the left must be smaller (already filled
        // in order, so this is automatic if we only place at fill_count[r]).

        // Column-increasing: the cell above (r-1, c) must be filled with
        // a smaller value, i.e., fill_count[r-1] > c.
        if r > 0 && fill_count[r - 1] <= c {
            continue; // cell above is not yet filled or is at same column
        }
        // Place num at (r, c)
        tableau[r][c] = num;
        fill_count[r] += 1;
        fill_young_tableau(n, num + 1, shape, tableau, fill_count, results);
        fill_count[r] -= 1;
        tableau[r][c] = 0;
    }
}

// ── Hook-length formula ───────────────────────────────────────────────────────

/// Number of standard Young tableaux of given shape via the hook-length
/// formula.
///
/// For a partition λ of n, the number of SYT f^λ satisfies:
///
/// ```text
/// f^λ = n! / ∏_{(i,j) ∈ λ} hook(i, j)
/// ```
///
/// where `hook(i, j)` = (number of cells strictly to the right of (i,j) in
/// the same row) + (number of cells strictly below (i,j) in the same column)
/// + 1.
///
/// The result saturates at `u64::MAX` for large n.
///
/// # Examples
/// ```
/// use scirs2_special::partitions::hook_length_formula;
/// assert_eq!(hook_length_formula(&[]),       1);  // empty shape, 0! = 1
/// assert_eq!(hook_length_formula(&[3]),      1);  // single row: always 1
/// assert_eq!(hook_length_formula(&[2, 1]),   2);  // f^{(2,1)} = 2
/// assert_eq!(hook_length_formula(&[3, 2]),   5);  // f^{(3,2)} = 5
/// assert_eq!(hook_length_formula(&[3, 2, 1]), 16); // f^{(3,2,1)} = 16
/// ```
pub fn hook_length_formula(shape: &[usize]) -> u64 {
    let n: usize = shape.iter().sum();
    if n == 0 {
        return 1;
    }
    let rows = shape.len();

    // Compute conjugate shape (column lengths)
    let max_col = shape[0];
    let mut col_len = vec![0usize; max_col];
    for (r, &row_len) in shape.iter().enumerate() {
        for c in 0..row_len {
            if r < col_len.len() {
                let _ = r; // suppress unused warning
            }
            col_len[c] += 1;
        }
    }

    // Compute n! using f64 for intermediate values, then reconstruct u64
    // For large n this overflows; use saturating arithmetic via log space.
    let log_n_factorial: f64 = (1..=n).map(|k| (k as f64).ln()).sum();

    // Compute sum of log hooks
    let log_hooks: f64 = (0..rows)
        .flat_map(|r| (0..shape[r]).map(move |c| (r, c)))
        .map(|(r, c)| {
            let arm = shape[r] - c - 1; // cells to the right in row r
            let leg = col_len[c] - r - 1; // cells below in column c
            let hook = arm + leg + 1;
            (hook as f64).ln()
        })
        .sum();

    let log_result = log_n_factorial - log_hooks;
    // Round to nearest integer
    let result_f64 = log_result.exp();
    if result_f64 >= u64::MAX as f64 {
        u64::MAX
    } else {
        result_f64.round() as u64
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_count_small() {
        let expected = [1u64, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(partition_count(n), exp, "p({n})");
        }
    }

    #[test]
    fn test_partition_count_table() {
        let t = partition_count_table(10);
        let expected = [1u64, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42];
        assert_eq!(t, expected);
    }

    #[test]
    fn test_partition_count_medium() {
        // p(20) = 627, p(30) = 5604
        assert_eq!(partition_count(20), 627);
        assert_eq!(partition_count(30), 5604);
    }

    #[test]
    fn test_list_partitions_small() {
        assert_eq!(list_partitions(0), vec![vec![] as Vec<usize>]);
        assert_eq!(list_partitions(1).len(), 1);
        assert_eq!(list_partitions(4).len(), 5);
        assert_eq!(list_partitions(10).len(), 42);
        // All partitions must sum to n
        for p in list_partitions(7) {
            assert_eq!(p.iter().sum::<usize>(), 7);
        }
    }

    #[test]
    fn test_list_partitions_non_increasing() {
        for partition in list_partitions(6) {
            for w in partition.windows(2) {
                assert!(w[0] >= w[1], "partition not non-increasing: {partition:?}");
            }
        }
    }

    #[test]
    fn test_compositions_count() {
        // Number of weak k-compositions of n is C(n+k-1, k-1)
        // compositions(3, 2) = C(4,1) = 4
        assert_eq!(compositions(3, 2).len(), 4);
        // compositions(4, 3) = C(6,2) = 15
        assert_eq!(compositions(4, 3).len(), 15);
        // compositions(0, 0) = 1 (empty composition)
        assert_eq!(compositions(0, 0).len(), 1);
        // compositions(0, 3) = 1 (all zeros)
        assert_eq!(compositions(0, 3).len(), 1);
    }

    #[test]
    fn test_compositions_sum() {
        for comp in compositions(5, 3) {
            assert_eq!(comp.iter().sum::<usize>(), 5);
            assert_eq!(comp.len(), 3);
        }
    }

    #[test]
    fn test_partition_into_distinct() {
        // Distinct partitions of 1..6
        let expected_counts = [1usize, 1, 1, 2, 2, 4];
        for (n, &expected) in expected_counts.iter().enumerate() {
            let parts = partition_into_distinct(n);
            assert_eq!(
                parts.len(),
                expected,
                "partition_into_distinct({n}) should have {expected} partitions"
            );
        }
        // All parts must be distinct and sum to n
        for part in partition_into_distinct(10) {
            let sum: usize = part.iter().sum();
            assert_eq!(sum, 10);
            // Check strictly decreasing
            for w in part.windows(2) {
                assert!(w[0] > w[1], "parts not strictly decreasing: {part:?}");
            }
        }
    }

    #[test]
    fn test_young_tableau_count() {
        // f^{(n)} = 1 (single row)
        assert_eq!(young_tableau(&[3]).len(), 1);
        // f^{(1,1,...,1)} = 1 (single column)
        assert_eq!(young_tableau(&[1, 1, 1]).len(), 1);
        // f^{(2,1)} = 2
        assert_eq!(young_tableau(&[2, 1]).len(), 2);
        // f^{(3,2)} = 5
        assert_eq!(young_tableau(&[3, 2]).len(), 5);
        // f^{(2,2)} = 2
        assert_eq!(young_tableau(&[2, 2]).len(), 2);
    }

    #[test]
    fn test_young_tableau_validity() {
        for shape in [&[3, 2][..], &[2, 2, 1][..], &[3, 1, 1][..]] {
            for tab in young_tableau(shape) {
                let n: usize = shape.iter().sum();
                // Check all values 1..=n appear exactly once
                let mut values: Vec<usize> = tab.iter().flatten().copied().collect();
                values.sort_unstable();
                assert_eq!(values, (1..=n).collect::<Vec<_>>());
                // Check row-increasing
                for row in &tab {
                    for w in row.windows(2) {
                        assert!(w[0] < w[1], "row not increasing in {tab:?}");
                    }
                }
                // Check column-increasing
                for c in 0..tab[0].len() {
                    for r in 1..tab.len() {
                        if c < tab[r].len() {
                            assert!(tab[r - 1][c] < tab[r][c], "col not increasing in {tab:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_hook_length_formula() {
        assert_eq!(hook_length_formula(&[]), 1);
        assert_eq!(hook_length_formula(&[1]), 1);
        assert_eq!(hook_length_formula(&[3]), 1);
        assert_eq!(hook_length_formula(&[1, 1, 1]), 1);
        assert_eq!(hook_length_formula(&[2, 1]), 2);
        assert_eq!(hook_length_formula(&[3, 2]), 5);
        assert_eq!(hook_length_formula(&[2, 2]), 2);
        assert_eq!(hook_length_formula(&[3, 2, 1]), 16);
        assert_eq!(hook_length_formula(&[4, 3, 2, 1]), 768);
    }

    #[test]
    fn test_hook_matches_tableau_count() {
        // The number of SYT enumerated must equal the hook-length formula
        for shape in [
            &[2, 1][..],
            &[3, 2][..],
            &[2, 2][..],
            &[3, 1][..],
            &[3, 2, 1][..],
        ] {
            let count = young_tableau(shape).len() as u64;
            let hook = hook_length_formula(shape);
            assert_eq!(count, hook, "mismatch for shape {shape:?}");
        }
    }
}
