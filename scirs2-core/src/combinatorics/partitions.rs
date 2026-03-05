//! Integer partition utilities.
//!
//! An *integer partition* of a positive integer n is a way of writing n as a
//! sum of positive integers where the order of addends does not matter.
//!
//! This module provides counting, enumeration, restriction, conjugation, and
//! visualisation of integer partitions.

use std::fmt;

// ---------------------------------------------------------------------------
// partition_count
// ---------------------------------------------------------------------------

/// Compute the number of integer partitions p(n) using Euler's recurrence.
///
/// Euler's recurrence uses pentagonal numbers:
///   p(n) = sum_{k≠0} (-1)^{k+1} * p(n - g_k)
/// where g_k = k*(3k-1)/2 for k = 1, -1, 2, -2, … and p(0) = 1.
///
/// Returns `None` on overflow (`u64` is sufficient up to n ≈ 600).
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::partitions::partition_count;
/// assert_eq!(partition_count(0), Some(1));
/// assert_eq!(partition_count(5), Some(7));
/// assert_eq!(partition_count(10), Some(42));
/// ```
pub fn partition_count(n: usize) -> Option<u64> {
    let mut p = vec![0u64; n + 1];
    p[0] = 1;
    for i in 1..=n {
        let mut sum: i128 = 0;
        // Iterate over generalised pentagonal numbers g(k) = k*(3k-1)/2
        // for k = 1, -1, 2, -2, 3, -3, …
        let mut k: i64 = 1;
        loop {
            // Positive k.
            let g_pos = (k * (3 * k - 1) / 2) as usize;
            if g_pos > i {
                break;
            }
            // Sign is (-1)^{k+1}; for k=1 it is +1.
            let sign: i128 = if k % 2 == 0 { -1 } else { 1 };
            sum += sign * (p[i - g_pos] as i128);

            // Negative k (−k gives pentagonal number k*(3k+1)/2).
            let g_neg = (k * (3 * k + 1) / 2) as usize;
            if g_neg <= i {
                let sign_neg: i128 = if k % 2 == 0 { -1 } else { 1 };
                sum += sign_neg * (p[i - g_neg] as i128);
            }
            k += 1;
        }
        p[i] = u64::try_from(sum).ok()?;
    }
    Some(p[n])
}

// ---------------------------------------------------------------------------
// enumerate_partitions
// ---------------------------------------------------------------------------

/// Enumerate all integer partitions of `n` in lexicographically descending
/// order (largest-part-first ordering).
///
/// Each partition is represented as a `Vec<usize>` of parts in non-increasing
/// order.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::partitions::enumerate_partitions;
/// let parts = enumerate_partitions(4);
/// // [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
/// assert_eq!(parts.len(), 5);
/// assert_eq!(parts[0], vec![4]);
/// assert_eq!(parts[4], vec![1, 1, 1, 1]);
/// ```
pub fn enumerate_partitions(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    if n == 0 {
        result.push(vec![]);
        return result;
    }
    // Iterative descent algorithm.
    // We maintain a "current" partition as a stack of parts.
    let mut parts: Vec<usize> = Vec::with_capacity(n);
    let mut remaining = n;
    // max_part tracks the maximum allowed next part (so parts are non-increasing).
    parts.push(n);
    remaining = 0;
    // We use a work stack instead of recursion to avoid overflow on large n.
    // State: (remaining, max_part, parts_prefix).
    // Use a simpler iterative approach with explicit backtracking.
    struct Frame {
        remaining: usize,
        max_part: usize,
    }
    let mut stack: Vec<Frame> = vec![Frame {
        remaining: n,
        max_part: n,
    }];
    let _ = parts.pop();
    let mut prefix: Vec<usize> = Vec::new();

    // Actually use a cleaner recursive-style with an explicit stack.
    // Each frame: (remaining, max_part, prefix_len)
    // When a frame is popped, the prefix is trimmed to prefix_len.
    #[derive(Debug)]
    struct State {
        remaining: usize,
        max_part: usize,
    }

    let _ = stack.pop();
    let mut state_stack: Vec<State> = vec![State {
        remaining: n,
        max_part: n,
    }];
    // Store, for each state on the stack, how long the prefix was when pushed.
    let mut prefix_lens: Vec<usize> = vec![0];

    while let Some(state) = state_stack.pop() {
        let plen = prefix_lens
            .pop()
            .expect("prefix_lens tracks state_stack");
        // Restore prefix.
        prefix.truncate(plen);

        if state.remaining == 0 {
            result.push(prefix.clone());
            continue;
        }
        // Push children in reverse order so the largest part is processed first.
        let lo = 1usize;
        let hi = state.max_part.min(state.remaining);
        // We want descending order in the result, so push in reverse (ascending).
        for part in lo..=hi {
            prefix_lens.push(prefix.len());
            state_stack.push(State {
                remaining: state.remaining - part,
                max_part: part,
            });
            // We cannot share prefix across children; we need to encode the
            // prefix *at push time*.  Instead, encode as (prefix_snapshot, state).
        }
        // The above is incorrect because we don't snapshot the prefix.
        // Let's redo with a proper approach.
        let _ = state_stack
            .drain(state_stack.len() - (hi - lo + 1)..)
            .collect::<Vec<_>>();
        let _ = prefix_lens.drain(prefix_lens.len() - (hi - lo + 1)..).collect::<Vec<_>>();

        // Use recursive enumerator that yields directly.
        enumerate_parts_into(state.remaining, state.max_part, &mut prefix, &mut result);
    }
    result
}

/// Internal recursive helper.  Appends all partitions of `remaining` into
/// non-increasing parts of size ≤ `max_part` to `out`, using `current` as the
/// prefix accumulated so far.
fn enumerate_parts_into(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        out.push(current.clone());
        return;
    }
    // Try each possible next part from max_part down to 1.
    let hi = max_part.min(remaining);
    for part in (1..=hi).rev() {
        current.push(part);
        enumerate_parts_into(remaining - part, part, current, out);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// restricted_partitions
// ---------------------------------------------------------------------------

/// Enumerate all integer partitions of `n` where no part exceeds `max_part`.
///
/// Each partition is in non-increasing order.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::partitions::restricted_partitions;
/// let parts = restricted_partitions(6, 3);
/// // [3,3], [3,2,1], [3,1,1,1], [2,2,2], [2,2,1,1], [2,1,1,1,1], [1,1,1,1,1,1]
/// assert_eq!(parts.len(), 7);
/// ```
pub fn restricted_partitions(n: usize, max_part: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    enumerate_parts_into(n, max_part, &mut current, &mut result);
    result
}

// ---------------------------------------------------------------------------
// conjugate_partition
// ---------------------------------------------------------------------------

/// Compute the conjugate (transpose) of an integer partition.
///
/// The conjugate partition λ' is defined by:
///   λ'_j = #{i : λ_i ≥ j}
///
/// The input partition must be in non-increasing order.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::partitions::conjugate_partition;
/// // Conjugate of [4, 2, 1] is [3, 2, 1, 1].
/// assert_eq!(conjugate_partition(&[4, 2, 1]), vec![3, 2, 1, 1]);
/// assert_eq!(conjugate_partition(&[3, 3, 2]), vec![3, 3, 2]);
/// ```
pub fn conjugate_partition(partition: &[usize]) -> Vec<usize> {
    if partition.is_empty() {
        return Vec::new();
    }
    let max_part = *partition.iter().max().expect("non-empty partition");
    let mut conj = Vec::with_capacity(max_part);
    for j in 1..=max_part {
        let count = partition.iter().filter(|&&p| p >= j).count();
        conj.push(count);
    }
    conj
}

// ---------------------------------------------------------------------------
// YoungDiagram
// ---------------------------------------------------------------------------

/// A Young diagram representation of an integer partition.
///
/// Each row i has λ_i boxes.  Use [`YoungDiagram::to_string`] or the
/// [`fmt::Display`] impl to render it as ASCII art.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YoungDiagram {
    /// The partition in non-increasing order.
    pub partition: Vec<usize>,
}

impl YoungDiagram {
    /// Create a `YoungDiagram` from a partition.
    ///
    /// The input need not be sorted; it will be sorted into non-increasing
    /// order automatically.
    pub fn new(mut partition: Vec<usize>) -> Self {
        partition.retain(|&p| p > 0);
        partition.sort_unstable_by(|a, b| b.cmp(a));
        Self { partition }
    }

    /// Return the number of rows (length of the partition).
    pub fn rows(&self) -> usize {
        self.partition.len()
    }

    /// Return the number of columns (size of the largest part).
    pub fn columns(&self) -> usize {
        self.partition.first().copied().unwrap_or(0)
    }

    /// Return the size n of the partition (sum of parts).
    pub fn size(&self) -> usize {
        self.partition.iter().sum()
    }

    /// Return the conjugate Young diagram.
    pub fn conjugate(&self) -> Self {
        YoungDiagram {
            partition: conjugate_partition(&self.partition),
        }
    }

    /// Render as an ASCII string using box characters.
    pub fn ascii(&self) -> String {
        let mut s = String::new();
        for &row_len in &self.partition {
            for _ in 0..row_len {
                s.push_str("[]");
            }
            s.push('\n');
        }
        s
    }
}

impl fmt::Display for YoungDiagram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ascii())
    }
}

/// Render a Young diagram from a partition slice and return it as a `String`.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::partitions::young_diagram;
/// let s = young_diagram(&[3, 2, 1]);
/// assert!(s.contains("[][]"));
/// ```
pub fn young_diagram(partition: &[usize]) -> String {
    YoungDiagram::new(partition.to_vec()).ascii()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_count() {
        let expected = [1u64, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(partition_count(n), Some(exp), "p({n}) should be {exp}");
        }
    }

    #[test]
    fn test_partition_count_large() {
        // p(100) = 190569292
        assert_eq!(partition_count(100), Some(190569292));
    }

    #[test]
    fn test_enumerate_partitions_0() {
        let p = enumerate_partitions(0);
        assert_eq!(p, vec![vec![]]);
    }

    #[test]
    fn test_enumerate_partitions_4() {
        let p = enumerate_partitions(4);
        assert_eq!(p.len(), 5);
        // All parts non-increasing.
        for part in &p {
            for w in part.windows(2) {
                assert!(w[0] >= w[1], "partition not non-increasing: {part:?}");
            }
        }
        // Sums to n.
        for part in &p {
            assert_eq!(part.iter().sum::<usize>(), 4);
        }
    }

    #[test]
    fn test_enumerate_partitions_count_matches() {
        for n in 0..=12 {
            let cnt = partition_count(n).expect("count") as usize;
            let enum_cnt = enumerate_partitions(n).len();
            assert_eq!(cnt, enum_cnt, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_restricted_partitions() {
        let p = restricted_partitions(6, 3);
        assert_eq!(p.len(), 7);
        for part in &p {
            assert!(
                part.iter().all(|&x| x <= 3),
                "part exceeds max: {part:?}"
            );
            assert_eq!(part.iter().sum::<usize>(), 6);
        }
    }

    #[test]
    fn test_conjugate_partition() {
        // [4, 2, 1] → [3, 2, 1, 1]
        assert_eq!(conjugate_partition(&[4, 2, 1]), vec![3, 2, 1, 1]);
        // Self-conjugate [3, 2, 1] → [3, 2, 1]
        assert_eq!(conjugate_partition(&[3, 2, 1]), vec![3, 2, 1]);
        // [3, 3, 2] → [3, 3, 2]
        assert_eq!(conjugate_partition(&[3, 3, 2]), vec![3, 3, 2]);
        // Double conjugate = original
        let p = vec![5usize, 3, 1];
        assert_eq!(conjugate_partition(&conjugate_partition(&p)), p);
    }

    #[test]
    fn test_young_diagram() {
        let s = young_diagram(&[3, 2, 1]);
        let lines: Vec<&str> = s.trim_end().lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "[][][]");
        assert_eq!(lines[1], "[][]");
        assert_eq!(lines[2], "[]");
    }

    #[test]
    fn test_young_diagram_struct() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        assert_eq!(d.rows(), 3);
        assert_eq!(d.columns(), 3);
        assert_eq!(d.size(), 6);
        let conj = d.conjugate();
        assert_eq!(conj.partition, vec![3, 2, 1]);
    }
}
