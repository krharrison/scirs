//! Schur Polynomials and Symmetric Functions
//!
//! Schur polynomials are a basis for the ring of symmetric functions, indexed
//! by integer partitions. They arise in representation theory, combinatorics,
//! algebraic geometry, and mathematical physics.
//!
//! This module implements:
//! - Integer partitions with Young diagram operations
//! - Schur polynomials via Jacobi-Trudi determinant
//! - Elementary, complete homogeneous, power sum, and monomial symmetric polynomials
//! - Littlewood-Richardson coefficients

use crate::error::{SpecialError, SpecialResult};
use std::collections::HashMap;

/// An integer partition λ = (λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0)
///
/// Represents a non-increasing sequence of positive integers (parts).
/// The weight (or size) is the sum of all parts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    /// Parts in non-increasing order; all parts > 0
    pub parts: Vec<usize>,
}

impl Partition {
    /// Create a partition from a vector of parts.
    ///
    /// Validates that parts are in non-increasing order and all positive.
    ///
    /// # Errors
    /// Returns `SpecialError::ValueError` if parts are not non-increasing or contain zeros.
    pub fn new(parts: Vec<usize>) -> SpecialResult<Self> {
        // Remove trailing zeros
        let mut parts = parts;
        while parts.last() == Some(&0) {
            parts.pop();
        }
        // Validate non-increasing
        for i in 1..parts.len() {
            if parts[i] > parts[i - 1] {
                return Err(SpecialError::ValueError(format!(
                    "Partition parts must be non-increasing, but parts[{}]={} > parts[{}]={}",
                    i,
                    parts[i],
                    i - 1,
                    parts[i - 1]
                )));
            }
        }
        Ok(Partition { parts })
    }

    /// Number of parts (length of the partition).
    pub fn n(&self) -> usize {
        self.parts.len()
    }

    /// Sum of all parts (weight / size of the partition).
    pub fn weight(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Conjugate partition (transpose of the Young diagram).
    ///
    /// If λ has diagram with rows λ_i, the conjugate λ' has column lengths
    /// λ'_j = #{i : λ_i ≥ j}.
    pub fn conjugate(&self) -> Partition {
        if self.parts.is_empty() {
            return Partition { parts: vec![] };
        }
        let max_part = *self.parts.iter().max().unwrap_or(&0);
        let mut conj_parts = Vec::with_capacity(max_part);
        for j in 1..=max_part {
            let count = self.parts.iter().filter(|&&p| p >= j).count();
            if count > 0 {
                conj_parts.push(count);
            }
        }
        Partition { parts: conj_parts }
    }

    /// Check if this partition dominates another in the dominance order.
    ///
    /// λ dominates μ (λ ≥ μ) iff for all k: λ_1+...+λ_k ≥ μ_1+...+μ_k.
    /// Both must have the same weight.
    pub fn dominates(&self, other: &Partition) -> bool {
        if self.weight() != other.weight() {
            return false;
        }
        let max_len = self.parts.len().max(other.parts.len());
        let mut sum_self = 0usize;
        let mut sum_other = 0usize;
        for k in 0..max_len {
            sum_self += self.parts.get(k).copied().unwrap_or(0);
            sum_other += other.parts.get(k).copied().unwrap_or(0);
            if sum_self < sum_other {
                return false;
            }
        }
        true
    }

    /// Check whether this is the empty partition.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Symmetric polynomial bases
// ────────────────────────────────────────────────────────────────────────────

/// Compute the k-th elementary symmetric polynomial e_k(x_1,...,x_n).
///
/// e_k = Σ_{i_1 < i_2 < ... < i_k} x_{i_1} * x_{i_2} * ... * x_{i_k}
///
/// Uses dynamic programming for efficiency.
/// e_0 = 1, e_k = 0 for k > n.
pub fn elementary_symmetric(k: usize, vars: &[f64]) -> f64 {
    let n = vars.len();
    if k > n {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    // DP table: dp[j] = e_j(vars[0..i])
    let mut dp = vec![0.0f64; k + 1];
    dp[0] = 1.0;
    for &x in vars.iter() {
        // Update in reverse to avoid using x twice (0-1 knapsack style)
        for j in (1..=k).rev() {
            dp[j] += dp[j - 1] * x;
        }
    }
    dp[k]
}

/// Compute the k-th complete homogeneous symmetric polynomial h_k(x_1,...,x_n).
///
/// h_k = Σ_{i_1 ≤ i_2 ≤ ... ≤ i_k} x_{i_1} * x_{i_2} * ... * x_{i_k}
///
/// Uses DP over variables with repetition allowed.
/// h_0 = 1, h_k computed by DP.
pub fn complete_homogeneous(k: usize, vars: &[f64]) -> f64 {
    let n = vars.len();
    if k == 0 {
        return 1.0;
    }
    if n == 0 {
        return 0.0;
    }
    // dp[v][j] = h_j restricted to first v variables
    // Recurrence: h_k(x_1,...,x_v) = Σ_{t=0}^{k} h_{k-t}(x_1,...,x_{v-1}) * x_v^t
    // Use 1D rolling DP
    let mut dp = vec![0.0f64; k + 1];
    dp[0] = 1.0;
    for &x in vars.iter() {
        // For each new variable x, update dp using "complete" (unbounded) inclusion
        // dp_new[j] = Σ_{t=0}^{j} dp_old[j-t] * x^t
        let old_dp = dp.clone();
        for j in 1..=k {
            let mut acc = old_dp[j];
            let mut xp = x;
            for t in 1..=j {
                acc += old_dp[j - t] * xp;
                xp *= x;
            }
            dp[j] = acc;
        }
    }
    dp[k]
}

/// Compute the k-th power sum symmetric polynomial p_k(x_1,...,x_n).
///
/// p_k = Σ_i x_i^k
/// p_0 = n (number of variables)
pub fn power_sum(k: usize, vars: &[f64]) -> f64 {
    if k == 0 {
        return vars.len() as f64;
    }
    vars.iter().map(|&x| x.powi(k as i32)).sum()
}

/// Compute the monomial symmetric polynomial m_λ(x_1,...,x_n).
///
/// m_λ = Σ_{α ~ λ} x^α where the sum is over distinct permutations of λ.
pub fn monomial_symmetric(partition: &Partition, vars: &[f64]) -> f64 {
    let n = vars.len();
    let lambda = &partition.parts;
    if lambda.is_empty() {
        return 1.0;
    }
    if lambda.len() > n {
        return 0.0;
    }

    // Pad lambda with zeros to length n
    let mut exponents = vec![0usize; n];
    for (i, &p) in lambda.iter().enumerate() {
        exponents[i] = p;
    }

    // Enumerate all distinct permutations of exponents.
    // Start from the lexicographically smallest (sorted ascending) and advance.
    let mut sum = 0.0f64;
    let mut perm = exponents.clone();
    perm.sort_unstable(); // start from smallest permutation

    // Generate all distinct permutations using next_permutation (lexicographic ascending)
    loop {
        let term: f64 = vars
            .iter()
            .zip(perm.iter())
            .map(|(&x, &e)| x.powi(e as i32))
            .product();
        sum += term;
        if !next_permutation_ascending(&mut perm) {
            break;
        }
    }
    sum
}

/// Advance to the next permutation in ascending lexicographic order.
/// Returns false if the permutation is already the last (descending) one.
fn next_permutation_ascending(arr: &mut [usize]) -> bool {
    let n = arr.len();
    if n <= 1 {
        return false;
    }
    // Find the rightmost position i such that arr[i] < arr[i+1]
    let mut i = n - 1;
    loop {
        if i == 0 {
            return false; // already at last permutation
        }
        i -= 1;
        if arr[i] < arr[i + 1] {
            break;
        }
    }
    // Find the smallest element in arr[i+1..] that is greater than arr[i]
    let mut j = n - 1;
    while arr[j] <= arr[i] {
        j -= 1;
    }
    arr.swap(i, j);
    // Reverse arr[i+1..] to get the next permutation
    arr[i + 1..].reverse();
    true
}

// ────────────────────────────────────────────────────────────────────────────
// Schur polynomial via Jacobi-Trudi identity
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Schur polynomial s_λ(x_1,...,x_n).
///
/// Uses the Jacobi-Trudi identity:
/// s_λ = det(h_{λ_i - i + j})_{1 ≤ i,j ≤ ℓ(λ)}
///
/// where h_k are complete homogeneous symmetric polynomials.
///
/// # Arguments
/// * `partition` - Integer partition λ
/// * `vars` - Evaluation variables x_1,...,x_n
///
/// # Returns
/// Value of the Schur polynomial
pub fn schur_polynomial(partition: &Partition, vars: &[f64]) -> f64 {
    if partition.is_empty() {
        return 1.0;
    }
    let lambda = &partition.parts;
    let l = lambda.len();

    // Build matrix M where M[i][j] = h_{lambda[i] - i + j}  (0-indexed: h_{lambda[i]-i+j})
    // using 0-indexed: M[i][j] = h_{lambda[i] - i + j}  for i,j in [0,l)
    let max_k = lambda[0] + l + 2;
    // Precompute h_0, h_1, ..., h_{max_k}
    let mut h = vec![0.0f64; max_k + 1];
    for k in 0..=max_k {
        h[k] = complete_homogeneous(k, vars);
    }

    // Build l×l matrix
    let mut mat = vec![vec![0.0f64; l]; l];
    for i in 0..l {
        for j in 0..l {
            let idx = lambda[i] as isize - i as isize + j as isize;
            if idx < 0 {
                mat[i][j] = 0.0;
            } else if idx as usize <= max_k {
                mat[i][j] = h[idx as usize];
            } else {
                mat[i][j] = 0.0;
            }
        }
    }

    // Compute determinant by LU decomposition (Gaussian elimination)
    determinant_f64(&mat)
}

/// Compute determinant of a square matrix by Gaussian elimination with partial pivoting.
fn determinant_f64(mat: &[Vec<f64>]) -> f64 {
    let n = mat.len();
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return mat[0][0];
    }
    let mut a = mat.to_vec();
    let mut sign = 1.0f64;

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return 0.0; // Singular
        }
        if max_row != col {
            a.swap(col, max_row);
            sign = -sign;
        }
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                let val = a[col][k];
                a[row][k] -= factor * val;
            }
        }
    }

    let mut det = sign;
    for i in 0..n {
        det *= a[i][i];
    }
    det
}

// ────────────────────────────────────────────────────────────────────────────
// Littlewood-Richardson coefficients
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Littlewood-Richardson coefficient c^ν_{λ,μ}.
///
/// This is the multiplicity of s_ν in the product s_λ * s_μ.
/// Equivalently, it counts LR-tableaux: semistandard Young tableaux of skew
/// shape ν/λ filled with the content of μ satisfying the reverse lattice word
/// (ballot sequence) condition.
///
/// # Arguments
/// * `lambda` - Partition λ
/// * `mu` - Partition μ
/// * `nu` - Partition ν
///
/// # Returns
/// The LR coefficient c^ν_{λ,μ} ≥ 0.
pub fn littlewood_richardson(lambda: &Partition, mu: &Partition, nu: &Partition) -> i64 {
    // Necessary conditions
    if nu.weight() != lambda.weight() + mu.weight() {
        return 0;
    }
    // nu must contain lambda (lambda ⊆ nu as Young diagrams)
    if !contains_partition(nu, lambda) {
        return 0;
    }
    // Compute via skew semistandard YT enumeration with reverse lattice word
    count_lr_tableaux(nu, lambda, mu)
}

/// Check if ν contains λ as a Young diagram (λ_i ≤ ν_i for all i).
fn contains_partition(nu: &Partition, lambda: &Partition) -> bool {
    if lambda.parts.len() > nu.parts.len() {
        return false;
    }
    for (i, &lp) in lambda.parts.iter().enumerate() {
        if lp > nu.parts.get(i).copied().unwrap_or(0) {
            return false;
        }
    }
    true
}

/// Count LR tableaux for skew shape ν/λ with content μ.
///
/// An LR-tableau is a filling of the skew shape ν/λ with positive integers such that:
/// 1. Rows weakly increase (→), columns strictly increase (↓)
/// 2. The reading word (right-to-left, top-to-bottom) is a ballot/lattice word
///    (at any prefix, #(i) ≥ #(i+1) for all i)
/// 3. The content is μ (μ_1 ones, μ_2 twos, ...)
fn count_lr_tableaux(nu: &Partition, lambda: &Partition, mu: &Partition) -> i64 {
    let n_rows = nu.parts.len();
    // Build the skew cells: list of (row, col) cells in ν/λ by rows
    let mut skew_rows: Vec<Vec<usize>> = Vec::new();
    for r in 0..n_rows {
        let nu_r = nu.parts.get(r).copied().unwrap_or(0);
        let la_r = lambda.parts.get(r).copied().unwrap_or(0);
        if nu_r > la_r {
            let row_cells: Vec<usize> = (la_r..nu_r).collect();
            skew_rows.push(row_cells);
        } else {
            skew_rows.push(vec![]);
        }
    }

    // Total number of cells to fill
    let total_cells: usize = skew_rows.iter().map(|r| r.len()).sum();
    if total_cells != mu.weight() {
        return 0;
    }
    if total_cells == 0 {
        return 1; // Empty filling
    }

    // Flatten cells in reading order (right-to-left within each row, top-to-bottom)
    // But for filling we go row by row left to right, and validate by reading word
    // Content: mu[0] copies of 1, mu[1] copies of 2, etc.
    let max_label = mu.parts.len();
    if max_label == 0 {
        return if total_cells == 0 { 1 } else { 0 };
    }

    // Build 2D grid for filling: grid[row][col] = label (0 = unfilled)
    // We use backtracking
    let mut grid: Vec<Vec<usize>> = (0..n_rows)
        .map(|r| vec![0usize; nu.parts.get(r).copied().unwrap_or(0)])
        .collect();

    // Build list of cells in row-major order for filling
    let mut cells: Vec<(usize, usize)> = Vec::new();
    for r in 0..n_rows {
        let nu_r = nu.parts.get(r).copied().unwrap_or(0);
        let la_r = lambda.parts.get(r).copied().unwrap_or(0);
        for c in la_r..nu_r {
            cells.push((r, c));
        }
    }
    // Pre-fill lambda cells with a sentinel large value for column increase check
    for r in 0..n_rows {
        let la_r = lambda.parts.get(r).copied().unwrap_or(0);
        for c in 0..la_r {
            grid[r][c] = usize::MAX; // sentinel: these are fixed by lambda
        }
    }

    let mut content = vec![0usize; max_label + 1]; // content[i] = how many i's placed
    let remaining: Vec<usize> = mu.parts.clone();

    let count = backtrack_lr(
        &cells,
        0,
        &mut grid,
        &mut content,
        &remaining,
        max_label,
        lambda,
        nu,
    );
    count
}

/// Backtracking enumeration of LR-tableaux.
#[allow(clippy::too_many_arguments)]
fn backtrack_lr(
    cells: &[(usize, usize)],
    idx: usize,
    grid: &mut Vec<Vec<usize>>,
    content: &mut Vec<usize>,
    remaining: &[usize],
    max_label: usize,
    lambda: &Partition,
    nu: &Partition,
) -> i64 {
    if idx == cells.len() {
        // Check ballot condition on the reading word
        if is_ballot_sequence(grid, lambda, nu) {
            return 1;
        } else {
            return 0;
        }
    }

    let (r, c) = cells[idx];
    let mut count = 0i64;

    for label in 1..=max_label {
        if content[label] >= remaining[label - 1] {
            continue;
        }
        // Check row weakly increasing: label >= grid[r][c-1] (if c-1 is in skew)
        let la_r = lambda.parts.get(r).copied().unwrap_or(0);
        if c > la_r {
            // Previous cell in same row
            let prev = grid[r][c - 1];
            if prev != usize::MAX && label < prev {
                continue;
            }
        }
        // Check column strictly increasing: label > grid[r-1][c] (if r-1 exists and has col c)
        if r > 0 {
            let nr_prev = nu.parts.get(r - 1).copied().unwrap_or(0);
            if c < nr_prev {
                let above = grid[r - 1][c];
                if above != usize::MAX && label <= above {
                    continue;
                }
                if above == usize::MAX && label == 0 {
                    continue; // shouldn't happen
                }
            }
        }

        grid[r][c] = label;
        content[label] += 1;
        count += backtrack_lr(
            cells,
            idx + 1,
            grid,
            content,
            remaining,
            max_label,
            lambda,
            nu,
        );
        grid[r][c] = 0;
        content[label] -= 1;
    }
    count
}

/// Check whether the reading word of the tableau is a ballot (lattice) sequence.
///
/// The reading word is formed by reading each row right-to-left, going top-to-bottom.
/// Ballot condition: at every prefix, the number of k's ≥ number of (k+1)'s for all k.
fn is_ballot_sequence(grid: &[Vec<usize>], lambda: &Partition, nu: &Partition) -> bool {
    let n_rows = nu.parts.len();
    let max_label = grid
        .iter()
        .flat_map(|r| r.iter().copied())
        .filter(|&v| v != usize::MAX && v > 0)
        .max()
        .unwrap_or(0);
    if max_label == 0 {
        return true;
    }
    let mut prefix_count = vec![0usize; max_label + 1];
    for r in 0..n_rows {
        let nu_r = nu.parts.get(r).copied().unwrap_or(0);
        let la_r = lambda.parts.get(r).copied().unwrap_or(0);
        // Read right-to-left
        for c in (la_r..nu_r).rev() {
            let v = grid[r][c];
            if v == 0 || v == usize::MAX {
                continue;
            }
            prefix_count[v] += 1;
            // Check ballot: for each k < max_label, count[k] >= count[k+1]
            for k in 1..max_label {
                if prefix_count[k] < prefix_count[k + 1] {
                    return false;
                }
            }
        }
    }
    true
}

// ────────────────────────────────────────────────────────────────────────────
// Higher-level utility: product of Schur polynomials
// ────────────────────────────────────────────────────────────────────────────

/// Expand the product s_λ * s_μ in terms of Schur polynomials.
///
/// Returns a map from partition ν to the LR-coefficient c^ν_{λ,μ}.
/// Only partitions ν with weight |λ|+|μ| are considered.
///
/// # Arguments
/// * `lambda` - First partition
/// * `mu` - Second partition
/// * `max_parts` - Maximum number of parts to consider in ν
///
/// # Returns
/// HashMap from Partition to coefficient
pub fn schur_product_expansion(
    lambda: &Partition,
    mu: &Partition,
    max_parts: usize,
) -> HashMap<Partition, i64> {
    let target_weight = lambda.weight() + mu.weight();
    let mut result = HashMap::new();
    // Enumerate all partitions of target_weight with at most max_parts parts
    let partitions = generate_partitions(target_weight, max_parts);
    for nu in partitions {
        let coeff = littlewood_richardson(lambda, mu, &nu);
        if coeff != 0 {
            result.insert(nu, coeff);
        }
    }
    result
}

/// Generate all partitions of `n` with at most `max_parts` parts.
pub fn generate_partitions(n: usize, max_parts: usize) -> Vec<Partition> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    gen_parts(n, n, max_parts, &mut current, &mut result);
    result
}

fn gen_parts(
    remaining: usize,
    max_part: usize,
    max_parts: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Partition>,
) {
    if remaining == 0 {
        if !current.is_empty() {
            result.push(Partition {
                parts: current.clone(),
            });
        }
        return;
    }
    if current.len() >= max_parts {
        return;
    }
    let limit = max_part.min(remaining);
    for p in (1..=limit).rev() {
        current.push(p);
        gen_parts(remaining - p, p, max_parts, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_partition_new_valid() {
        let p = Partition::new(vec![3, 2, 1]).expect("valid partition");
        assert_eq!(p.parts, vec![3, 2, 1]);
        assert_eq!(p.weight(), 6);
        assert_eq!(p.n(), 3);
    }

    #[test]
    fn test_partition_new_invalid() {
        let result = Partition::new(vec![2, 3, 1]);
        assert!(result.is_err(), "Non-decreasing should be rejected");
    }

    #[test]
    fn test_partition_conjugate() {
        // λ = (3,2,1) has conjugate (3,2,1) (self-conjugate staircase)
        let p = Partition::new(vec![3, 2, 1]).expect("valid");
        let conj = p.conjugate();
        assert_eq!(conj.parts, vec![3, 2, 1]);

        // λ = (3,3) has conjugate (2,2,2)
        let p2 = Partition::new(vec![3, 3]).expect("valid");
        let conj2 = p2.conjugate();
        assert_eq!(conj2.parts, vec![2, 2, 2]);
    }

    #[test]
    fn test_partition_dominates() {
        let p1 = Partition::new(vec![3, 1]).expect("valid");
        let p2 = Partition::new(vec![2, 2]).expect("valid");
        // (3,1) dominates (2,2): 3≥2, 3+1=4≥2+2=4 ✓
        assert!(p1.dominates(&p2));
        // (2,2) does not dominate (3,1): 2 < 3
        assert!(!p2.dominates(&p1));
    }

    #[test]
    fn test_elementary_symmetric() {
        let vars = [1.0, 2.0, 3.0];
        // e_0 = 1
        assert_relative_eq!(elementary_symmetric(0, &vars), 1.0, epsilon = 1e-12);
        // e_1 = 1+2+3 = 6
        assert_relative_eq!(elementary_symmetric(1, &vars), 6.0, epsilon = 1e-12);
        // e_2 = 1*2 + 1*3 + 2*3 = 2+3+6 = 11
        assert_relative_eq!(elementary_symmetric(2, &vars), 11.0, epsilon = 1e-12);
        // e_3 = 1*2*3 = 6
        assert_relative_eq!(elementary_symmetric(3, &vars), 6.0, epsilon = 1e-12);
        // e_4 = 0 (k > n)
        assert_relative_eq!(elementary_symmetric(4, &vars), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_complete_homogeneous() {
        let vars = [2.0, 3.0];
        // h_0 = 1
        assert_relative_eq!(complete_homogeneous(0, &vars), 1.0, epsilon = 1e-12);
        // h_1 = 2+3 = 5
        assert_relative_eq!(complete_homogeneous(1, &vars), 5.0, epsilon = 1e-12);
        // h_2 = 4+6+9 = 19  (x^2 + xy + y^2)
        assert_relative_eq!(complete_homogeneous(2, &vars), 19.0, epsilon = 1e-12);
    }

    #[test]
    fn test_power_sum() {
        let vars = [1.0, 2.0, 3.0];
        assert_relative_eq!(power_sum(0, &vars), 3.0, epsilon = 1e-12);
        assert_relative_eq!(power_sum(1, &vars), 6.0, epsilon = 1e-12);
        assert_relative_eq!(power_sum(2, &vars), 14.0, epsilon = 1e-12);
    }

    #[test]
    fn test_schur_partition_1_xyz() {
        // s_{(1)}(x,y,z) = x + y + z = p_1
        let vars = [1.5, 2.0, 3.5];
        let lambda = Partition::new(vec![1]).expect("valid");
        let s = schur_polynomial(&lambda, &vars);
        let p1 = power_sum(1, &vars);
        assert_relative_eq!(s, p1, epsilon = 1e-10);
    }

    #[test]
    fn test_schur_partition_2_xy() {
        // s_{(2)}(x,y) = x^2 + xy + y^2
        let x = 2.0f64;
        let y = 3.0f64;
        let vars = [x, y];
        let lambda = Partition::new(vec![2]).expect("valid");
        let s = schur_polynomial(&lambda, &vars);
        let expected = x * x + x * y + y * y;
        assert_relative_eq!(s, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_schur_empty_partition() {
        // s_{()} = 1
        let vars = [1.0, 2.0];
        let lambda = Partition::new(vec![]).expect("valid");
        let s = schur_polynomial(&lambda, &vars);
        assert_relative_eq!(s, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_littlewood_richardson_simple() {
        // c^{(2,1)}_{(1),(1,1)} = 1
        let lambda = Partition::new(vec![1]).expect("valid");
        let mu = Partition::new(vec![1, 1]).expect("valid");
        let nu = Partition::new(vec![2, 1]).expect("valid");
        let c = littlewood_richardson(&lambda, &mu, &nu);
        assert_eq!(c, 1, "LR coefficient c^(2,1)_(1),(1,1) should be 1");
    }

    #[test]
    fn test_littlewood_richardson_zero() {
        // c^{(3)}_{(1),(1)} = 0 (wrong weight: |λ|+|μ| = 2 but |ν| = 3)
        let lambda = Partition::new(vec![1]).expect("valid");
        let mu = Partition::new(vec![1]).expect("valid");
        let nu = Partition::new(vec![3]).expect("valid");
        let c = littlewood_richardson(&lambda, &mu, &nu);
        assert_eq!(c, 0);
    }

    #[test]
    fn test_littlewood_richardson_s1_s1() {
        // s_{(1)} * s_{(1)} = s_{(2)} + s_{(1,1)}
        // So c^{(2)}_{(1),(1)} = 1, c^{(1,1)}_{(1),(1)} = 1
        let lambda = Partition::new(vec![1]).expect("valid");
        let mu = Partition::new(vec![1]).expect("valid");

        let nu2 = Partition::new(vec![2]).expect("valid");
        let nu11 = Partition::new(vec![1, 1]).expect("valid");

        assert_eq!(littlewood_richardson(&lambda, &mu, &nu2), 1);
        assert_eq!(littlewood_richardson(&lambda, &mu, &nu11), 1);
    }

    #[test]
    fn test_monomial_symmetric() {
        let vars = [1.0, 2.0, 3.0];
        // m_{(1)}(x,y,z) = x+y+z
        let p = Partition::new(vec![1]).expect("valid");
        let m = monomial_symmetric(&p, &vars);
        assert_relative_eq!(m, 6.0, epsilon = 1e-12);

        // m_{(1,1)}(x,y,z) = xy+xz+yz = e_2
        let p11 = Partition::new(vec![1, 1]).expect("valid");
        let m11 = monomial_symmetric(&p11, &vars);
        let e2 = elementary_symmetric(2, &vars);
        assert_relative_eq!(m11, e2, epsilon = 1e-12);
    }

    #[test]
    fn test_generate_partitions() {
        let parts = generate_partitions(4, 4);
        // Partitions of 4: (4), (3,1), (2,2), (2,1,1), (1,1,1,1)
        assert_eq!(parts.len(), 5);
    }
}
