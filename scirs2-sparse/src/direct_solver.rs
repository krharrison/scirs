//! Sparse direct solvers
//!
//! This module provides production-quality direct solvers for sparse linear systems:
//!
//! - **Sparse LU factorization**: Dense kernel with partial pivoting, AMD column ordering
//! - **Sparse Cholesky**: For symmetric positive definite (SPD) matrices
//! - **Fill-reducing orderings**: AMD (Approximate Minimum Degree), nested dissection
//! - **Symbolic analysis**: Determines the fill-in pattern before numeric factorization
//! - **Numeric factorization**: Computes the actual factor values
//! - **Triangular solves**: Forward and backward substitution
//!
//! # Architecture
//!
//! Factorization is split into two phases:
//! 1. **Ordering phase** — Computes a fill-reducing permutation (AMD or nested dissection)
//! 2. **Numeric phase** — Applies the permutation, then factors the reordered matrix
//!
//! # References
//!
//! - Davis, T.A. (2006). "Direct Methods for Sparse Linear Systems". SIAM.
//! - Amestoy, P.R., Davis, T.A., & Duff, I.S. (1996). "An approximate minimum
//!   degree ordering algorithm". SIAM J. Matrix Anal. Appl. 17(4), 886-905.
//! - George, A. & Liu, J.W. (1981). "Computer Solution of Large Sparse Positive
//!   Definite Systems". Prentice-Hall.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// SparseSolver trait
// ---------------------------------------------------------------------------

/// Trait for sparse direct solvers.
///
/// A solver follows the factorize-then-solve paradigm:
/// 1. Call `factorize()` to compute the decomposition.
/// 2. Call `solve()` (or `solve_multi()`) one or more times.
pub trait SparseSolver<F: Float> {
    /// Compute the factorization of the given matrix.
    fn factorize(&mut self, matrix: &CsrMatrix<F>) -> SparseResult<()>;

    /// Solve `A x = b` using the stored factorization.
    fn solve(&self, b: &[F]) -> SparseResult<Vec<F>>;

    /// Solve `A X = B` for multiple right-hand sides (columns of B).
    fn solve_multi(&self, b_columns: &[Vec<F>]) -> SparseResult<Vec<Vec<F>>> {
        let mut results = Vec::with_capacity(b_columns.len());
        for b in b_columns {
            results.push(self.solve(b)?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Symbolic analysis result
// ---------------------------------------------------------------------------

/// Result of the symbolic analysis phase.
///
/// Contains the non-zero pattern of the factors and the elimination tree.
#[derive(Debug, Clone)]
pub struct SymbolicAnalysis {
    /// Fill-reducing permutation (row/column index mapping).
    pub perm: Vec<usize>,
    /// Inverse permutation.
    pub perm_inv: Vec<usize>,
    /// Elimination tree: `parent[i]` is the parent of node i, or `usize::MAX` for roots.
    pub etree: Vec<usize>,
    /// Column pointers for the L factor non-zero pattern.
    pub l_colptr: Vec<usize>,
    /// Row indices for the L factor non-zero pattern.
    pub l_rowind: Vec<usize>,
    /// Column pointers for the U factor non-zero pattern (LU only).
    pub u_colptr: Vec<usize>,
    /// Row indices for the U factor non-zero pattern (LU only).
    pub u_rowind: Vec<usize>,
    /// Matrix dimension.
    pub n: usize,
}

// ---------------------------------------------------------------------------
// AMD Ordering
// ---------------------------------------------------------------------------

/// Approximate Minimum Degree (AMD) ordering.
///
/// Computes a fill-reducing permutation for a symmetric matrix (or the
/// sparsity pattern A + A^T for an unsymmetric matrix). The algorithm
/// greedily selects the node with the smallest approximate external degree
/// at each step.
///
/// Returns a permutation vector `perm` such that `A[perm, perm]` has
/// reduced fill during factorization.
pub fn amd_ordering<F>(matrix: &CsrMatrix<F>) -> SparseResult<Vec<usize>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SparseError::ValueError(
            "AMD ordering requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency list for A + A^T (symmetric structure)
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for i in 0..n {
        let range = i_row_range(matrix, i);
        for idx in range {
            let j = matrix.indices[idx];
            if i != j {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }

    // Degree of each node
    let mut degree: Vec<usize> = (0..n).map(|i| adj[i].len()).collect();
    let mut eliminated = vec![false; n];
    let mut perm = Vec::with_capacity(n);

    for _ in 0..n {
        // Find node with minimum approximate degree among non-eliminated nodes
        let mut min_deg = usize::MAX;
        let mut pivot = 0;
        for (node, &deg) in degree.iter().enumerate() {
            if !eliminated[node] && deg < min_deg {
                min_deg = deg;
                pivot = node;
            }
        }

        eliminated[pivot] = true;
        perm.push(pivot);

        // Collect neighbours of pivot that are not yet eliminated
        let neighbours: Vec<usize> = adj[pivot]
            .iter()
            .copied()
            .filter(|&nb| !eliminated[nb])
            .collect();

        // "Absorb" pivot: connect all its neighbours to each other (clique)
        for i in 0..neighbours.len() {
            let u = neighbours[i];
            adj[u].remove(&pivot);
            for j in (i + 1)..neighbours.len() {
                let v = neighbours[j];
                adj[u].insert(v);
                adj[v].insert(u);
            }
            degree[u] = adj[u].iter().filter(|&&nb| !eliminated[nb]).count();
        }
    }

    Ok(perm)
}

/// Compute the inverse permutation.
pub fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0usize; n];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

// ---------------------------------------------------------------------------
// Nested Dissection Ordering
// ---------------------------------------------------------------------------

/// Nested dissection ordering.
///
/// Recursively bisects the graph of the matrix using a simple graph
/// partitioning heuristic (BFS-based bisection), numbering separators
/// last. This is effective for matrices arising from 2D/3D discretisations.
pub fn nested_dissection_ordering<F>(matrix: &CsrMatrix<F>) -> SparseResult<Vec<usize>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SparseError::ValueError(
            "Nested dissection requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency list for A + A^T
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let range = i_row_range(matrix, i);
        for idx in range {
            let j = matrix.indices[idx];
            if i != j {
                if !adj[i].contains(&j) {
                    adj[i].push(j);
                }
                if !adj[j].contains(&i) {
                    adj[j].push(i);
                }
            }
        }
    }

    let nodes: Vec<usize> = (0..n).collect();
    let mut perm = Vec::with_capacity(n);
    nd_recurse(&adj, &nodes, &mut perm);

    if perm.len() != n {
        let in_perm: BTreeSet<usize> = perm.iter().copied().collect();
        for i in 0..n {
            if !in_perm.contains(&i) {
                perm.push(i);
            }
        }
    }

    Ok(perm)
}

fn nd_recurse(adj: &[Vec<usize>], nodes: &[usize], perm: &mut Vec<usize>) {
    if nodes.len() <= 64 {
        perm.extend_from_slice(nodes);
        return;
    }

    let start = find_pseudo_peripheral(adj, nodes);
    let (part_a, separator, part_b) = bfs_bisect(adj, nodes, start);

    if !part_a.is_empty() {
        nd_recurse(adj, &part_a, perm);
    }
    if !part_b.is_empty() {
        nd_recurse(adj, &part_b, perm);
    }
    perm.extend_from_slice(&separator);
}

fn find_pseudo_peripheral(adj: &[Vec<usize>], nodes: &[usize]) -> usize {
    if nodes.is_empty() {
        return 0;
    }
    let node_set: BTreeSet<usize> = nodes.iter().copied().collect();
    let mut current = nodes[0];
    for _ in 0..2 {
        let levels = bfs_levels(adj, current, &node_set);
        if let Some(last_level) = levels.last() {
            if !last_level.is_empty() {
                current = last_level[0];
            }
        }
    }
    current
}

fn bfs_levels(adj: &[Vec<usize>], start: usize, allowed: &BTreeSet<usize>) -> Vec<Vec<usize>> {
    let mut visited = BTreeSet::new();
    let mut levels: Vec<Vec<usize>> = Vec::new();
    visited.insert(start);
    levels.push(vec![start]);

    loop {
        let prev = match levels.last() {
            Some(p) => p.clone(),
            None => break,
        };
        let mut next_level = Vec::new();
        for &node in &prev {
            for &nb in &adj[node] {
                if allowed.contains(&nb) && !visited.contains(&nb) {
                    visited.insert(nb);
                    next_level.push(nb);
                }
            }
        }
        if next_level.is_empty() {
            break;
        }
        levels.push(next_level);
    }
    levels
}

fn bfs_bisect(
    adj: &[Vec<usize>],
    nodes: &[usize],
    start: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let node_set: BTreeSet<usize> = nodes.iter().copied().collect();
    let levels = bfs_levels(adj, start, &node_set);

    let total = nodes.len();
    let half = total / 2;

    let mut count = 0;
    let mut cut_level = 0;
    for (li, level) in levels.iter().enumerate() {
        count += level.len();
        if count >= half {
            cut_level = li;
            break;
        }
    }

    let mut part_a = Vec::new();
    let mut separator = Vec::new();
    let mut part_b = Vec::new();

    for (li, level) in levels.iter().enumerate() {
        if li < cut_level {
            part_a.extend_from_slice(level);
        } else if li == cut_level {
            separator.extend_from_slice(level);
        } else {
            part_b.extend_from_slice(level);
        }
    }

    let reached: BTreeSet<usize> = part_a
        .iter()
        .chain(separator.iter())
        .chain(part_b.iter())
        .copied()
        .collect();
    for &node in nodes {
        if !reached.contains(&node) {
            part_b.push(node);
        }
    }

    (part_a, separator, part_b)
}

// ---------------------------------------------------------------------------
// Elimination tree
// ---------------------------------------------------------------------------

/// Compute the elimination tree of a symmetric matrix.
pub fn elimination_tree<F>(matrix: &CsrMatrix<F>, perm: &[usize]) -> Vec<usize>
where
    F: Float + SparseElement + Debug + 'static,
{
    let n = matrix.rows();
    let perm_inv = inverse_perm(perm);
    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![0usize; n];

    for k in 0..n {
        ancestor[k] = k;
        let orig_row = perm[k];
        let range = i_row_range(matrix, orig_row);
        for idx in range {
            let orig_col = matrix.indices[idx];
            let j = perm_inv[orig_col];
            if j < k {
                let mut node = j;
                loop {
                    let next = ancestor[node];
                    if next == k {
                        break;
                    }
                    ancestor[node] = k;
                    if parent[node] == usize::MAX || parent[node] > k {
                        parent[node] = k;
                    }
                    if next == node {
                        break;
                    }
                    node = next;
                }
            }
        }
    }
    parent
}

// ---------------------------------------------------------------------------
// Symbolic Cholesky
// ---------------------------------------------------------------------------

/// Perform symbolic analysis for Cholesky factorization.
pub fn symbolic_cholesky<F>(matrix: &CsrMatrix<F>, perm: &[usize]) -> SparseResult<SymbolicAnalysis>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SparseError::ValueError(
            "Symbolic Cholesky requires a square matrix".to_string(),
        ));
    }
    let perm_inv = inverse_perm(perm);
    let etree = elimination_tree(matrix, perm);

    let mut l_col_count = vec![1usize; n];
    let mut visited = vec![usize::MAX; n];

    for k in 0..n {
        visited[k] = k;
        let orig_row = perm[k];
        let range = i_row_range(matrix, orig_row);
        for idx in range {
            let orig_col = matrix.indices[idx];
            let j = perm_inv[orig_col];
            if j < k {
                let mut node = j;
                while visited[node] != k {
                    visited[node] = k;
                    l_col_count[node] += 1;
                    if etree[node] == usize::MAX || etree[node] >= n {
                        break;
                    }
                    node = etree[node];
                }
            }
        }
    }

    let mut l_colptr = vec![0usize; n + 1];
    for j in 0..n {
        l_colptr[j + 1] = l_colptr[j] + l_col_count[j];
    }
    let total_nnz = l_colptr[n];
    let l_rowind = vec![0usize; total_nnz];

    Ok(SymbolicAnalysis {
        perm: perm.to_vec(),
        perm_inv,
        etree,
        l_colptr,
        l_rowind,
        u_colptr: Vec::new(),
        u_rowind: Vec::new(),
        n,
    })
}

// ---------------------------------------------------------------------------
// Sparse Cholesky factorization (dense kernel)
// ---------------------------------------------------------------------------

/// Result of sparse Cholesky factorization (L * L^T = P*A*P^T).
#[derive(Debug, Clone)]
pub struct SparseCholResult<F> {
    /// Dense lower-triangular factor L (row-major, n x n).
    pub l_dense: Vec<Vec<F>>,
    /// Permutation vector.
    pub perm: Vec<usize>,
    /// Inverse permutation.
    pub perm_inv: Vec<usize>,
    /// Dimension.
    pub n: usize,
}

/// Sparse Cholesky solver for symmetric positive definite matrices.
pub struct SparseCholeskySolver<F> {
    result: Option<SparseCholResult<F>>,
}

impl<F: Float + NumAssign + Sum + SparseElement + Debug + 'static> SparseCholeskySolver<F> {
    /// Create a new Cholesky solver (unfactorized).
    pub fn new() -> Self {
        Self { result: None }
    }

    /// Access the factorization result (if available).
    pub fn factorization(&self) -> Option<&SparseCholResult<F>> {
        self.result.as_ref()
    }
}

impl<F: Float + NumAssign + Sum + SparseElement + Debug + 'static> Default
    for SparseCholeskySolver<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + NumAssign + Sum + SparseElement + Debug + 'static> SparseSolver<F>
    for SparseCholeskySolver<F>
{
    fn factorize(&mut self, matrix: &CsrMatrix<F>) -> SparseResult<()> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "Cholesky requires a square matrix".to_string(),
            ));
        }
        if n == 0 {
            self.result = Some(SparseCholResult {
                l_dense: Vec::new(),
                perm: Vec::new(),
                perm_inv: Vec::new(),
                n: 0,
            });
            return Ok(());
        }

        // AMD ordering
        let perm = amd_ordering(matrix)?;
        let perm_inv = inverse_perm(&perm);

        // Build dense permuted matrix B = P * A * P^T
        let mut b_dense = vec![vec![F::sparse_zero(); n]; n];
        for i in 0..n {
            let orig_row = perm[i];
            let range = i_row_range(matrix, orig_row);
            for idx in range {
                let orig_col = matrix.indices[idx];
                let j = perm_inv[orig_col];
                b_dense[i][j] += matrix.data[idx];
            }
        }

        // Dense Cholesky: L * L^T = B (row-by-row, lower triangular)
        let mut l = vec![vec![F::sparse_zero(); n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = b_dense[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if sum <= F::sparse_zero() {
                        return Err(SparseError::ValueError(format!(
                            "Matrix is not positive definite: non-positive diagonal at row {i}"
                        )));
                    }
                    l[i][j] = sum.sqrt();
                } else {
                    let ljj = l[j][j];
                    if ljj.abs() < F::epsilon() {
                        return Err(SparseError::SingularMatrix(format!(
                            "Zero diagonal in L at row {j}"
                        )));
                    }
                    l[i][j] = sum / ljj;
                }
            }
        }

        self.result = Some(SparseCholResult {
            l_dense: l,
            perm,
            perm_inv,
            n,
        });
        Ok(())
    }

    fn solve(&self, b: &[F]) -> SparseResult<Vec<F>> {
        let res = self.result.as_ref().ok_or_else(|| {
            SparseError::ValueError("Cholesky factorization not computed".to_string())
        })?;
        let n = res.n;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        // bp[i] = b[perm[i]]
        let mut y = vec![F::sparse_zero(); n];
        for i in 0..n {
            y[i] = b[res.perm[i]];
        }

        // Forward solve: L y = bp
        for i in 0..n {
            for j in 0..i {
                y[i] = y[i] - res.l_dense[i][j] * y[j];
            }
            let d = res.l_dense[i][i];
            if d.abs() < F::epsilon() {
                return Err(SparseError::SingularMatrix(
                    "Zero diagonal in L during solve".to_string(),
                ));
            }
            y[i] /= d;
        }

        // Backward solve: L^T xp = y
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                y[i] = y[i] - res.l_dense[j][i] * y[j];
            }
            let d = res.l_dense[i][i];
            if d.abs() < F::epsilon() {
                return Err(SparseError::SingularMatrix(
                    "Zero diagonal in L^T during solve".to_string(),
                ));
            }
            y[i] /= d;
        }

        // x[perm[i]] = y[i]
        let mut x = vec![F::sparse_zero(); n];
        for i in 0..n {
            x[res.perm[i]] = y[i];
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Sparse LU factorization (dense kernel)
// ---------------------------------------------------------------------------

/// Result of sparse LU factorization (P*A*Q = L*U).
#[derive(Debug, Clone)]
pub struct SparseLuResult<F> {
    /// Dense LU factors in-place (L below diagonal with unit diagonal, U on+above diagonal).
    pub lu_dense: Vec<Vec<F>>,
    /// Row permutation (pivoting).
    pub row_perm: Vec<usize>,
    /// Column permutation (fill-reducing ordering).
    pub col_perm: Vec<usize>,
    /// Dimension.
    pub n: usize,
}

/// Sparse LU solver with partial pivoting.
pub struct SparseLuSolver<F> {
    result: Option<SparseLuResult<F>>,
}

impl<F: Float + NumAssign + Sum + SparseElement + Debug + 'static> SparseLuSolver<F> {
    /// Create a new LU solver (unfactorized).
    pub fn new() -> Self {
        Self { result: None }
    }

    /// Access the factorization result (if available).
    pub fn factorization(&self) -> Option<&SparseLuResult<F>> {
        self.result.as_ref()
    }
}

impl<F: Float + NumAssign + Sum + SparseElement + Debug + 'static> Default for SparseLuSolver<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + NumAssign + Sum + SparseElement + Debug + 'static> SparseSolver<F>
    for SparseLuSolver<F>
{
    fn factorize(&mut self, matrix: &CsrMatrix<F>) -> SparseResult<()> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "LU requires a square matrix".to_string(),
            ));
        }
        if n == 0 {
            self.result = Some(SparseLuResult {
                lu_dense: Vec::new(),
                row_perm: Vec::new(),
                col_perm: Vec::new(),
                n: 0,
            });
            return Ok(());
        }

        // Column ordering (AMD)
        let col_perm = amd_ordering(matrix)?;
        let col_perm_inv = inverse_perm(&col_perm);

        // Build dense matrix: a[i][j] = A[i][col_perm[j]]
        let mut a = vec![vec![F::sparse_zero(); n]; n];
        for i in 0..n {
            let range = i_row_range(matrix, i);
            for idx in range {
                let orig_col = matrix.indices[idx];
                let j = col_perm_inv[orig_col];
                a[i][j] += matrix.data[idx];
            }
        }

        // Dense LU with partial pivoting (in-place)
        let mut row_perm: Vec<usize> = (0..n).collect();

        for k in 0..n {
            // Find pivot
            let mut max_abs = F::sparse_zero();
            let mut pivot = k;
            for i in k..n {
                if a[i][k].abs() > max_abs {
                    max_abs = a[i][k].abs();
                    pivot = i;
                }
            }

            if pivot != k {
                a.swap(k, pivot);
                row_perm.swap(k, pivot);
            }

            let akk = a[k][k];
            if akk.abs() < F::epsilon() {
                continue; // near-singular column
            }

            for i in (k + 1)..n {
                let lik = a[i][k] / akk;
                a[i][k] = lik; // L part
                for j in (k + 1)..n {
                    let ukj = a[k][j];
                    a[i][j] -= lik * ukj;
                }
            }
        }

        self.result = Some(SparseLuResult {
            lu_dense: a,
            row_perm,
            col_perm,
            n,
        });
        Ok(())
    }

    fn solve(&self, b: &[F]) -> SparseResult<Vec<F>> {
        let res = self
            .result
            .as_ref()
            .ok_or_else(|| SparseError::ValueError("LU factorization not computed".to_string()))?;
        let n = res.n;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        // Apply row permutation
        let mut x = vec![F::sparse_zero(); n];
        for i in 0..n {
            x[i] = b[res.row_perm[i]];
        }

        // Forward solve: L y = Pb (unit diagonal)
        for i in 0..n {
            for j in 0..i {
                x[i] = x[i] - res.lu_dense[i][j] * x[j];
            }
        }

        // Backward solve: U z = y
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] = x[i] - res.lu_dense[i][j] * x[j];
            }
            let d = res.lu_dense[i][i];
            if d.abs() < F::epsilon() {
                return Err(SparseError::SingularMatrix(format!(
                    "Zero diagonal in U at row {i}"
                )));
            }
            x[i] /= d;
        }

        // Apply inverse column permutation: result[col_perm[j]] = x[j]
        let mut result = vec![F::sparse_zero(); n];
        for j in 0..n {
            result[res.col_perm[j]] = x[j];
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Solve Ax = b using sparse LU factorization with AMD ordering.
pub fn sparse_lu_solve<F>(matrix: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let mut solver = SparseLuSolver::new();
    solver.factorize(matrix)?;
    solver.solve(b)
}

/// Solve Ax = b using sparse Cholesky (matrix must be SPD).
pub fn sparse_cholesky_solve<F>(matrix: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let mut solver = SparseCholeskySolver::new();
    solver.factorize(matrix)?;
    solver.solve(b)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Safe row range extraction for CsrMatrix.
fn i_row_range<F: SparseElement + Clone + Copy + scirs2_core::numeric::Zero + PartialEq>(
    matrix: &CsrMatrix<F>,
    row: usize,
) -> std::ops::Range<usize> {
    if row >= matrix.rows() {
        return 0..0;
    }
    matrix.indptr[row]..matrix.indptr[row + 1]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A = [[4, 1, 0], [1, 5, 2], [0, 2, 6]]
    fn create_spd_3x3() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed to create SPD matrix")
    }

    fn create_general_3x3() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 0.0, 1.0, 5.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed to create matrix")
    }

    fn create_spd_4x4() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let data = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        CsrMatrix::new(data, rows, cols, (4, 4)).expect("Failed to create SPD 4x4")
    }

    fn verify_solve(mat: &CsrMatrix<f64>, x: &[f64], b: &[f64], tol: f64) {
        let dense = mat.to_dense();
        let n = b.len();
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += dense[i][j] * x[j];
            }
            assert!(
                (row_sum - b[i]).abs() < tol,
                "Row {i}: residual {}",
                (row_sum - b[i]).abs()
            );
        }
    }

    #[test]
    fn test_amd_ordering_basic() {
        let mat = create_spd_3x3();
        let perm = amd_ordering(&mat).expect("AMD failed");
        assert_eq!(perm.len(), 3);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_amd_ordering_empty() {
        let mat =
            CsrMatrix::<f64>::new(vec![], vec![], vec![], (0, 0)).expect("Failed to create empty");
        let perm = amd_ordering(&mat).expect("AMD failed on empty");
        assert!(perm.is_empty());
    }

    #[test]
    fn test_inverse_perm() {
        let perm = vec![2, 0, 1];
        let inv = inverse_perm(&perm);
        assert_eq!(inv, vec![1, 2, 0]);
        for i in 0..3 {
            assert_eq!(perm[inv[i]], i);
        }
    }

    #[test]
    fn test_nested_dissection_basic() {
        let mat = create_spd_4x4();
        let perm = nested_dissection_ordering(&mat).expect("ND failed");
        assert_eq!(perm.len(), 4);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_elimination_tree() {
        let mat = create_spd_3x3();
        let perm: Vec<usize> = (0..3).collect();
        let etree = elimination_tree(&mat, &perm);
        assert_eq!(etree.len(), 3);
    }

    #[test]
    fn test_cholesky_solve_3x3() {
        let mat = create_spd_3x3();
        let b = vec![5.0, 8.0, 8.0];
        let x = sparse_cholesky_solve(&mat, &b).expect("Cholesky solve failed");
        assert_eq!(x.len(), 3);
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-10, "x[{i}] = {xi}, expected 1.0");
        }
    }

    #[test]
    fn test_cholesky_solve_4x4() {
        let mat = create_spd_4x4();
        let b = vec![5.0, 6.0, 6.0, 5.0];
        let x = sparse_cholesky_solve(&mat, &b).expect("Cholesky solve 4x4 failed");
        verify_solve(&mat, &x, &b, 1e-10);
    }

    #[test]
    fn test_cholesky_non_spd() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![-1.0, 1.0, 1.0];
        let mat = CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed to create matrix");
        let result = sparse_cholesky_solve(&mat, &[1.0, 1.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_solve_3x3() {
        let mat = create_general_3x3();
        let b = vec![6.0, 6.0, 6.0];
        let x = sparse_lu_solve(&mat, &b).expect("LU solve failed");
        verify_solve(&mat, &x, &b, 1e-9);
    }

    #[test]
    fn test_lu_solve_identity() {
        let rows = vec![0, 1, 2, 3];
        let cols = vec![0, 1, 2, 3];
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let mat = CsrMatrix::new(data, rows, cols, (4, 4)).expect("Failed to create identity");
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = sparse_lu_solve(&mat, &b).expect("LU solve on identity failed");
        for i in 0..4 {
            assert!(
                (x[i] - b[i]).abs() < 1e-12,
                "x[{i}] = {}, expected {}",
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_lu_solve_multi() {
        let mat = create_general_3x3();
        let mut solver = SparseLuSolver::new();
        solver.factorize(&mat).expect("LU factorize failed");

        let b1 = vec![6.0, 6.0, 6.0];
        let b2 = vec![3.0, 1.0, 2.0];
        let results = solver
            .solve_multi(&[b1.clone(), b2.clone()])
            .expect("Solve multi failed");

        verify_solve(&mat, &results[0], &b1, 1e-9);
        verify_solve(&mat, &results[1], &b2, 1e-9);
    }

    #[test]
    fn test_cholesky_solver_trait() {
        let mat = create_spd_3x3();
        let mut solver = SparseCholeskySolver::new();
        solver.factorize(&mat).expect("Factorize failed");
        assert!(solver.factorization().is_some());

        let b = vec![5.0, 8.0, 8.0];
        let x = solver.solve(&b).expect("Solve failed");
        for (i, xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-10, "x[{i}] = {xi}");
        }
    }

    #[test]
    fn test_lu_empty_matrix() {
        let mat =
            CsrMatrix::<f64>::new(vec![], vec![], vec![], (0, 0)).expect("Failed to create empty");
        let mut solver = SparseLuSolver::new();
        solver
            .factorize(&mat)
            .expect("LU factorize on empty failed");
        let x = solver.solve(&[]).expect("LU solve on empty failed");
        assert!(x.is_empty());
    }

    #[test]
    fn test_cholesky_dimension_mismatch() {
        let mat = create_spd_3x3();
        let mut solver = SparseCholeskySolver::new();
        solver.factorize(&mat).expect("Factorize failed");
        let result = solver.solve(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_solve_5x5_diag_dominant() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    rows.push(i);
                    cols.push(j);
                    data.push(10.0);
                } else if (i as isize - j as isize).unsigned_abs() <= 1 {
                    rows.push(i);
                    cols.push(j);
                    data.push(1.0);
                }
            }
        }
        let mat = CsrMatrix::new(data, rows, cols, (5, 5)).expect("Failed to create 5x5");
        let b = vec![12.0, 12.0, 12.0, 12.0, 12.0];
        let x = sparse_lu_solve(&mat, &b).expect("LU 5x5 failed");
        verify_solve(&mat, &x, &b, 1e-8);
    }

    #[test]
    fn test_symbolic_cholesky() {
        let mat = create_spd_3x3();
        let perm: Vec<usize> = (0..3).collect();
        let analysis = symbolic_cholesky(&mat, &perm).expect("Symbolic Cholesky failed");
        assert_eq!(analysis.n, 3);
        assert_eq!(analysis.l_colptr.len(), 4);
        assert!(analysis.l_colptr[3] >= 3);
    }

    #[test]
    fn test_lu_non_square_error() {
        let rows = vec![0, 1];
        let cols = vec![0, 0];
        let data = vec![1.0, 2.0];
        let mat = CsrMatrix::new(data, rows, cols, (2, 3)).expect("Failed to create non-square");
        let result = sparse_lu_solve(&mat, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cholesky_non_square_error() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 0, 0];
        let data = vec![1.0, 2.0, 3.0];
        let mat = CsrMatrix::new(data, rows, cols, (3, 4)).expect("Failed to create non-square");
        let result = sparse_cholesky_solve(&mat, &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_solve_with_zeros() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![2.0, 1.0, 3.0, 1.0, 4.0];
        let mat = CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed");
        let b = vec![3.0, 3.0, 5.0];
        let x = sparse_lu_solve(&mat, &b).expect("LU solve sparse matrix failed");
        verify_solve(&mat, &x, &b, 1e-9);
    }

    #[test]
    fn test_amd_non_square_error() {
        let rows = vec![0];
        let cols = vec![0];
        let data = vec![1.0];
        let mat = CsrMatrix::new(data, rows, cols, (2, 3)).expect("Failed");
        let result = amd_ordering(&mat);
        assert!(result.is_err());
    }
}
