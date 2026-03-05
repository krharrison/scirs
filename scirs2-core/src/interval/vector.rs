//! Interval vectors and matrices with verified arithmetic.
//!
//! This module provides:
//!
//! * [`IntervalVector`] — a column vector whose entries are `Interval<f64>`.
//! * [`IntervalMatrix`] — a row-major rectangular matrix of `Interval<f64>`.
//! * [`gaussian_elimination_interval`] — a verified linear-system solver
//!   based on Gaussian elimination with interval pivoting; the solution
//!   interval is guaranteed to contain the true solution if the system is
//!   non-singular and the coefficient matrix / RHS are faithfully contained
//!   in the supplied intervals.

use super::interval::Interval;
use crate::error::{CoreError, CoreResult};
// Vec, String, format are available from std prelude

// ---------------------------------------------------------------------------
// IntervalVector
// ---------------------------------------------------------------------------

/// A column vector of `Interval<f64>` entries.
#[derive(Clone, Debug)]
pub struct IntervalVector {
    data: Vec<Interval<f64>>,
}

impl IntervalVector {
    /// Construct an interval vector from a `Vec`.
    #[inline]
    pub fn new(data: Vec<Interval<f64>>) -> Self {
        Self { data }
    }

    /// Construct a zero-width (point) interval vector from a slice of `f64`.
    pub fn from_point_slice(values: &[f64]) -> Self {
        Self {
            data: values.iter().copied().map(Interval::point).collect(),
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the vector has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Access the i-th entry.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&Interval<f64>> {
        self.data.get(i)
    }

    /// Mutable access to the i-th entry.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut Interval<f64>> {
        self.data.get_mut(i)
    }

    /// Iterator over entries.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, Interval<f64>> {
        self.data.iter()
    }

    /// Mutable iterator over entries.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, Interval<f64>> {
        self.data.iter_mut()
    }

    /// Element-wise addition `self + rhs`.
    pub fn add(&self, rhs: &Self) -> CoreResult<Self> {
        if self.len() != rhs.len() {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new("IntervalVector::add: length mismatch"),
            ));
        }
        Ok(Self {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        })
    }

    /// Element-wise subtraction `self - rhs`.
    pub fn sub(&self, rhs: &Self) -> CoreResult<Self> {
        if self.len() != rhs.len() {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new("IntervalVector::sub: length mismatch"),
            ));
        }
        Ok(Self {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        })
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: Interval<f64>) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * s).collect(),
        }
    }

    /// Dot product (returns a scalar interval).
    pub fn dot(&self, rhs: &Self) -> CoreResult<Interval<f64>> {
        if self.len() != rhs.len() {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new("IntervalVector::dot: length mismatch"),
            ));
        }
        let mut acc = Interval::point(0.0_f64);
        for (&a, &b) in self.data.iter().zip(rhs.data.iter()) {
            acc = acc + a * b;
        }
        Ok(acc)
    }

    /// Euclidean norm bound: `[0, mag(v)]` where `mag(v) = sqrt(sum(hi_i^2))`.
    pub fn norm_bound(&self) -> Interval<f64> {
        let sq_hi: f64 = self.data.iter().map(|x: &Interval<f64>| x.mag().powi(2)).sum();
        Interval::new(0.0, sq_hi.sqrt())
    }

    /// Check whether every component of `self` is contained in the
    /// corresponding component of `other`.
    pub fn contained_in(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b): (&Interval<f64>, &Interval<f64>)| b.contains_interval(a))
    }

    /// Return the underlying data slice.
    #[inline]
    pub fn as_slice(&self) -> &[Interval<f64>] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// IntervalMatrix
// ---------------------------------------------------------------------------

/// A row-major rectangular matrix of `Interval<f64>` entries.
///
/// Stored as a flat `Vec<Interval<f64>>` in row-major order with `rows * cols`
/// entries.
#[derive(Clone, Debug)]
pub struct IntervalMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Interval<f64>>,
}

impl IntervalMatrix {
    /// Construct a matrix from a flat row-major vector.
    ///
    /// Returns `Err` if `data.len() != rows * cols`.
    pub fn from_flat(rows: usize, cols: usize, data: Vec<Interval<f64>>) -> CoreResult<Self> {
        if data.len() != rows * cols {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new(&format!(
                    "IntervalMatrix::from_flat: expected {} elements, got {}",
                    rows * cols,
                    data.len()
                )),
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// Construct a matrix from a nested `Vec<Vec<f64>>` (rows × cols).
    pub fn from_f64_rows(rows_data: &[Vec<f64>]) -> CoreResult<Self> {
        let n_rows = rows_data.len();
        if n_rows == 0 {
            return Ok(Self {
                rows: 0,
                cols: 0,
                data: Vec::new(),
            });
        }
        let n_cols = rows_data[0].len();
        for (i, row) in rows_data.iter().enumerate() {
            let row: &Vec<f64> = row;
            if row.len() != n_cols {
                return Err(CoreError::InvalidInput(
                    crate::error::ErrorContext::new(&format!(
                        "IntervalMatrix::from_f64_rows: row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n_cols
                    )),
                ));
            }
        }
        let data: Vec<Interval<f64>> = rows_data
            .iter()
            .flat_map(|row: &Vec<f64>| row.iter().map(|&v| Interval::point(v)))
            .collect();
        Ok(Self {
            rows: n_rows,
            cols: n_cols,
            data,
        })
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Access element at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&Interval<f64>> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    /// Mutable access to element at `(row, col)`.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Interval<f64>> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            self.data.get_mut(row * self.cols + col)
        }
    }

    /// Set element at `(row, col)`.
    pub fn set(&mut self, row: usize, col: usize, val: Interval<f64>) -> CoreResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new("IntervalMatrix::set: index out of bounds"),
            ));
        }
        self.data[row * self.cols + col] = val;
        Ok(())
    }

    /// Get a row as an `IntervalVector`.
    pub fn row(&self, r: usize) -> CoreResult<IntervalVector> {
        if r >= self.rows {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new("IntervalMatrix::row: index out of bounds"),
            ));
        }
        let start = r * self.cols;
        Ok(IntervalVector::new(self.data[start..start + self.cols].to_vec()))
    }

    /// Matrix-vector multiply `self * v` using interval arithmetic.
    ///
    /// Returns `Err` if dimensions are incompatible.
    pub fn mul_vec(&self, v: &IntervalVector) -> CoreResult<IntervalVector> {
        if self.cols != v.len() {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new(&format!(
                    "IntervalMatrix::mul_vec: matrix cols {} != vector len {}",
                    self.cols,
                    v.len()
                )),
            ));
        }
        let mut result = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let mut acc = Interval::point(0.0_f64);
            for c in 0..self.cols {
                let a = self.data[r * self.cols + c];
                let b = *v.get(c).expect("index within bounds");
                acc = acc + a * b;
            }
            result.push(acc);
        }
        Ok(IntervalVector::new(result))
    }

    /// Matrix-matrix multiply `self * rhs` using interval arithmetic.
    pub fn mul_mat(&self, rhs: &Self) -> CoreResult<Self> {
        if self.cols != rhs.rows {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new(&format!(
                    "IntervalMatrix::mul_mat: self.cols {} != rhs.rows {}",
                    self.cols, rhs.rows
                )),
            ));
        }
        let n = self.rows;
        let m = rhs.cols;
        let k = self.cols;
        let mut out = vec![Interval::point(0.0_f64); n * m];
        for i in 0..n {
            for j in 0..m {
                let mut acc = Interval::point(0.0_f64);
                for l in 0..k {
                    acc = acc + self.data[i * k + l] * rhs.data[l * m + j];
                }
                out[i * m + j] = acc;
            }
        }
        Self::from_flat(n, m, out)
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        let mut data = vec![Interval::point(0.0_f64); self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// Gaussian elimination with interval arithmetic
// ---------------------------------------------------------------------------

/// Solve the linear system `A x = b` using interval Gaussian elimination.
///
/// Returns an `IntervalVector` `x` such that the true solution (assuming
/// the system is non-singular and the inputs correctly enclose their true
/// values) is guaranteed to be contained in the returned intervals.
///
/// # Algorithm
///
/// This is a direct implementation of interval Gaussian elimination with
/// partial pivoting:
///
/// 1. Form the augmented matrix `[A | b]`.
/// 2. For each pivot column: choose the row with maximum `mag(pivot interval)`
///    to minimise interval blow-up.
/// 3. Perform forward elimination using interval arithmetic.
/// 4. Perform back substitution using interval arithmetic.
///
/// # Errors
///
/// * `CoreError::InvalidInput` — if `A` is not square or dimensions are
///   inconsistent.
/// * `CoreError::Computation` — if a zero (or near-zero) pivot is encountered
///   (the system may be singular or poorly conditioned).
///
/// # Note on verified enclosures
///
/// The returned intervals are *rigorous* in the sense that every arithmetic
/// step uses outward-rounded interval arithmetic.  However, for ill-conditioned
/// systems the intervals may be very wide.  For a tighter enclosure consider
/// combining this with an iterative refinement step.
pub fn gaussian_elimination_interval(
    a: &IntervalMatrix,
    b: &IntervalVector,
) -> CoreResult<IntervalVector> {
    let n = a.rows();
    if a.cols() != n {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("gaussian_elimination_interval: A must be square"),
        ));
    }
    if b.len() != n {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new(
                "gaussian_elimination_interval: b length must equal number of rows",
            ),
        ));
    }
    if n == 0 {
        return Ok(IntervalVector::new(Vec::new()));
    }

    // Build augmented matrix [A | b] with n rows, (n+1) columns
    let aug_cols = n + 1;
    let mut aug: Vec<Interval<f64>> = Vec::with_capacity(n * aug_cols);
    for r in 0..n {
        for c in 0..n {
            aug.push(
                *a.get(r, c)
                    .ok_or_else(|| CoreError::InvalidInput(
                        crate::error::ErrorContext::new("gaussian_elimination_interval: index error"),
                    ))?,
            );
        }
        aug.push(
            *b.get(r)
                .ok_or_else(|| CoreError::InvalidInput(
                    crate::error::ErrorContext::new("gaussian_elimination_interval: b index error"),
                ))?,
        );
    }

    // Helper closures operating on the flat augmented matrix
    let idx = |r: usize, c: usize| -> usize { r * aug_cols + c };

    // Forward elimination
    for col in 0..n {
        // Partial pivoting: find row with maximum mag in pivot column
        let pivot_row = {
            let mut best = col;
            let mut best_mag = aug[idx(col, col)].mag();
            for r in (col + 1)..n {
                let m = aug[idx(r, col)].mag();
                if m > best_mag {
                    best_mag = m;
                    best = r;
                }
            }
            best
        };

        // Check that the pivot is non-zero
        let pivot = aug[idx(pivot_row, col)];
        if pivot.mig() == 0.0 && pivot.mag() == 0.0 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(&format!(
                    "gaussian_elimination_interval: zero pivot at column {}",
                    col
                )),
            ));
        }

        // Swap rows col and pivot_row
        if pivot_row != col {
            for c in 0..aug_cols {
                aug.swap(idx(col, c), idx(pivot_row, c));
            }
        }

        let pivot = aug[idx(col, col)];

        // Eliminate entries below pivot
        for r in (col + 1)..n {
            let factor = aug[idx(r, col)] / pivot;
            aug[idx(r, col)] = Interval::point(0.0_f64); // exact zero
            for c in (col + 1)..aug_cols {
                let rhs_val = aug[idx(col, c)];
                let current = aug[idx(r, c)];
                aug[idx(r, c)] = current - factor * rhs_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![Interval::point(0.0_f64); n];
    for i in (0..n).rev() {
        let mut rhs = aug[idx(i, n)]; // right-hand side
        for j in (i + 1)..n {
            rhs = rhs - aug[idx(i, j)] * x[j];
        }
        let pivot = aug[idx(i, i)];
        if pivot.mig() == 0.0 && pivot.mag() == 0.0 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(&format!(
                    "gaussian_elimination_interval: zero pivot during back-substitution at row {}",
                    i
                )),
            ));
        }
        x[i] = rhs / pivot;
    }

    Ok(IntervalVector::new(x))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(lo: f64, hi: f64) -> Interval<f64> {
        Interval::new(lo, hi)
    }

    fn ip(x: f64) -> Interval<f64> {
        Interval::point(x)
    }

    #[test]
    fn test_matvec() {
        // [[1,0],[0,1]] * [2,3] = [2,3]
        let a = IntervalMatrix::from_f64_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]])
            .expect("identity matrix");
        let v = IntervalVector::from_point_slice(&[2.0, 3.0]);
        let r = a.mul_vec(&v).expect("mul_vec");
        assert!(r.get(0).expect("get").contains(2.0));
        assert!(r.get(1).expect("get").contains(3.0));
    }

    #[test]
    fn test_gaussian_identity() {
        // I * x = b => x = b
        let a = IntervalMatrix::from_f64_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]])
            .expect("identity");
        let b = IntervalVector::from_point_slice(&[3.0, 7.0]);
        let x = gaussian_elimination_interval(&a, &b).expect("solve");
        assert!(
            x.get(0).expect("get").contains(3.0),
            "x[0] = {:?}",
            x.get(0)
        );
        assert!(
            x.get(1).expect("get").contains(7.0),
            "x[1] = {:?}",
            x.get(1)
        );
    }

    #[test]
    fn test_gaussian_2x2() {
        // 2x + y = 5
        // x + 3y = 10
        // Solution: x=1, y=3
        let a = IntervalMatrix::from_f64_rows(&[vec![2.0, 1.0], vec![1.0, 3.0]])
            .expect("matrix");
        let b = IntervalVector::from_point_slice(&[5.0, 10.0]);
        let x = gaussian_elimination_interval(&a, &b).expect("solve");
        assert!(
            x.get(0).expect("x[0]").contains(1.0),
            "x[0] should contain 1.0, got {:?}",
            x.get(0)
        );
        assert!(
            x.get(1).expect("x[1]").contains(3.0),
            "x[1] should contain 3.0, got {:?}",
            x.get(1)
        );
    }

    #[test]
    fn test_gaussian_with_intervals() {
        // Interval coefficients: a ∈ [1.9, 2.1], b ∈ [0.9, 1.1]
        // RHS: [4.9, 5.1]
        // True point solution (at midpoints) matches the point system
        let a = IntervalMatrix::from_flat(
            2,
            2,
            vec![
                iv(1.9, 2.1),
                iv(0.9, 1.1),
                iv(0.9, 1.1),
                iv(2.9, 3.1),
            ],
        )
        .expect("matrix");
        let b = IntervalVector::new(vec![iv(4.9, 5.1), iv(9.9, 10.1)]);
        let x = gaussian_elimination_interval(&a, &b).expect("solve");
        // The solution must contain x≈1, y≈3 with some widening from the intervals
        assert!(
            x.get(0).expect("x[0]").contains(1.0),
            "x[0] = {:?}",
            x.get(0)
        );
        assert!(
            x.get(1).expect("x[1]").contains(3.0),
            "x[1] = {:?}",
            x.get(1)
        );
    }

    #[test]
    fn test_dot_product() {
        let u = IntervalVector::from_point_slice(&[1.0, 2.0, 3.0]);
        let v = IntervalVector::from_point_slice(&[4.0, 5.0, 6.0]);
        let d = u.dot(&v).expect("dot");
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!(d.contains(32.0), "dot = {:?}", d);
    }

    #[test]
    fn test_matrix_multiply() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = IntervalMatrix::from_f64_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]])
            .expect("a");
        let b = IntervalMatrix::from_f64_rows(&[vec![5.0, 6.0], vec![7.0, 8.0]])
            .expect("b");
        let c = a.mul_mat(&b).expect("mul_mat");
        assert!(c.get(0, 0).expect("00").contains(19.0));
        assert!(c.get(0, 1).expect("01").contains(22.0));
        assert!(c.get(1, 0).expect("10").contains(43.0));
        assert!(c.get(1, 1).expect("11").contains(50.0));
    }
}
