//! Flat (`Vec<f64>`) structured matrix types: Toeplitz, Circulant, Hankel.
//!
//! These types use plain `Vec<f64>` storage and are distinct from the
//! generic ndarray-based types in `crate::structured`.

use crate::error::{LinalgError, LinalgResult};

// ─────────────────────────────────────────────────────────────────────────────
// Toeplitz
// ─────────────────────────────────────────────────────────────────────────────

/// Toeplitz matrix T where `T[i,j] = t[i-j]` for an n×n matrix.
///
/// Represented by its first row `T[0,:]` and first column `T[:,0]`.
/// The (0,0) element must agree in both vectors.
#[derive(Debug, Clone)]
pub struct FlatToeplitz {
    /// First row: T\[0,0\], T\[0,1\], …, T\[0,n-1\].
    pub first_row: Vec<f64>,
    /// First column: T\[0,0\], T\[1,0\], …, T\[n-1,0\].
    pub first_col: Vec<f64>,
    /// Matrix dimension n.
    pub n: usize,
}

impl FlatToeplitz {
    /// Construct a new FlatToeplitz matrix.
    ///
    /// # Errors
    /// Returns an error if `first_row` and `first_col` have different lengths or the
    /// (0,0) elements disagree.
    pub fn new(first_row: Vec<f64>, first_col: Vec<f64>) -> LinalgResult<Self> {
        let n = first_row.len();
        if first_col.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "first_row length {} ≠ first_col length {}",
                n,
                first_col.len()
            )));
        }
        if n == 0 {
            return Err(LinalgError::DimensionError("n must be > 0".into()));
        }
        if (first_row[0] - first_col[0]).abs() > 1e-14 * first_row[0].abs().max(1.0) {
            return Err(LinalgError::ValueError(
                "first_row[0] and first_col[0] must be equal".into(),
            ));
        }
        Ok(Self {
            first_row,
            first_col,
            n,
        })
    }

    /// Return element T[i, j].
    pub fn get(&self, i: usize, j: usize) -> LinalgResult<f64> {
        if i >= self.n || j >= self.n {
            return Err(LinalgError::IndexError(format!(
                "({},{}) out of bounds for {}×{}",
                i, j, self.n, self.n
            )));
        }
        Ok(if i <= j {
            self.first_row[j - i]
        } else {
            self.first_col[i - j]
        })
    }

    /// Convert to dense row-major vector of length n*n.
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.n;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i <= j {
                    self.first_row[j - i]
                } else {
                    self.first_col[i - j]
                };
            }
        }
        a
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Circulant
// ─────────────────────────────────────────────────────────────────────────────

/// Circulant matrix C where `C[i,j] = c[(i - j + n) % n]`.
///
/// Equivalently, row `i` is a cyclic right-shift of row `i-1` by one position.
/// This is the _standard_ (cyclic-convolution) convention, matching the
/// FFT-based matvec `y = IDFT(DFT(c) ⊙ DFT(x))`.
///
/// Fully determined by its first column (= `first_row` field here).
#[derive(Debug, Clone)]
pub struct FlatCirculant {
    /// First column (= first row for this convention): `c[0], c[1], …, c[n-1]`.
    pub first_row: Vec<f64>,
    /// Matrix dimension n.
    pub n: usize,
}

impl FlatCirculant {
    /// Construct a new FlatCirculant from the first row.
    ///
    /// # Errors
    /// Returns an error if `first_row` is empty.
    pub fn new(first_row: Vec<f64>) -> LinalgResult<Self> {
        let n = first_row.len();
        if n == 0 {
            return Err(LinalgError::DimensionError("n must be > 0".into()));
        }
        Ok(Self { first_row, n })
    }

    /// Return element C[i, j] = c[(i - j + n) % n].
    pub fn get(&self, i: usize, j: usize) -> LinalgResult<f64> {
        if i >= self.n || j >= self.n {
            return Err(LinalgError::IndexError(format!(
                "({},{}) out of bounds for {}×{}",
                i, j, self.n, self.n
            )));
        }
        let idx = (i + self.n - j) % self.n;
        Ok(self.first_row[idx])
    }

    /// Convert to dense row-major vector of length n*n.
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.n;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let idx = (i + n - j) % n;
                a[i * n + j] = self.first_row[idx];
            }
        }
        a
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hankel
// ─────────────────────────────────────────────────────────────────────────────

/// Hankel matrix H where `H[i,j] = h[i+j]`.
///
/// Represented by the first row `H[0,:]` and last column `H[:,n-1]`.
/// The element `H[0,n-1]` must agree: `first_row[n-1] == last_col[0]`.
#[derive(Debug, Clone)]
pub struct FlatHankel {
    /// First row: H\[0,0\], H\[0,1\], …, H\[0,n-1\].
    pub first_row: Vec<f64>,
    /// Last column: H\[0,n-1\], H\[1,n-1\], …, H\[n-1,n-1\].
    pub last_col: Vec<f64>,
    /// Matrix dimension n.
    pub n: usize,
}

impl FlatHankel {
    /// Construct a new FlatHankel matrix.
    ///
    /// # Errors
    /// Returns an error if lengths mismatch or the shared corner element disagrees.
    pub fn new(first_row: Vec<f64>, last_col: Vec<f64>) -> LinalgResult<Self> {
        let n = first_row.len();
        if last_col.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "first_row length {} ≠ last_col length {}",
                n,
                last_col.len()
            )));
        }
        if n == 0 {
            return Err(LinalgError::DimensionError("n must be > 0".into()));
        }
        // H[0, n-1] = first_row[n-1] = last_col[0].
        if (first_row[n - 1] - last_col[0]).abs() > 1e-14 * first_row[n - 1].abs().max(1.0) {
            return Err(LinalgError::ValueError(
                "first_row[n-1] and last_col[0] must be equal (corner element)".into(),
            ));
        }
        Ok(Self {
            first_row,
            last_col,
            n,
        })
    }

    /// Return element H[i, j].
    pub fn get(&self, i: usize, j: usize) -> LinalgResult<f64> {
        if i >= self.n || j >= self.n {
            return Err(LinalgError::IndexError(format!(
                "({},{}) out of bounds for {}×{}",
                i, j, self.n, self.n
            )));
        }
        let k = i + j;
        Ok(if k < self.n {
            self.first_row[k]
        } else {
            self.last_col[k - (self.n - 1)]
        })
    }

    /// Convert to dense row-major vector of length n*n.
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.n;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let k = i + j;
                a[i * n + j] = if k < n {
                    self.first_row[k]
                } else {
                    self.last_col[k - (n - 1)]
                };
            }
        }
        a
    }
}
