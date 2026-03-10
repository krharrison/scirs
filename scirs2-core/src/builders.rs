//! # Ergonomic Builder Patterns for Array Construction
//!
//! This module provides fluent builder patterns that make common array-construction
//! tasks more discoverable and IDE-friendly. Instead of remembering multiple
//! constructors scattered across ndarray, users can use a single entry-point and
//! let IDE autocomplete guide them.
//!
//! ## Design Goals
//!
//! - **Discoverability**: All construction paths live under `MatrixBuilder`,
//!   `VectorBuilder`, and `ArrayBuilder` — easy to find in IDEs.
//! - **No unwrap**: Every fallible operation returns `CoreResult`.
//! - **Generic**: Works for any numeric type satisfying the appropriate traits.
//! - **Zero-cost**: The builders are thin wrappers; all cost is in the actual
//!   array allocation, matching what you would write by hand.
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_core::builders::{MatrixBuilder, VectorBuilder, ArrayBuilder};
//!
//! // 2D Matrix construction
//! let eye3 = MatrixBuilder::<f64>::eye(3);
//! let zeros = MatrixBuilder::<f64>::zeros(4, 4);
//! let from_data = MatrixBuilder::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2)
//!     .expect("correct element count");
//! let from_fn = MatrixBuilder::from_fn(3, 3, |r, c| if r == c { 1.0f64 } else { 0.0 });
//!
//! // 1D Vector construction
//! let linspace = VectorBuilder::<f64>::linspace(0.0, 1.0, 11);
//! let arange = VectorBuilder::<f64>::arange(0.0, 5.0, 1.0);
//! let logspace = VectorBuilder::<f64>::logspace(0.0, 3.0, 4);
//! let from_vec = VectorBuilder::from_vec(vec![1.0, 2.0, 3.0]);
//!
//! // Generic multi-dim array
//! let shaped = ArrayBuilder::<f64, _>::zeros(ndarray::Ix2(3, 4));
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use ::ndarray::{Array1, Array2, ArrayD, Dimension, IntoDimension, IxDyn, ShapeError};
use num_traits::{Float, One, Zero};
use std::fmt::Display;
use std::ops::MulAssign;

// ============================================================================
// MatrixBuilder — 2D Matrix Construction
// ============================================================================

/// Fluent builder for two-dimensional matrices.
///
/// All methods are associated functions (no `new()` required), making them
/// trivially discoverable via IDE autocomplete when typing `MatrixBuilder::`.
///
/// # Type Parameter
///
/// `T` must be numeric. Common choices: `f64`, `f32`, `i32`, `i64`, `u64`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::builders::MatrixBuilder;
///
/// // Identity matrix
/// let eye = MatrixBuilder::<f64>::eye(3);
/// assert_eq!(eye[[0, 0]], 1.0);
/// assert_eq!(eye[[0, 1]], 0.0);
///
/// // Zeros / ones
/// let z = MatrixBuilder::<f64>::zeros(2, 3);
/// let o = MatrixBuilder::<f64>::ones(2, 3);
///
/// // From closure — computed element-by-element
/// let computed = MatrixBuilder::from_fn(3, 3, |r, c| (r * 3 + c) as f64);
/// assert_eq!(computed[[1, 2]], 5.0);
/// ```
pub struct MatrixBuilder<T>(std::marker::PhantomData<T>);

impl<T> MatrixBuilder<T>
where
    T: Clone + Zero,
{
    /// Create a matrix of all zeros with shape `(rows, cols)`.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::<f64>::zeros(3, 4);
    /// assert_eq!(m.shape(), &[3, 4]);
    /// assert_eq!(m[[0, 0]], 0.0);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Array2<T> {
        Array2::<T>::zeros((rows, cols))
    }

    /// Build a matrix from a flat `Vec` of elements in row-major order.
    ///
    /// Returns an error if the number of elements does not match `rows * cols`.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], 2, 2)
    ///     .expect("element count matches");
    /// assert_eq!(m[[0, 0]], 1.0);
    /// assert_eq!(m[[1, 1]], 4.0);
    /// ```
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> CoreResult<Array2<T>> {
        if data.len() != rows * cols {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "MatrixBuilder::from_vec: expected {} elements for a {}×{} matrix, got {}",
                rows * cols,
                rows,
                cols,
                data.len()
            ))));
        }
        Array2::from_shape_vec((rows, cols), data).map_err(|e: ShapeError| {
            CoreError::InvalidInput(ErrorContext::new(format!(
                "MatrixBuilder::from_vec shape error: {e}"
            )))
        })
    }
}

impl<T> MatrixBuilder<T>
where
    T: Clone + Zero + One,
{
    /// Create a square identity matrix of size `n × n`.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let eye = MatrixBuilder::<f64>::eye(3);
    /// assert_eq!(eye[[2, 2]], 1.0);
    /// assert_eq!(eye[[0, 1]], 0.0);
    /// ```
    pub fn eye(n: usize) -> Array2<T> {
        let mut m = Array2::<T>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] = T::one();
        }
        m
    }

    /// Create a matrix of all ones with shape `(rows, cols)`.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::<f64>::ones(2, 3);
    /// assert_eq!(m[[1, 2]], 1.0);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Array2<T> {
        Array2::<T>::from_elem((rows, cols), T::one())
    }
}

impl<T> MatrixBuilder<T>
where
    T: Clone,
{
    /// Create a matrix filled with a single constant value.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::full(3, 3, 7_i32);
    /// assert_eq!(m[[0, 0]], 7);
    /// assert_eq!(m[[2, 2]], 7);
    /// ```
    pub fn full(rows: usize, cols: usize, value: T) -> Array2<T> {
        Array2::from_elem((rows, cols), value)
    }

    /// Create a matrix where each element is produced by calling `f(row, col)`.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::from_fn(3, 3, |r, c| (r * 3 + c) as f64);
    /// assert_eq!(m[[0, 0]], 0.0);
    /// assert_eq!(m[[2, 2]], 8.0);
    /// ```
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Array2<T>
    where
        F: FnMut(usize, usize) -> T,
    {
        Array2::from_shape_fn((rows, cols), |(r, c)| f(r, c))
    }
}

impl<T> MatrixBuilder<T>
where
    T: Float + Clone,
{
    /// Create a matrix populated with uniform random values in `[0, 1)` using a seeded
    /// ChaCha8 RNG for reproducibility.
    ///
    /// The `seed` parameter lets callers produce deterministic results in tests
    /// and benchmarks while still getting varied values in production by passing
    /// different seeds.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::<f64>::rand(3, 3, 42);
    /// assert_eq!(m.shape(), &[3, 3]);
    /// // All values should be in [0, 1)
    /// assert!(m.iter().all(|&v| v >= 0.0 && v < 1.0));
    /// ```
    pub fn rand(rows: usize, cols: usize, seed: u64) -> Array2<T> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((rows, cols), |_| {
            // Generate a uniform f64 in [0, 1) and cast to T
            use rand::Rng;
            let v: f64 = rng.random();
            T::from(v).unwrap_or_else(T::zero)
        })
    }

    /// Create a matrix populated with standard normal (`N(0, 1)`) random values.
    ///
    /// ```rust
    /// use scirs2_core::builders::MatrixBuilder;
    ///
    /// let m = MatrixBuilder::<f64>::randn(4, 4, 0);
    /// assert_eq!(m.shape(), &[4, 4]);
    /// ```
    pub fn randn(rows: usize, cols: usize, seed: u64) -> Array2<T> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((rows, cols), |_| {
            let v: f64 = StandardNormal.sample(&mut rng);
            T::from(v).unwrap_or_else(T::zero)
        })
    }
}

// ============================================================================
// VectorBuilder — 1D Array Construction
// ============================================================================

/// Fluent builder for one-dimensional arrays (vectors).
///
/// Provides NumPy-like constructors (`linspace`, `arange`, `logspace`) as well
/// as the standard `zeros`, `ones`, `from_vec`, and `from_fn` constructors.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::builders::VectorBuilder;
///
/// let v = VectorBuilder::<f64>::linspace(0.0, 1.0, 5);
/// assert!((v[0] - 0.0).abs() < 1e-12);
/// assert!((v[4] - 1.0).abs() < 1e-12);
///
/// let r = VectorBuilder::<f64>::arange(0.0, 5.0, 1.0);
/// assert_eq!(r.len(), 5);
/// ```
pub struct VectorBuilder<T>(std::marker::PhantomData<T>);

impl<T> VectorBuilder<T>
where
    T: Clone + Zero,
{
    /// Create a vector of all zeros with `n` elements.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::<f64>::zeros(5);
    /// assert_eq!(v.len(), 5);
    /// assert_eq!(v[3], 0.0);
    /// ```
    pub fn zeros(n: usize) -> Array1<T> {
        Array1::<T>::zeros(n)
    }

    /// Build a vector from a `Vec`.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::from_vec(vec![1.0_f64, 2.0, 3.0]);
    /// assert_eq!(v[1], 2.0);
    /// ```
    pub fn from_vec(data: Vec<T>) -> Array1<T> {
        Array1::from(data)
    }
}

impl<T> VectorBuilder<T>
where
    T: Clone + Zero + One,
{
    /// Create a vector of all ones with `n` elements.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::<f64>::ones(4);
    /// assert_eq!(v[2], 1.0);
    /// ```
    pub fn ones(n: usize) -> Array1<T> {
        Array1::from_elem(n, T::one())
    }
}

impl<T> VectorBuilder<T>
where
    T: Clone,
{
    /// Create a vector where element `i` is produced by `f(i)`.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let squares = VectorBuilder::from_fn(5, |i| (i * i) as f64);
    /// assert_eq!(squares[3], 9.0);
    /// ```
    pub fn from_fn<F>(n: usize, mut f: F) -> Array1<T>
    where
        F: FnMut(usize) -> T,
    {
        Array1::from_shape_fn(n, |i| f(i))
    }

    /// Create a vector filled with a constant value.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::full(3, 7_i32);
    /// assert_eq!(v[0], 7);
    /// ```
    pub fn full(n: usize, value: T) -> Array1<T> {
        Array1::from_elem(n, value)
    }
}

impl<T> VectorBuilder<T>
where
    T: Float + Display + Clone + MulAssign,
{
    /// Create `n` evenly spaced values from `start` to `stop` (inclusive).
    ///
    /// This is the analogue of NumPy's `np.linspace`.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::<f64>::linspace(0.0, 4.0, 5);
    /// assert!((v[0] - 0.0).abs() < 1e-12);
    /// assert!((v[2] - 2.0).abs() < 1e-12);
    /// assert!((v[4] - 4.0).abs() < 1e-12);
    /// ```
    pub fn linspace(start: T, stop: T, n: usize) -> Array1<T> {
        if n == 0 {
            return Array1::from(vec![]);
        }
        if n == 1 {
            return Array1::from(vec![start]);
        }
        let steps = T::from(n - 1).unwrap_or_else(T::one);
        Array1::from_shape_fn(n, |i| {
            let t = T::from(i).unwrap_or_else(T::zero);
            start + (stop - start) * (t / steps)
        })
    }

    /// Create values from `start` up to (but not including) `stop` with step `step`.
    ///
    /// This is the analogue of NumPy's `np.arange`.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::<f64>::arange(0.0, 5.0, 1.0);
    /// assert_eq!(v.len(), 5);
    /// assert!((v[0] - 0.0).abs() < 1e-12);
    /// assert!((v[4] - 4.0).abs() < 1e-12);
    ///
    /// // Fractional step
    /// let v2 = VectorBuilder::<f64>::arange(0.0, 1.0, 0.5);
    /// assert_eq!(v2.len(), 2);
    /// ```
    pub fn arange(start: T, stop: T, step: T) -> Array1<T> {
        if step == T::zero() || (stop - start).signum() != step.signum() {
            return Array1::from(vec![]);
        }
        let n_float = ((stop - start) / step).ceil();
        let n = n_float.to_usize().unwrap_or(0).max(0);
        Array1::from_shape_fn(n, |i| start + step * T::from(i).unwrap_or_else(T::zero))
    }

    /// Create `n` values evenly spaced on a logarithmic scale.
    ///
    /// The values span from `10^start` to `10^stop` (inclusive), analogous to
    /// NumPy's `np.logspace(start, stop, n, base=10)`.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// // 4 values from 10^0 = 1 to 10^3 = 1000
    /// let v = VectorBuilder::<f64>::logspace(0.0, 3.0, 4);
    /// assert!((v[0] - 1.0).abs() < 1e-10);
    /// assert!((v[3] - 1000.0).abs() < 1e-8);
    /// ```
    pub fn logspace(start: T, stop: T, n: usize) -> Array1<T> {
        let lin = Self::linspace(start, stop, n);
        lin.mapv(|x| T::from(10.0_f64).unwrap_or_else(T::one).powf(x))
    }

    /// Create `n` uniform random values in `[0, 1)` using a seeded ChaCha8 RNG.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::<f64>::rand(5, 42);
    /// assert_eq!(v.len(), 5);
    /// assert!(v.iter().all(|&x| x >= 0.0 && x < 1.0));
    /// ```
    pub fn rand(n: usize, seed: u64) -> Array1<T> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array1::from_shape_fn(n, |_| {
            use rand::Rng;
            let v: f64 = rng.random();
            T::from(v).unwrap_or_else(T::zero)
        })
    }

    /// Create `n` standard-normal random values using a seeded ChaCha8 RNG.
    ///
    /// ```rust
    /// use scirs2_core::builders::VectorBuilder;
    ///
    /// let v = VectorBuilder::<f64>::randn(5, 0);
    /// assert_eq!(v.len(), 5);
    /// ```
    pub fn randn(n: usize, seed: u64) -> Array1<T> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array1::from_shape_fn(n, |_| {
            let v: f64 = StandardNormal.sample(&mut rng);
            T::from(v).unwrap_or_else(T::zero)
        })
    }
}

// ============================================================================
// ArrayBuilder — Generic N-dimensional Array Construction
// ============================================================================

/// Generic builder for N-dimensional arrays.
///
/// Where `MatrixBuilder` targets exactly 2D and `VectorBuilder` targets exactly 1D,
/// `ArrayBuilder` works with any [`ndarray::Dimension`] and is useful when the
/// shape is determined at runtime.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::builders::ArrayBuilder;
///
/// let a2 = ArrayBuilder::<f64, _>::zeros(ndarray::Ix2(3, 4));
/// assert_eq!(a2.shape(), &[3, 4]);
///
/// let a3 = ArrayBuilder::<f64, _>::zeros(ndarray::Ix3(2, 3, 4));
/// assert_eq!(a3.shape(), &[2, 3, 4]);
///
/// // Dynamic dimension
/// let ad = ArrayBuilder::<f64, ndarray::IxDyn>::zeros_dyn(&[2, 3, 4]);
/// assert_eq!(ad.shape(), &[2, 3, 4]);
/// ```
pub struct ArrayBuilder<T, D>(std::marker::PhantomData<(T, D)>);

impl<T, D> ArrayBuilder<T, D>
where
    T: Clone + Zero,
    D: Dimension,
{
    /// Create a zeros array with the given shape.
    ///
    /// ```rust
    /// use scirs2_core::builders::ArrayBuilder;
    ///
    /// let a = ArrayBuilder::<f64, _>::zeros(ndarray::Ix2(3, 4));
    /// assert_eq!(a.shape(), &[3, 4]);
    /// ```
    pub fn zeros<Sh>(shape: Sh) -> ::ndarray::Array<T, D>
    where
        Sh: IntoDimension<Dim = D>,
    {
        ::ndarray::Array::zeros(shape)
    }

    /// Create an array filled with a constant value.
    ///
    /// ```rust
    /// use scirs2_core::builders::ArrayBuilder;
    ///
    /// let a = ArrayBuilder::<i32, _>::full(ndarray::Ix2(2, 3), 7);
    /// assert_eq!(a[[0, 0]], 7);
    /// ```
    pub fn full<Sh>(shape: Sh, value: T) -> ::ndarray::Array<T, D>
    where
        Sh: IntoDimension<Dim = D>,
    {
        ::ndarray::Array::from_elem(shape, value)
    }

    /// Create an array where each element is produced by a closure receiving the
    /// dimension pattern (e.g. `(row, col)` for 2D, `(i, j, k)` for 3D, etc.).
    ///
    /// ```rust
    /// use scirs2_core::builders::ArrayBuilder;
    ///
    /// // 3×3 matrix: element = row + col
    /// let a = ArrayBuilder::<usize, ndarray::Ix2>::from_fn(
    ///     ndarray::Ix2(3, 3),
    ///     |(r, c)| r + c,
    /// );
    /// assert_eq!(a[[2, 2]], 4);
    /// ```
    pub fn from_fn<Sh, F>(shape: Sh, f: F) -> ::ndarray::Array<T, D>
    where
        Sh: IntoDimension<Dim = D>,
        F: FnMut(D::Pattern) -> T,
    {
        ::ndarray::Array::from_shape_fn(shape, f)
    }

    /// Build an array from a flat `Vec` of elements in C-order (row-major).
    ///
    /// Returns a `CoreError` if the element count does not match the given shape.
    ///
    /// ```rust
    /// use scirs2_core::builders::ArrayBuilder;
    ///
    /// let a = ArrayBuilder::<f64, ndarray::Ix2>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     ndarray::Ix2(2, 3),
    /// ).expect("element count matches");
    /// assert_eq!(a[[1, 2]], 6.0);
    /// ```
    pub fn from_vec<Sh>(data: Vec<T>, shape: Sh) -> CoreResult<::ndarray::Array<T, D>>
    where
        Sh: IntoDimension<Dim = D>,
    {
        ::ndarray::Array::from_shape_vec(shape, data).map_err(|e: ShapeError| {
            CoreError::InvalidInput(ErrorContext::new(format!(
                "ArrayBuilder::from_vec shape error: {e}"
            )))
        })
    }
}

impl<T> ArrayBuilder<T, IxDyn>
where
    T: Clone + Zero,
{
    /// Create a dynamic-dimensional zeros array from a runtime shape slice.
    ///
    /// ```rust
    /// use scirs2_core::builders::ArrayBuilder;
    ///
    /// let a = ArrayBuilder::<f64, ndarray::IxDyn>::zeros_dyn(&[2, 3, 4]);
    /// assert_eq!(a.ndim(), 3);
    /// assert_eq!(a.shape(), &[2, 3, 4]);
    /// ```
    pub fn zeros_dyn(shape: &[usize]) -> ArrayD<T> {
        ArrayD::zeros(IxDyn(shape))
    }

    /// Create a dynamic-dimensional array filled with `value`.
    pub fn full_dyn(shape: &[usize], value: T) -> ArrayD<T> {
        ArrayD::from_elem(IxDyn(shape), value)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- MatrixBuilder tests ---

    #[test]
    fn test_matrix_zeros() {
        let m = MatrixBuilder::<f64>::zeros(3, 4);
        assert_eq!(m.shape(), &[3, 4]);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_matrix_ones() {
        let m = MatrixBuilder::<f64>::ones(2, 5);
        assert_eq!(m.shape(), &[2, 5]);
        assert!(m.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_matrix_eye() {
        let eye = MatrixBuilder::<f64>::eye(3);
        assert_eq!(eye.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(eye[[i, j]], expected);
            }
        }
    }

    #[test]
    fn test_matrix_from_vec() {
        let m = MatrixBuilder::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], 2, 2)
            .expect("element count should match");
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 2.0);
        assert_eq!(m[[1, 0]], 3.0);
        assert_eq!(m[[1, 1]], 4.0);
    }

    #[test]
    fn test_matrix_from_vec_error() {
        // Wrong element count → error
        let result = MatrixBuilder::<f64>::from_vec(vec![1.0, 2.0, 3.0], 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_from_fn() {
        let m = MatrixBuilder::from_fn(3, 3, |r, c| (r * 3 + c) as f64);
        for r in 0..3 {
            for c in 0..3 {
                assert_abs_diff_eq!(m[[r, c]], (r * 3 + c) as f64);
            }
        }
    }

    #[test]
    fn test_matrix_full() {
        let m = MatrixBuilder::full(3, 3, 42_i32);
        assert!(m.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_matrix_rand() {
        let m = MatrixBuilder::<f64>::rand(10, 10, 99);
        assert_eq!(m.shape(), &[10, 10]);
        assert!(m.iter().all(|&v| v >= 0.0 && v < 1.0));
        // Deterministic: same seed → same values
        let m2 = MatrixBuilder::<f64>::rand(10, 10, 99);
        assert_eq!(m, m2);
    }

    #[test]
    fn test_matrix_randn() {
        let m = MatrixBuilder::<f64>::randn(100, 100, 0);
        // Mean should be roughly 0, std roughly 1
        let mean = m.mean().expect("non-empty");
        assert!(mean.abs() < 0.5, "mean={mean}");
    }

    // --- VectorBuilder tests ---

    #[test]
    fn test_vector_zeros() {
        let v = VectorBuilder::<f64>::zeros(5);
        assert_eq!(v.len(), 5);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vector_ones() {
        let v = VectorBuilder::<f64>::ones(4);
        assert_eq!(v.len(), 4);
        assert!(v.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_vector_from_vec() {
        let v = VectorBuilder::from_vec(vec![10.0_f64, 20.0, 30.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v[1], 20.0);
    }

    #[test]
    fn test_vector_from_fn() {
        let v = VectorBuilder::from_fn(5, |i| i as f64 * 2.0);
        assert_abs_diff_eq!(v[3], 6.0);
    }

    #[test]
    fn test_vector_full() {
        let v = VectorBuilder::full(4, 1.23_f64);
        assert!(v.iter().all(|&x| (x - 1.23).abs() < 1e-12));
    }

    #[test]
    fn test_vector_linspace() {
        let v = VectorBuilder::<f64>::linspace(0.0, 4.0, 5);
        assert_eq!(v.len(), 5);
        for (i, &val) in v.iter().enumerate() {
            assert_abs_diff_eq!(val, i as f64, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_vector_linspace_single() {
        let v = VectorBuilder::<f64>::linspace(3.0, 3.0, 1);
        assert_eq!(v.len(), 1);
        assert_abs_diff_eq!(v[0], 3.0);
    }

    #[test]
    fn test_vector_linspace_empty() {
        let v = VectorBuilder::<f64>::linspace(0.0, 1.0, 0);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_vector_arange() {
        let v = VectorBuilder::<f64>::arange(0.0, 5.0, 1.0);
        assert_eq!(v.len(), 5);
        for (i, &val) in v.iter().enumerate() {
            assert_abs_diff_eq!(val, i as f64, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_vector_arange_fractional() {
        let v = VectorBuilder::<f64>::arange(0.0, 1.0, 0.5);
        assert_eq!(v.len(), 2);
        assert_abs_diff_eq!(v[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_vector_arange_empty() {
        // step 0 → empty
        let v = VectorBuilder::<f64>::arange(0.0, 5.0, 0.0);
        assert_eq!(v.len(), 0);
        // wrong direction → empty
        let v2 = VectorBuilder::<f64>::arange(5.0, 0.0, 1.0);
        assert_eq!(v2.len(), 0);
    }

    #[test]
    fn test_vector_logspace() {
        let v = VectorBuilder::<f64>::logspace(0.0, 3.0, 4);
        assert_eq!(v.len(), 4);
        assert_abs_diff_eq!(v[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v[1], 10.0, epsilon = 1e-8);
        assert_abs_diff_eq!(v[2], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(v[3], 1000.0, epsilon = 1e-4);
    }

    #[test]
    fn test_vector_rand() {
        let v = VectorBuilder::<f64>::rand(20, 7);
        assert_eq!(v.len(), 20);
        assert!(v.iter().all(|&x| x >= 0.0 && x < 1.0));
        // Determinism
        let v2 = VectorBuilder::<f64>::rand(20, 7);
        assert_eq!(v, v2);
    }

    #[test]
    fn test_vector_randn() {
        let v = VectorBuilder::<f64>::randn(1000, 123);
        assert_eq!(v.len(), 1000);
        let mean = v.mean().expect("non-empty");
        assert!(mean.abs() < 0.2, "mean={mean}");
    }

    // --- ArrayBuilder tests ---

    #[test]
    fn test_array_builder_zeros_2d() {
        let a = ArrayBuilder::<f64, ::ndarray::Ix2>::zeros(::ndarray::Ix2(3, 4));
        assert_eq!(a.shape(), &[3, 4]);
        assert!(a.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_array_builder_zeros_3d() {
        let a = ArrayBuilder::<f64, ::ndarray::Ix3>::zeros(::ndarray::Ix3(2, 3, 4));
        assert_eq!(a.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_array_builder_zeros_dyn() {
        let a = ArrayBuilder::<f64, ::ndarray::IxDyn>::zeros_dyn(&[2, 3, 4]);
        assert_eq!(a.ndim(), 3);
        assert_eq!(a.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_array_builder_full() {
        let a = ArrayBuilder::<i32, ::ndarray::Ix2>::full(::ndarray::Ix2(3, 3), 7);
        assert!(a.iter().all(|&v| v == 7));
    }

    #[test]
    fn test_array_builder_from_vec_ok() {
        let a = ArrayBuilder::<f64, ::ndarray::Ix2>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ::ndarray::Ix2(2, 3),
        )
        .expect("valid shape");
        assert_eq!(a[[1, 2]], 6.0);
    }

    #[test]
    fn test_array_builder_from_vec_err() {
        let result = ArrayBuilder::<f64, ::ndarray::Ix2>::from_vec(
            vec![1.0, 2.0, 3.0],
            ::ndarray::Ix2(2, 3), // needs 6 elements
        );
        assert!(result.is_err());
    }
}
