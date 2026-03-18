//! Complete ndarray re-export for SciRS2 ecosystem
//!
//! This module provides a single, unified access point for ALL ndarray functionality,
//! ensuring SciRS2 POLICY compliance across the entire ecosystem.
//!
//! ## Design Philosophy
//!
//! 1. **Complete Feature Parity**: All ndarray functionality available through scirs2-core
//! 2. **Zero Breaking Changes**: Existing ndarray_ext continues to work
//! 3. **Policy Compliance**: No need for direct ndarray imports anywhere
//! 4. **Single Source of Truth**: One place for all array operations
//!
//! ## Usage
//!
//! ```rust
//! // Instead of:
//! use ndarray::{Array, array, s, Axis};  // ❌ POLICY violation
//!
//! // Use:
//! use scirs2_core::ndarray::*;  // ✅ POLICY compliant
//!
//! let arr = array![[1, 2], [3, 4]];
//! let slice = arr.slice(s![.., 0]);
//! ```

// ========================================
// COMPLETE NDARRAY RE-EXPORT
// ========================================

// Complete ndarray 0.17 re-export (no version switching needed anymore)
// Use ::ndarray to refer to the external crate (not this module)
pub use ::ndarray::*;

// Note: All macros (array!, s!, azip!, etc.) are already included via `pub use ::ndarray::*;`

// ========================================
// NDARRAY-RELATED CRATE RE-EXPORTS
// ========================================

#[cfg(feature = "random")]
pub use ndarray_rand::{rand_distr as distributions, RandomExt, SamplingStrategy};

// Note: ndarray_rand is compatible with both ndarray 0.16 and 0.17

// NOTE: ndarray_linalg removed - using OxiBLAS via scirs2_core::linalg module

#[cfg(feature = "array_stats")]
pub use ndarray_stats::{
    errors as stats_errors, interpolate, CorrelationExt, DeviationExt, MaybeNan, QuantileExt,
    Sort1dExt, SummaryStatisticsExt,
};

// NOTE: ndarray_npy removed to eliminate `zip` crate from dependency tree (COOLJAPAN Pure Rust Policy)

// ========================================
// ENHANCED FUNCTIONALITY
// ========================================

/// Additional utilities for SciRS2 ecosystem
pub mod utils {
    use super::*;

    /// Create an identity matrix
    pub fn eye<A>(n: usize) -> Array2<A>
    where
        A: Clone + num_traits::Zero + num_traits::One,
    {
        let mut arr = Array2::zeros((n, n));
        for i in 0..n {
            arr[[i, i]] = A::one();
        }
        arr
    }

    /// Create a diagonal matrix from a vector
    pub fn diag<A>(v: &Array1<A>) -> Array2<A>
    where
        A: Clone + num_traits::Zero,
    {
        let n = v.len();
        let mut arr = Array2::zeros((n, n));
        for i in 0..n {
            arr[[i, i]] = v[i].clone();
        }
        arr
    }

    /// Check if arrays are approximately equal
    pub fn allclose<A, D>(
        a: &ArrayBase<impl Data<Elem = A>, D>,
        b: &ArrayBase<impl Data<Elem = A>, D>,
        rtol: A,
        atol: A,
    ) -> bool
    where
        A: PartialOrd
            + std::ops::Sub<Output = A>
            + std::ops::Mul<Output = A>
            + std::ops::Add<Output = A>
            + Clone,
        D: Dimension,
    {
        if a.shape() != b.shape() {
            return false;
        }

        a.iter().zip(b.iter()).all(|(a_val, b_val)| {
            let diff = if a_val > b_val {
                a_val.clone() - b_val.clone()
            } else {
                b_val.clone() - a_val.clone()
            };

            let threshold = atol.clone()
                + rtol.clone()
                    * (if a_val > b_val {
                        a_val.clone()
                    } else {
                        b_val.clone()
                    });

            diff <= threshold
        })
    }

    /// Concatenate arrays along an axis
    pub fn concatenate<A, D>(
        axis: Axis,
        arrays: &[ArrayView<A, D>],
    ) -> Result<Array<A, D>, ShapeError>
    where
        A: Clone,
        D: Dimension + RemoveAxis,
    {
        ndarray::concatenate(axis, arrays)
    }

    /// Stack arrays along a new axis
    pub fn stack<A, D>(
        axis: Axis,
        arrays: &[ArrayView<A, D>],
    ) -> Result<Array<A, D::Larger>, ShapeError>
    where
        A: Clone,
        D: Dimension,
        D::Larger: RemoveAxis,
    {
        ndarray::stack(axis, arrays)
    }
}

// ========================================
// COMPATIBILITY LAYER
// ========================================

/// Compatibility module for smooth migration from fragmented imports
/// and ndarray version changes (SciRS2 POLICY compliance)
pub mod compat {
    pub use super::*;
    use crate::numeric::{Float, FromPrimitive};

    /// Alias for commonly used types to match existing usage patterns
    pub type DynArray<T> = ArrayD<T>;
    pub type Matrix<T> = Array2<T>;
    pub type Vector<T> = Array1<T>;
    pub type Tensor3<T> = Array3<T>;
    pub type Tensor4<T> = Array4<T>;

    /// Compatibility extensions for ndarray statistical operations
    ///
    /// This trait provides stable statistical operation APIs that remain consistent
    /// across ndarray version updates, implementing the SciRS2 POLICY principle
    /// of isolating external dependency changes to scirs2-core only.
    ///
    /// ## Rationale
    ///
    /// ndarray's statistical methods have changed across versions:
    /// - v0.16: `.mean()` returns `Option<T>`
    /// - v0.17: `.mean()` returns `T` directly (may be NaN for invalid operations)
    ///
    /// This trait provides a consistent API regardless of the underlying ndarray version.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use scirs2_core::ndarray::{Array1, compat::ArrayStatCompat};
    ///
    /// let data = Array1::from(vec![1.0, 2.0, 3.0]);
    /// let mean = data.mean_or(0.0);  // Stable API across ndarray versions
    /// ```
    pub trait ArrayStatCompat<T> {
        /// Compute the mean of the array, returning a default value if computation fails
        ///
        /// This method abstracts over ndarray version differences:
        /// - For ndarray 0.16: Unwraps the Option, using default if None
        /// - For ndarray 0.17+: Returns the value, using default if NaN
        fn mean_or(&self, default: T) -> T;

        /// Compute the variance with optional default
        fn var_or(&self, ddof: T, default: T) -> T;

        /// Compute the standard deviation with optional default
        fn std_or(&self, ddof: T, default: T) -> T;
    }

    impl<T, S, D> ArrayStatCompat<T> for ArrayBase<S, D>
    where
        T: Float + FromPrimitive,
        S: Data<Elem = T>,
        D: Dimension,
    {
        fn mean_or(&self, default: T) -> T {
            // ndarray returns Option<T> in both 0.16 and 0.17
            self.mean().unwrap_or(default)
        }

        fn var_or(&self, ddof: T, default: T) -> T {
            // ndarray returns T directly (may be NaN for invalid inputs)
            let v = self.var(ddof);
            if v.is_nan() {
                default
            } else {
                v
            }
        }

        fn std_or(&self, ddof: T, default: T) -> T {
            // ndarray returns T directly (may be NaN for invalid inputs)
            let s = self.std(ddof);
            if s.is_nan() {
                default
            } else {
                s
            }
        }
    }

    /// Re-export from ndarray_ext for backward compatibility
    pub use crate::ndarray_ext::{
        broadcast_1d_to_2d,
        broadcast_apply,
        fancy_index_2d,
        // Keep existing extended functionality
        indexing,
        is_broadcast_compatible,
        manipulation,
        mask_select,
        matrix,
        reshape_2d,
        split_2d,
        stack_2d,
        stats,
        take_2d,
        transpose_2d,
        where_condition,
    };
}

// ========================================
// PRELUDE MODULE
// ========================================

/// Prelude module with most commonly used items
pub mod prelude {
    pub use super::{
        arr1,
        arr2,
        // Essential macros
        array,
        azip,
        // Utilities
        concatenate,
        s,
        stack,

        stack as stack_fn,
        // Essential types
        Array,
        Array0,
        Array1,
        Array2,
        Array3,
        ArrayD,
        ArrayView,
        ArrayView1,
        ArrayView2,
        ArrayViewMut,

        // Common operations
        Axis,
        // Essential traits
        Dimension,
        Ix1,
        Ix2,
        Ix3,
        IxDyn,
        ScalarOperand,
        ShapeBuilder,

        Zip,
    };

    #[cfg(feature = "random")]
    pub use super::RandomExt;

    // Useful type aliases
    pub type Matrix<T> = super::Array2<T>;
    pub type Vector<T> = super::Array1<T>;
}

// ========================================
// EXAMPLES MODULE
// ========================================

#[cfg(test)]
pub mod examples {
    //! Examples demonstrating unified ndarray access through scirs2-core

    use super::*;

    /// Example: Using all essential ndarray features through scirs2-core
    ///
    /// ```
    /// use scirs2_core::ndarray::*;
    ///
    /// // Create arrays using the array! macro
    /// let a = array![[1, 2, 3], [4, 5, 6]];
    ///
    /// // Use the s! macro for slicing
    /// let row = a.slice(s![0, ..]);
    /// let col = a.slice(s![.., 1]);
    ///
    /// // Use Axis for operations
    /// let sum_axis0 = a.sum_axis(Axis(0));
    /// let mean_axis1 = a.mean_axis(Axis(1));
    ///
    /// // Stack and concatenate
    /// let b = array![[7, 8, 9], [10, 11, 12]];
    /// let stacked = stack![Axis(0), a, b];
    ///
    /// // Views and iteration
    /// for row in a.axis_iter(Axis(0)) {
    ///     println!("Row: {:?}", row);
    /// }
    /// ```
    #[test]
    fn test_complete_functionality() {
        // Array creation
        let a = array![[1., 2.], [3., 4.]];
        assert_eq!(a.shape(), &[2, 2]);

        // Slicing with s! macro
        let slice = a.slice(s![.., 0]);
        assert_eq!(slice.len(), 2);

        // Mathematical operations
        let b = &a + &a;
        assert_eq!(b[[0, 0]], 2.);

        // Axis operations
        let sum = a.sum_axis(Axis(0));
        assert_eq!(sum.len(), 2);

        // Broadcasting
        let c = array![1., 2.];
        let d = &a + &c;
        assert_eq!(d[[0, 0]], 2.);
    }
}

// ========================================
// MIGRATION GUIDE
// ========================================

pub mod migration_guide {
    //! # Migration Guide: From Fragmented to Unified ndarray Access
    //!
    //! ## Before (Fragmented, Policy-Violating)
    //!
    //! ```rust,ignore
    //! // Different files used different imports
    //! use scirs2_autograd::ndarray::{Array1, array};
    //! use scirs2_core::ndarray_ext::{ArrayView};
    //! use ndarray::{s!, Axis};  // POLICY VIOLATION!
    //! ```
    //!
    //! ## After (Unified, Policy-Compliant)
    //!
    //! ```rust,ignore
    //! // Single, consistent import
    //! use scirs2_core::ndarray::*;
    //!
    //! // Everything works:
    //! let arr = array![[1, 2], [3, 4]];
    //! let slice = arr.slice(s![.., 0]);
    //! let view: ArrayView<_, _> = arr.view();
    //! let sum = arr.sum_axis(Axis(0));
    //! ```
    //!
    //! ## Benefits
    //!
    //! 1. **Single Import Path**: No more confusion about where to import from
    //! 2. **Complete Functionality**: All ndarray features available
    //! 3. **Policy Compliance**: No direct ndarray imports needed
    //! 4. **Future-Proof**: Centralized control over array functionality
    //!
    //! ## Quick Reference
    //!
    //! | Old Import | New Import |
    //! |------------|------------|
    //! | `use ndarray::{Array, array}` | `use scirs2_core::ndarray::{Array, array}` |
    //! | `use scirs2_autograd::ndarray::*` | `use scirs2_core::ndarray::*` |
    //! | `use scirs2_core::ndarray_ext::*` | `use scirs2_core::ndarray::*` |
    //! | `use ndarray::{s!, Axis}` | `use scirs2_core::ndarray::{s, Axis}` |
}

// Re-export compatibility traits for easy access
pub use compat::ArrayStatCompat;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_macro_available() {
        let arr = array![[1, 2], [3, 4]];
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[0, 0]], 1);
    }

    #[test]
    fn test_s_macro_available() {
        let arr = array![[1, 2, 3], [4, 5, 6]];
        let slice = arr.slice(s![.., 1..]);
        assert_eq!(slice.shape(), &[2, 2]);
    }

    #[test]
    fn test_axis_operations() {
        let arr = array![[1., 2.], [3., 4.]];
        let sum = arr.sum_axis(Axis(0));
        assert_eq!(sum, array![4., 6.]);
    }

    #[test]
    fn test_views_and_iteration() {
        let mut arr = array![[1, 2], [3, 4]];

        // Test immutable view first
        {
            let view: ArrayView<_, _> = arr.view();
            for val in view.iter() {
                assert!(*val > 0);
            }
        }

        // Test mutable view after immutable view is dropped
        {
            let mut view_mut: ArrayViewMut<_, _> = arr.view_mut();
            for val in view_mut.iter_mut() {
                *val *= 2;
            }
        }

        assert_eq!(arr[[0, 0]], 2);
    }

    #[test]
    fn test_concatenate_and_stack() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 6], [7, 8]];

        // Concatenate along axis 0
        let concat = concatenate(Axis(0), &[a.view(), b.view()]).expect("Operation failed");
        assert_eq!(concat.shape(), &[4, 2]);

        // Stack along new axis
        let stacked =
            crate::ndarray::stack(Axis(0), &[a.view(), b.view()]).expect("Operation failed");
        assert_eq!(stacked.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_zip_operations() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];
        let mut c = array![0, 0, 0];

        azip!((a in &a, b in &b, c in &mut c) {
            *c = a + b;
        });

        assert_eq!(c, array![5, 7, 9]);
    }

    #[test]
    fn test_array_stat_compat() {
        use compat::ArrayStatCompat;

        // Test mean_or
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(data.mean_or(0.0), 3.0);

        let empty = Array1::<f64>::from(vec![]);
        assert_eq!(empty.mean_or(0.0), 0.0);

        // Test var_or
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = data.var_or(1.0, 0.0);
        assert!(var > 0.0);

        // Test std_or
        let std = data.std_or(1.0, 0.0);
        assert!(std > 0.0);
    }
}

// ========================================
// NDARRAY-LINALG COMPATIBILITY LAYER
// ========================================

/// ndarray-linalg compatibility layer for backward compatibility
///
/// Provides traits matching ndarray-linalg API using OxiBLAS v0.1.2+ backend
#[cfg(feature = "linalg")]
pub mod ndarray_linalg {
    use crate::linalg::prelude::*;
    use crate::ndarray::*;
    use num_complex::Complex;

    // Import OxiBLAS v0.1.2+ Complex functions
    use oxiblas_ndarray::lapack::{
        cholesky_hermitian_ndarray, eig_hermitian_ndarray, qr_complex_ndarray, svd_complex_ndarray,
    };

    // Re-export error types
    pub use crate::linalg::{LapackError, LapackResult};

    /// UPLO enum for triangular matrix specification
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum UPLO {
        Upper,
        Lower,
    }

    /// Linear system solver trait
    pub trait Solve<A> {
        fn solve_into(&self, b: &Array1<A>) -> Result<Array1<A>, LapackError>;
    }

    impl Solve<f64> for Array2<f64> {
        #[inline]
        fn solve_into(&self, b: &Array1<f64>) -> Result<Array1<f64>, LapackError> {
            solve_ndarray(self, b)
        }
    }

    impl Solve<Complex<f64>> for Array2<Complex<f64>> {
        #[inline]
        fn solve_into(
            &self,
            b: &Array1<Complex<f64>>,
        ) -> Result<Array1<Complex<f64>>, LapackError> {
            solve_ndarray(self, b)
        }
    }

    /// SVD trait
    pub trait SVD {
        type Elem;
        type Real;

        fn svd(
            &self,
            compute_u: bool,
            compute_vt: bool,
        ) -> Result<(Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>), LapackError>;
    }

    impl SVD for Array2<f64> {
        type Elem = f64;
        type Real = f64;

        #[inline]
        fn svd(
            &self,
            _compute_u: bool,
            _compute_vt: bool,
        ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), LapackError> {
            let result = svd_ndarray(self)?;
            Ok((result.u, result.s, result.vt))
        }
    }

    impl SVD for Array2<Complex<f64>> {
        type Elem = Complex<f64>;
        type Real = f64;

        #[inline]
        fn svd(
            &self,
            _compute_u: bool,
            _compute_vt: bool,
        ) -> Result<(Array2<Complex<f64>>, Array1<f64>, Array2<Complex<f64>>), LapackError>
        {
            let result = svd_complex_ndarray(self)?;
            Ok((result.u, result.s, result.vt))
        }
    }

    /// Hermitian/symmetric eigenvalue decomposition trait
    pub trait Eigh {
        type Elem;
        type Real;

        fn eigh(&self, uplo: UPLO)
            -> Result<(Array1<Self::Real>, Array2<Self::Elem>), LapackError>;
    }

    impl Eigh for Array2<f64> {
        type Elem = f64;
        type Real = f64;

        #[inline]
        fn eigh(&self, _uplo: UPLO) -> Result<(Array1<f64>, Array2<f64>), LapackError> {
            let result = eig_symmetric(self)?;
            Ok((result.eigenvalues, result.eigenvectors))
        }
    }

    impl Eigh for Array2<Complex<f64>> {
        type Elem = Complex<f64>;
        type Real = f64;

        #[inline]
        fn eigh(&self, _uplo: UPLO) -> Result<(Array1<f64>, Array2<Complex<f64>>), LapackError> {
            eig_hermitian_ndarray(self)
        }
    }

    /// Matrix norm trait
    pub trait Norm {
        type Real;

        fn norm_l2(&self) -> Result<Self::Real, LapackError>;
    }

    impl Norm for Array2<f64> {
        type Real = f64;

        #[inline]
        fn norm_l2(&self) -> Result<f64, LapackError> {
            let sum_sq: f64 = self.iter().map(|x| x * x).sum();
            Ok(sum_sq.sqrt())
        }
    }

    impl Norm for Array2<Complex<f64>> {
        type Real = f64;

        #[inline]
        fn norm_l2(&self) -> Result<f64, LapackError> {
            let sum_sq: f64 = self.iter().map(|x| x.norm_sqr()).sum();
            Ok(sum_sq.sqrt())
        }
    }

    // Norm trait for Array1 (vectors)
    impl Norm for Array1<f64> {
        type Real = f64;

        #[inline]
        fn norm_l2(&self) -> Result<f64, LapackError> {
            let sum_sq: f64 = self.iter().map(|x| x * x).sum();
            Ok(sum_sq.sqrt())
        }
    }

    impl Norm for Array1<Complex<f64>> {
        type Real = f64;

        #[inline]
        fn norm_l2(&self) -> Result<f64, LapackError> {
            let sum_sq: f64 = self.iter().map(|x| x.norm_sqr()).sum();
            Ok(sum_sq.sqrt())
        }
    }

    /// QR decomposition trait
    pub trait QR {
        type Elem;

        fn qr(&self) -> Result<(Array2<Self::Elem>, Array2<Self::Elem>), LapackError>;
    }

    impl QR for Array2<f64> {
        type Elem = f64;

        #[inline]
        fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), LapackError> {
            let result = qr_ndarray(self)?;
            Ok((result.q, result.r))
        }
    }

    impl QR for Array2<Complex<f64>> {
        type Elem = Complex<f64>;

        #[inline]
        fn qr(&self) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>), LapackError> {
            let result = qr_complex_ndarray(self)?;
            Ok((result.q, result.r))
        }
    }

    /// Eigenvalue decomposition trait (general matrices)
    pub trait Eig {
        type Elem;

        fn eig(&self) -> Result<(Array1<Self::Elem>, Array2<Self::Elem>), LapackError>;
    }

    // For general complex matrices use the complex QR algorithm (shifted QR iteration).
    // The algorithm:
    //  1. Reduce A to upper Hessenberg form H = Q^H A Q via Householder reflections.
    //  2. Apply complex QR iteration with Wilkinson shifts on H until convergence.
    //  3. Extract eigenvalues from the diagonal of the converged quasi-triangular form.
    //  4. Compute right eigenvectors by back-substitution on the upper-triangular Schur form.
    //  5. Transform eigenvectors back: X = Q * V.
    impl Eig for Array2<Complex<f64>> {
        type Elem = Complex<f64>;

        fn eig(&self) -> Result<(Array1<Complex<f64>>, Array2<Complex<f64>>), LapackError> {
            let (m, n) = self.dim();
            if m != n {
                return Err(LapackError::DimensionMismatch(
                    "Matrix must be square for eigendecomposition".to_string(),
                ));
            }
            if n == 0 {
                return Ok((
                    Array1::<Complex<f64>>::zeros(0),
                    Array2::<Complex<f64>>::zeros((0, 0)),
                ));
            }
            if n == 1 {
                let eigenvalue = self[[0, 0]];
                let eigenvector = Array2::from_elem((1, 1), Complex::new(1.0, 0.0));
                return Ok((Array1::from_vec(vec![eigenvalue]), eigenvector));
            }

            // Step 1: Reduce to upper Hessenberg form via Householder reflections.
            let mut h = self.clone();
            let mut q = Array2::<Complex<f64>>::eye(n);

            for col in 0..n.saturating_sub(2) {
                let xlen = n - col - 1;
                if xlen == 0 {
                    continue;
                }

                let mut x: Vec<Complex<f64>> = (col + 1..n).map(|r| h[[r, col]]).collect();

                let norm_x = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                if norm_x < 1e-300 {
                    continue;
                }

                let phase = if x[0].norm() > 1e-300 {
                    x[0] / x[0].norm()
                } else {
                    Complex::new(1.0, 0.0)
                };
                x[0] += phase * norm_x;

                let norm_v = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                if norm_v < 1e-300 {
                    continue;
                }
                let v: Vec<Complex<f64>> = x.iter().map(|vi| *vi / norm_v).collect();

                // Apply (I - 2vv^H) from the left to H
                for c in col..n {
                    let dot: Complex<f64> = v
                        .iter()
                        .enumerate()
                        .map(|(i, &vi)| vi.conj() * h[[col + 1 + i, c]])
                        .sum();
                    for (i, &vi) in v.iter().enumerate() {
                        h[[col + 1 + i, c]] -= Complex::new(2.0, 0.0) * vi * dot;
                    }
                }

                // Apply (I - 2vv^H) from the right to H
                for r in 0..n {
                    let dot: Complex<f64> = v
                        .iter()
                        .enumerate()
                        .map(|(i, &vi)| h[[r, col + 1 + i]] * vi)
                        .sum();
                    for (i, &vi) in v.iter().enumerate() {
                        h[[r, col + 1 + i]] -= Complex::new(2.0, 0.0) * dot * vi.conj();
                    }
                }

                // Accumulate Q
                for r in 0..n {
                    let dot: Complex<f64> = v
                        .iter()
                        .enumerate()
                        .map(|(i, &vi)| q[[r, col + 1 + i]] * vi)
                        .sum();
                    for (i, &vi) in v.iter().enumerate() {
                        q[[r, col + 1 + i]] -= Complex::new(2.0, 0.0) * dot * vi.conj();
                    }
                }

                // Zero subdiagonal entries below the first subdiagonal in column col
                for r in col + 2..n {
                    h[[r, col]] = Complex::new(0.0, 0.0);
                }
            }

            // Step 2: Complex QR algorithm with Wilkinson shifts.
            const MAX_ITER: usize = 30;
            let mut p = n;

            'outer: while p > 1 {
                // Deflation check
                let mut deflated = false;
                for l in (1..p).rev() {
                    let sub = h[[l, l - 1]].norm();
                    let diag = h[[l - 1, l - 1]].norm() + h[[l, l]].norm();
                    if sub <= 1e-14 * diag || sub <= f64::MIN_POSITIVE.sqrt() {
                        h[[l, l - 1]] = Complex::new(0.0, 0.0);
                        if l == p - 1 {
                            p -= 1;
                            deflated = true;
                            break;
                        }
                    }
                }
                if deflated {
                    continue 'outer;
                }

                let mut converged_inner = false;
                for _iter in 0..MAX_ITER {
                    // Wilkinson shift from bottom 2x2 block
                    let a_sub = h[[p - 2, p - 2]];
                    let b_sub = h[[p - 2, p - 1]];
                    let c_sub = h[[p - 1, p - 2]];
                    let d_sub = h[[p - 1, p - 1]];
                    let tr = a_sub + d_sub;
                    let det = a_sub * d_sub - b_sub * c_sub;
                    let disc = (tr * tr - Complex::new(4.0, 0.0) * det).sqrt();
                    let mu1 = (tr + disc) * Complex::new(0.5, 0.0);
                    let mu2 = (tr - disc) * Complex::new(0.5, 0.0);
                    let shift = if (mu1 - d_sub).norm() < (mu2 - d_sub).norm() {
                        mu1
                    } else {
                        mu2
                    };

                    // One QR step with Givens rotations (preserves Hessenberg structure)
                    for k in 0..p.saturating_sub(1) {
                        let a_g = if k == 0 {
                            h[[0, 0]] - shift
                        } else {
                            h[[k, k - 1]]
                        };
                        let b_g = h[[k + 1, k]];
                        let r = (a_g.norm_sqr() + b_g.norm_sqr()).sqrt();
                        if r < 1e-300 {
                            continue;
                        }
                        let c = a_g / r;
                        let s = b_g / r;

                        // Apply from the left (rows k and k+1)
                        let col_start = if k == 0 { 0 } else { k - 1 };
                        for j in col_start..n {
                            let t1 = c.conj() * h[[k, j]] + s.conj() * h[[k + 1, j]];
                            let t2 = -s * h[[k, j]] + c * h[[k + 1, j]];
                            h[[k, j]] = t1;
                            h[[k + 1, j]] = t2;
                        }

                        // Apply from the right (cols k and k+1)
                        let row_max = (k + 2).min(p);
                        for i in 0..row_max {
                            let t1 = h[[i, k]] * c + h[[i, k + 1]] * s;
                            let t2 = h[[i, k]] * (-s.conj()) + h[[i, k + 1]] * c.conj();
                            h[[i, k]] = t1;
                            h[[i, k + 1]] = t2;
                        }

                        // Accumulate in Q
                        for i in 0..n {
                            let t1 = q[[i, k]] * c + q[[i, k + 1]] * s;
                            let t2 = q[[i, k]] * (-s.conj()) + q[[i, k + 1]] * c.conj();
                            q[[i, k]] = t1;
                            q[[i, k + 1]] = t2;
                        }
                    }

                    let sub_norm = h[[p - 1, p - 2]].norm();
                    let diag_norm = h[[p - 2, p - 2]].norm() + h[[p - 1, p - 1]].norm();
                    if sub_norm <= 1e-14 * diag_norm || sub_norm <= f64::MIN_POSITIVE.sqrt() {
                        h[[p - 1, p - 2]] = Complex::new(0.0, 0.0);
                        p -= 1;
                        converged_inner = true;
                        break;
                    }
                }

                if !converged_inner {
                    p -= 1; // Force deflation to avoid infinite loop
                }
            }

            // Step 3: Extract eigenvalues from diagonal of the Schur form
            let eigenvalues: Array1<Complex<f64>> = Array1::from_iter((0..n).map(|i| h[[i, i]]));

            // Step 4: Compute right eigenvectors by back-substitution from the
            // upper-triangular Schur form.
            let mut vecs = Array2::<Complex<f64>>::zeros((n, n));
            for ei in 0..n {
                let lambda = eigenvalues[ei];
                let mut v = vec![Complex::new(0.0, 0.0); n];
                v[ei] = Complex::new(1.0, 0.0);

                for row in (0..ei).rev() {
                    let mut sum = Complex::new(0.0, 0.0);
                    for col in row + 1..=ei {
                        sum += h[[row, col]] * v[col];
                    }
                    let diag = h[[row, row]] - lambda;
                    v[row] = if diag.norm() > 1e-14 {
                        -sum / diag
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                }

                let norm = v.iter().map(|vi| vi.norm_sqr()).sum::<f64>().sqrt();
                if norm > 1e-300 {
                    for vi in &mut v {
                        *vi /= norm;
                    }
                } else {
                    v[ei] = Complex::new(1.0, 0.0);
                }

                for row in 0..n {
                    vecs[[row, ei]] = v[row];
                }
            }

            // Step 5: Transform eigenvectors back to original basis: X = Q * V
            let eigenvectors = q.dot(&vecs);
            Ok((eigenvalues, eigenvectors))
        }
    }

    /// Cholesky decomposition trait
    pub trait Cholesky {
        type Elem;

        fn cholesky(&self, uplo: UPLO) -> Result<Array2<Self::Elem>, LapackError>;
    }

    impl Cholesky for Array2<f64> {
        type Elem = f64;

        #[inline]
        fn cholesky(&self, _uplo: UPLO) -> Result<Array2<f64>, LapackError> {
            let result = cholesky_ndarray(self)?;
            Ok(result.l)
        }
    }

    impl Cholesky for Array2<Complex<f64>> {
        type Elem = Complex<f64>;

        #[inline]
        fn cholesky(&self, _uplo: UPLO) -> Result<Array2<Complex<f64>>, LapackError> {
            let result = cholesky_hermitian_ndarray(self)?;
            Ok(result.l)
        }
    }
}
