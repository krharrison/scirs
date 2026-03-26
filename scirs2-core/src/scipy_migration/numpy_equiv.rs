//! NumPy -> SciRS2 / ndarray equivalence reference
//!
//! This module provides a searchable mapping from NumPy function names
//! to their SciRS2 or ndarray equivalents. NumPy's core array type
//! is represented by `ndarray::Array` (re-exported via `scirs2_core::ndarray`).
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::scipy_migration::numpy_equiv::{search_numpy, numpy_table};
//!
//! let hits = search_numpy("zeros");
//! assert!(!hits.is_empty());
//! ```

/// A single NumPy equivalence entry.
#[derive(Debug, Clone)]
pub struct NumpyEntry {
    /// NumPy function path (e.g., `"numpy.zeros"`)
    pub numpy_path: &'static str,
    /// Rust equivalent (ndarray or scirs2 path)
    pub rust_path: &'static str,
    /// Notes about differences
    pub notes: &'static str,
    /// NumPy usage example (Python)
    pub numpy_example: &'static str,
    /// Rust usage example
    pub rust_example: &'static str,
}

/// Returns the full NumPy equivalence table.
pub fn numpy_table() -> &'static [NumpyEntry] {
    &NUMPY_TABLE
}

/// Search for a NumPy function by name (case-insensitive partial match).
pub fn search_numpy(query: &str) -> Vec<&'static NumpyEntry> {
    let lower = query.to_lowercase();
    NUMPY_TABLE
        .iter()
        .filter(|e| e.numpy_path.to_lowercase().contains(&lower))
        .collect()
}

// ---------------------------------------------------------------------------
// NumPy Equivalence Table
// ---------------------------------------------------------------------------

static NUMPY_TABLE: [NumpyEntry; 42] = [
    // ===== Array Creation =====
    NumpyEntry {
        numpy_path: "numpy.array",
        rust_path: "ndarray::array! or ndarray::Array::from_vec",
        notes: "Use the array![] macro for literals, or Array::from_vec / from_shape_vec for dynamic data.",
        numpy_example: r#"import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([[1, 2], [3, 4]])"#,
        rust_example: r#"use ndarray::{array, Array1, Array2};
let a = array![1.0, 2.0, 3.0];
let b = array![[1.0, 2.0], [3.0, 4.0]];"#,
    },
    NumpyEntry {
        numpy_path: "numpy.zeros",
        rust_path: "ndarray::Array::zeros",
        notes: "Create an array filled with zeros.",
        numpy_example: r#"a = np.zeros((3, 4))"#,
        rust_example: r#"use ndarray::Array2;
let a = Array2::<f64>::zeros((3, 4));"#,
    },
    NumpyEntry {
        numpy_path: "numpy.ones",
        rust_path: "ndarray::Array::ones",
        notes: "Create an array filled with ones.",
        numpy_example: r#"a = np.ones((3, 4))"#,
        rust_example: r#"use ndarray::Array2;
let a = Array2::<f64>::ones((3, 4));"#,
    },
    NumpyEntry {
        numpy_path: "numpy.eye",
        rust_path: "ndarray::Array2::eye",
        notes: "Identity matrix.",
        numpy_example: r#"I = np.eye(3)"#,
        rust_example: r#"use ndarray::Array2;
let eye = Array2::<f64>::eye(3);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.full",
        rust_path: "ndarray::Array::from_elem",
        notes: "Create an array filled with a given value.",
        numpy_example: r#"a = np.full((3, 4), 7.0)"#,
        rust_example: r#"use ndarray::Array2;
let a = Array2::<f64>::from_elem((3, 4), 7.0);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.arange",
        rust_path: "ndarray::Array::range or ndarray::Array::linspace",
        notes: "arange with step: use Array::range(start, end, step).",
        numpy_example: r#"a = np.arange(0, 10, 0.5)"#,
        rust_example: r#"use ndarray::Array1;
let a = Array1::range(0.0, 10.0, 0.5);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linspace",
        rust_path: "ndarray::Array::linspace",
        notes: "Evenly spaced values over an interval.",
        numpy_example: r#"a = np.linspace(0, 1, 50)"#,
        rust_example: r#"use ndarray::Array1;
let a = Array1::linspace(0.0, 1.0, 50);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.empty",
        rust_path: "ndarray::Array::uninit or Array::zeros",
        notes: "Rust does not have uninitialized arrays in safe code. Use zeros() or from_elem().",
        numpy_example: r#"a = np.empty((3, 4))"#,
        rust_example: r#"use ndarray::Array2;
// No direct equivalent; use zeros instead:
let a = Array2::<f64>::zeros((3, 4));"#,
    },
    // ===== Shape Manipulation =====
    NumpyEntry {
        numpy_path: "numpy.reshape",
        rust_path: "ndarray::Array::into_shape_with_order",
        notes: "Reshape an array. The new shape must have the same total number of elements.",
        numpy_example: r#"b = a.reshape((2, 3))"#,
        rust_example: r#"let b = a.into_shape_with_order((2, 3))?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.transpose",
        rust_path: "ndarray::Array::t() or .reversed_axes()",
        notes: "Transpose. .t() returns a view; .reversed_axes() consumes the array.",
        numpy_example: r#"b = a.T
b = np.transpose(a)"#,
        rust_example: r#"let b = a.t();  // transposed view
let b = a.reversed_axes();  // consumes a"#,
    },
    NumpyEntry {
        numpy_path: "numpy.concatenate",
        rust_path: "ndarray::concatenate or ndarray::stack",
        notes: "Join arrays along an existing axis. Use ndarray::concatenate![] macro.",
        numpy_example: r#"c = np.concatenate([a, b], axis=0)"#,
        rust_example: r#"use ndarray::concatenate;
use ndarray::Axis;
let c = concatenate(Axis(0), &[a.view(), b.view()])?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.stack",
        rust_path: "ndarray::stack",
        notes: "Stack arrays along a new axis.",
        numpy_example: r#"c = np.stack([a, b], axis=0)"#,
        rust_example: r#"use ndarray::{stack, Axis};
let c = stack(Axis(0), &[a.view(), b.view()])?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.split",
        rust_path: "ndarray slicing",
        notes: "No direct split function; use array slicing with s![] macro.",
        numpy_example: r#"parts = np.split(a, 3, axis=0)"#,
        rust_example: r#"use ndarray::s;
let part0 = a.slice(s![0..n, ..]);
let part1 = a.slice(s![n..2*n, ..]);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.flatten",
        rust_path: "ndarray::Array::into_raw_vec or .iter()",
        notes: "Flatten to 1-D. Use .into_raw_vec() for owned, or .iter() for iteration.",
        numpy_example: r#"flat = a.flatten()"#,
        rust_example: r#"let flat: Vec<f64> = a.into_raw_vec();
// or iterate:
let flat: Array1<f64> = a.iter().cloned().collect();"#,
    },
    NumpyEntry {
        numpy_path: "numpy.squeeze",
        rust_path: "ndarray: remove_axis",
        notes: "Remove size-1 axes. Use .index_axis(Axis(n), 0) to remove a specific axis.",
        numpy_example: r#"b = np.squeeze(a)"#,
        rust_example: r#"use ndarray::Axis;
let b = a.index_axis(Axis(0), 0);  // remove axis 0 if size==1"#,
    },
    // ===== Math Operations =====
    NumpyEntry {
        numpy_path: "numpy.dot",
        rust_path: "ndarray .dot() method",
        notes: "Matrix/vector dot product. Array2.dot(&Array2), Array1.dot(&Array1).",
        numpy_example: r#"c = np.dot(a, b)"#,
        rust_example: r#"let c = a.dot(&b);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.matmul",
        rust_path: "ndarray .dot() method",
        notes: "Same as np.dot for 2-D arrays. For batched, see scirs2_linalg::batch_matmul.",
        numpy_example: r#"c = np.matmul(a, b)
c = a @ b"#,
        rust_example: r#"let c = a.dot(&b);
// Batched: use scirs2_linalg::prelude::batch_matmul"#,
    },
    NumpyEntry {
        numpy_path: "numpy.sum",
        rust_path: "ndarray .sum() or .sum_axis()",
        notes: "Sum all elements or along an axis.",
        numpy_example: r#"s = np.sum(a)
s = np.sum(a, axis=0)"#,
        rust_example: r#"let s = a.sum();
let s = a.sum_axis(ndarray::Axis(0));"#,
    },
    NumpyEntry {
        numpy_path: "numpy.mean",
        rust_path: "ndarray .mean() or scirs2_stats::mean",
        notes: "Mean value. ndarray's .mean() returns Option; scirs2_stats::mean returns Result.",
        numpy_example: r#"m = np.mean(a)"#,
        rust_example: r#"let m = a.mean();  // returns Option<f64>
// or:
use scirs2_stats::mean;
let m = mean(&a.view())?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.var",
        rust_path: "ndarray .var(ddof) or scirs2_stats::var",
        notes: "Variance. ndarray uses .var(ddof); scirs2_stats::var(x, ddof, workers).",
        numpy_example: r#"v = np.var(a, ddof=1)"#,
        rust_example: r#"let v = a.var(1.0);
// or:
use scirs2_stats::var;
let v = var(&a.view(), 1, None)?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.std",
        rust_path: "ndarray .std(ddof) or scirs2_stats::std",
        notes: "Standard deviation.",
        numpy_example: r#"s = np.std(a, ddof=1)"#,
        rust_example: r#"let s = a.std(1.0);
// or:
use scirs2_stats::std;
let s = std(&a.view(), 1, None)?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.max / numpy.min",
        rust_path: "ndarray: a.iter().cloned().fold() or a.fold()",
        notes: "No built-in max/min on ndarray; iterate or use scirs2_core utilities.",
        numpy_example: r#"mx = np.max(a)
mn = np.min(a)"#,
        rust_example: r#"let mx = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let mn = a.iter().cloned().fold(f64::INFINITY, f64::min);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.abs",
        rust_path: "ndarray .mapv(f64::abs)",
        notes: "Element-wise absolute value.",
        numpy_example: r#"b = np.abs(a)"#,
        rust_example: r#"let b = a.mapv(f64::abs);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.sqrt",
        rust_path: "ndarray .mapv(f64::sqrt)",
        notes: "Element-wise square root.",
        numpy_example: r#"b = np.sqrt(a)"#,
        rust_example: r#"let b = a.mapv(f64::sqrt);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.exp",
        rust_path: "ndarray .mapv(f64::exp)",
        notes: "Element-wise exponential.",
        numpy_example: r#"b = np.exp(a)"#,
        rust_example: r#"let b = a.mapv(f64::exp);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.log",
        rust_path: "ndarray .mapv(f64::ln)",
        notes: "Element-wise natural logarithm. Note: Rust uses ln(), not log().",
        numpy_example: r#"b = np.log(a)"#,
        rust_example: r#"let b = a.mapv(f64::ln);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.sin / numpy.cos / numpy.tan",
        rust_path: "ndarray .mapv(f64::sin) / .mapv(f64::cos) / .mapv(f64::tan)",
        notes: "Element-wise trigonometric functions.",
        numpy_example: r#"b = np.sin(a)"#,
        rust_example: r#"let b = a.mapv(f64::sin);"#,
    },
    NumpyEntry {
        numpy_path: "numpy.clip",
        rust_path: "ndarray .mapv(|x| x.clamp(lo, hi))",
        notes: "Clip values to a range.",
        numpy_example: r#"b = np.clip(a, 0, 1)"#,
        rust_example: r#"let b = a.mapv(|x: f64| x.clamp(0.0, 1.0));"#,
    },
    NumpyEntry {
        numpy_path: "numpy.where",
        rust_path: "ndarray .mapv() with conditional or Zip",
        notes: "Conditional element selection. Use mapv or ndarray::Zip for element-wise conditions.",
        numpy_example: r#"b = np.where(a > 0, a, 0)"#,
        rust_example: r#"let b = a.mapv(|x: f64| if x > 0.0 { x } else { 0.0 });"#,
    },
    // ===== Linear Algebra =====
    NumpyEntry {
        numpy_path: "numpy.linalg.det",
        rust_path: "scirs2_linalg::prelude::det",
        notes: "Matrix determinant.",
        numpy_example: r#"d = np.linalg.det(a)"#,
        rust_example: r#"use scirs2_linalg::prelude::det;
let d = det(&a.view())?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linalg.inv",
        rust_path: "scirs2_linalg::prelude::inv",
        notes: "Matrix inverse.",
        numpy_example: r#"a_inv = np.linalg.inv(a)"#,
        rust_example: r#"use scirs2_linalg::prelude::inv;
let a_inv = inv(&a.view())?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linalg.solve",
        rust_path: "scirs2_linalg::prelude::solve",
        notes: "Solve Ax = b.",
        numpy_example: r#"x = np.linalg.solve(a, b)"#,
        rust_example: r#"use scirs2_linalg::prelude::solve;
let x = solve(&a.view(), &b.view())?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linalg.eig",
        rust_path: "scirs2_linalg::prelude::eig",
        notes: "Eigenvalue decomposition.",
        numpy_example: r#"vals, vecs = np.linalg.eig(a)"#,
        rust_example: r#"use scirs2_linalg::prelude::eig;
let (vals, vecs) = eig(&a.view())?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linalg.svd",
        rust_path: "scirs2_linalg::prelude::svd",
        notes: "Singular Value Decomposition.",
        numpy_example: r#"U, s, Vt = np.linalg.svd(a)"#,
        rust_example: r#"use scirs2_linalg::prelude::svd;
let (u, s, vt) = svd(&a.view())?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linalg.norm",
        rust_path: "scirs2_linalg::prelude::vector_norm / matrix_norm",
        notes: "Vector or matrix norm. Use vector_norm for 1-D, matrix_norm for 2-D.",
        numpy_example: r#"n = np.linalg.norm(a)"#,
        rust_example: r#"use scirs2_linalg::prelude::{vector_norm, matrix_norm};
let n = vector_norm(&v.view(), 2.0)?;
let n = matrix_norm(&m.view(), "fro")?;"#,
    },
    NumpyEntry {
        numpy_path: "numpy.linalg.lstsq",
        rust_path: "scirs2_linalg::prelude::lstsq",
        notes: "Least-squares solution.",
        numpy_example: r#"x, res, rank, sv = np.linalg.lstsq(a, b, rcond=None)"#,
        rust_example: r#"use scirs2_linalg::prelude::lstsq;
let result = lstsq(&a.view(), &b.view())?;"#,
    },
    // ===== Indexing & Slicing =====
    NumpyEntry {
        numpy_path: "numpy indexing: a[i, j]",
        rust_path: "ndarray: a[[i, j]] or a.get([i, j])",
        notes: "Direct indexing uses [[i, j]]. .get() returns Option for bounds checking.",
        numpy_example: r#"val = a[2, 3]"#,
        rust_example: r#"let val = a[[2, 3]];
// Safe version:
let val = a.get([2, 3]);"#,
    },
    NumpyEntry {
        numpy_path: "numpy slicing: a[1:3, :]",
        rust_path: "ndarray: a.slice(s![1..3, ..])",
        notes: "Use the s![] macro for slicing. Supports ranges, steps, and negative indices.",
        numpy_example: r#"b = a[1:3, :]
c = a[::2, :]"#,
        rust_example: r#"use ndarray::s;
let b = a.slice(s![1..3, ..]);
let c = a.slice(s![..;2, ..]);"#,
    },
    NumpyEntry {
        numpy_path: "numpy boolean indexing: a[a > 0]",
        rust_path: "ndarray: filtering with iterators",
        notes: "No direct boolean indexing; use .iter().filter() or .mapv() with conditions.",
        numpy_example: r#"b = a[a > 0]"#,
        rust_example: r#"let b: Vec<f64> = a.iter().filter(|&&x| x > 0.0).cloned().collect();"#,
    },
    // ===== Random =====
    NumpyEntry {
        numpy_path: "numpy.random.rand",
        rust_path: "scirs2_core::random",
        notes: "Uniform [0, 1) random array. Use scirs2_core random utilities.",
        numpy_example: r#"a = np.random.rand(3, 4)"#,
        rust_example: r#"use scirs2_core::random;
// Generate uniform random values using scirs2_core random module"#,
    },
    NumpyEntry {
        numpy_path: "numpy.random.randn",
        rust_path: "scirs2_core::random",
        notes: "Standard normal random array.",
        numpy_example: r#"a = np.random.randn(3, 4)"#,
        rust_example: r#"use scirs2_core::random;
// Generate normal random values using scirs2_core random module"#,
    },
    NumpyEntry {
        numpy_path: "numpy.random.seed",
        rust_path: "scirs2_core::random (seed-based RNG)",
        notes: "Use seeded RNG from scirs2_core::random for reproducibility.",
        numpy_example: r#"np.random.seed(42)"#,
        rust_example: r#"use scirs2_core::random;
// Create seeded RNG for reproducible results"#,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numpy_equiv_table_not_empty() {
        assert!(!numpy_table().is_empty());
        assert_eq!(numpy_table().len(), 42);
    }

    #[test]
    fn test_search_numpy() {
        let results = search_numpy("zeros");
        assert!(!results.is_empty());
        assert!(results.iter().any(|e| e.numpy_path == "numpy.zeros"));
    }

    #[test]
    fn test_search_numpy_case_insensitive() {
        let upper = search_numpy("ZEROS");
        let lower = search_numpy("zeros");
        assert_eq!(upper.len(), lower.len());
    }

    #[test]
    fn test_search_numpy_linalg() {
        let results = search_numpy("linalg");
        assert!(results.len() >= 5);
    }

    #[test]
    fn test_all_rust_paths_non_empty() {
        for entry in numpy_table() {
            assert!(
                !entry.rust_path.is_empty(),
                "Empty rust_path for {}",
                entry.numpy_path
            );
        }
    }

    #[test]
    fn test_no_duplicate_numpy_paths() {
        let table = numpy_table();
        let mut seen = std::collections::HashSet::new();
        for entry in table {
            assert!(
                seen.insert(entry.numpy_path),
                "Duplicate numpy_path: {}",
                entry.numpy_path
            );
        }
    }
}
