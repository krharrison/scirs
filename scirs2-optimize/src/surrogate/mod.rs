//! Surrogate-Assisted Optimization
//!
//! This module provides surrogate model-based optimization methods that build
//! approximate models of expensive objective functions to guide the search
//! for the optimum with fewer function evaluations.
//!
//! ## Surrogate Models
//!
//! - **RBF Surrogate**: Radial Basis Function interpolation (polyharmonic, multiquadric, thin-plate)
//! - **Kriging**: Gaussian Process surrogate with nugget parameter for noise handling
//! - **Ensemble**: Ensemble of surrogates with automatic model selection
//!
//! ## Usage
//!
//! All surrogates implement the [`SurrogateModel`] trait, which provides a common
//! interface for fitting, predicting, and estimating uncertainty.

pub mod ensemble;
pub mod kriging;
pub mod rbf_surrogate;

pub use ensemble::{EnsembleOptions, EnsembleSurrogate, ModelSelectionCriterion};
pub use kriging::{CorrelationFunction, KrigingOptions, KrigingSurrogate};
pub use rbf_surrogate::{RbfKernel, RbfOptions, RbfSurrogate};

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Common trait for surrogate models
pub trait SurrogateModel {
    /// Fit the surrogate model to the provided data
    ///
    /// # Arguments
    /// * `x` - Training points, shape (n_samples, n_features)
    /// * `y` - Function values at training points, shape (n_samples,)
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> OptimizeResult<()>;

    /// Predict the function value at a new point
    ///
    /// # Arguments
    /// * `x` - Point to predict at, shape (n_features,)
    ///
    /// # Returns
    /// Predicted function value
    fn predict(&self, x: &Array1<f64>) -> OptimizeResult<f64>;

    /// Predict the function value and uncertainty at a new point
    ///
    /// # Arguments
    /// * `x` - Point to predict at, shape (n_features,)
    ///
    /// # Returns
    /// (predicted mean, predicted standard deviation)
    fn predict_with_uncertainty(&self, x: &Array1<f64>) -> OptimizeResult<(f64, f64)>;

    /// Predict at multiple points
    ///
    /// # Arguments
    /// * `x` - Points to predict at, shape (n_points, n_features)
    ///
    /// # Returns
    /// Predicted values, shape (n_points,)
    fn predict_batch(&self, x: &Array2<f64>) -> OptimizeResult<Array1<f64>> {
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);
        for i in 0..n {
            predictions[i] = self.predict(&x.row(i).to_owned())?;
        }
        Ok(predictions)
    }

    /// Get the number of training points
    fn n_samples(&self) -> usize;

    /// Get the dimensionality of the problem
    fn n_features(&self) -> usize;

    /// Add a new data point and update the model
    fn update(&mut self, x: &Array1<f64>, y: f64) -> OptimizeResult<()>;
}

/// Compute pairwise squared Euclidean distances between rows of X and Y
pub fn pairwise_sq_distances(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let m = y.nrows();
    let mut dists = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut sq_dist = 0.0;
            for k in 0..x.ncols() {
                let diff = x[[i, k]] - y[[j, k]];
                sq_dist += diff * diff;
            }
            dists[[i, j]] = sq_dist;
        }
    }
    dists
}

/// Solve a symmetric positive definite linear system Ax = b using Cholesky decomposition
/// Returns x
pub fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(OptimizeError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }
    if n != b.len() {
        return Err(OptimizeError::InvalidInput(
            "Matrix and vector dimensions must match".to_string(),
        ));
    }

    // Cholesky factorization: A = L * L^T
    let mut l = Array2::zeros((n, n));
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - sum;
        if diag <= 0.0 {
            return Err(OptimizeError::ComputationError(
                "Matrix is not positive definite".to_string(),
            ));
        }
        l[[j, j]] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
        }
    }

    // Forward substitution: L * z = b
    let mut z = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * z[j];
        }
        z[i] = (b[i] - sum) / l[[i, i]];
    }

    // Back substitution: L^T * x = z
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        x[i] = (z[i] - sum) / l[[i, i]];
    }

    Ok(x)
}

/// Solve a general linear system Ax = b using LU decomposition with partial pivoting
pub fn solve_general(a: &Array2<f64>, b: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(OptimizeError::InvalidInput(
            "Dimension mismatch in linear system".to_string(),
        ));
    }

    // LU decomposition with partial pivoting
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[[i, k]].abs() > max_val {
                max_val = lu[[i, k]].abs();
                max_row = i;
            }
        }

        if max_val < 1e-30 {
            return Err(OptimizeError::ComputationError(
                "Singular or near-singular matrix in linear solve".to_string(),
            ));
        }

        // Swap rows
        if max_row != k {
            perm.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        // Eliminate
        for i in (k + 1)..n {
            lu[[i, k]] /= lu[[k, k]];
            for j in (k + 1)..n {
                lu[[i, j]] -= lu[[i, k]] * lu[[k, j]];
            }
        }
    }

    // Apply permutation to b
    let mut pb = Array1::zeros(n);
    for i in 0..n {
        pb[i] = b[perm[i]];
    }

    // Forward substitution (L * y = Pb)
    let mut y = pb;
    for i in 1..n {
        for j in 0..i {
            y[i] -= lu[[i, j]] * y[j];
        }
    }

    // Back substitution (U * x = y)
    let mut x = y;
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu[[i, j]] * x[j];
        }
        x[i] /= lu[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_distances() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0])
            .expect("Array creation failed");
        let y = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0])
            .expect("Array creation failed");
        let dists = pairwise_sq_distances(&x, &y);
        assert!((dists[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((dists[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((dists[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((dists[[1, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_spd() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // Solution: x = [-1/8, 3/4]
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0])
            .expect("Array creation failed");
        let b = Array1::from_vec(vec![1.0, 2.0]);
        let x = solve_spd(&a, &b).expect("SPD solve failed");
        assert!((x[0] - (-0.125)).abs() < 1e-10);
        assert!((x[1] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_solve_general() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Array creation failed");
        let b = Array1::from_vec(vec![5.0, 11.0]);
        let x = solve_general(&a, &b).expect("General solve failed");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }
}
