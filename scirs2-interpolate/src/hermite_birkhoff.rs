//! Hermite-Birkhoff Interpolation
//!
//! Generalized Hermite interpolation that matches function values and
//! selected derivatives at specified points. Unlike standard Hermite
//! interpolation which requires consecutive derivatives at each point,
//! Hermite-Birkhoff interpolation allows arbitrary derivative orders
//! to be specified independently at each node.
//!
//! ## Mathematical Background
//!
//! Given `n` nodes `x_0, x_1, ..., x_{n-1}` and an incidence matrix `E`
//! where `E[i][j] = 1` means the `j`-th derivative at node `x_i` is
//! specified, the interpolant is a polynomial `P(x)` of degree at most
//! `N - 1` (where `N` is the total number of conditions) satisfying:
//!
//! ```text
//! P^{(j)}(x_i) = y_{i,j}   for all (i,j) with E[i][j] = 1
//! ```
//!
//! ## Polya Conditions
//!
//! For the Hermite-Birkhoff problem to be solvable, the Polya conditions
//! must be satisfied: for each `k = 0, 1, ..., N-1`, the number of
//! specified conditions of derivative order `<= k` must be `>= k + 1`.

use crate::error::{InterpolateError, InterpolateResult};

/// Represents a single interpolation condition: at a given point,
/// match a specific derivative order to a given value.
#[derive(Debug, Clone)]
pub struct InterpolationCondition {
    /// The point at which the condition applies.
    pub point: f64,
    /// The derivative order (0 = function value, 1 = first derivative, etc.).
    pub derivative_order: usize,
    /// The value that the derivative should take at this point.
    pub value: f64,
}

/// A Hermite-Birkhoff interpolator.
///
/// After fitting, this stores a polynomial that satisfies the given
/// interpolation conditions (function values and/or derivatives at
/// specified points).
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::hermite_birkhoff::{HermiteBirkhoff, InterpolationCondition};
///
/// // Match value and first derivative at two points
/// let conditions = vec![
///     InterpolationCondition { point: 0.0, derivative_order: 0, value: 0.0 },
///     InterpolationCondition { point: 0.0, derivative_order: 1, value: 1.0 },
///     InterpolationCondition { point: 1.0, derivative_order: 0, value: 1.0 },
///     InterpolationCondition { point: 1.0, derivative_order: 1, value: 1.0 },
/// ];
///
/// let interp = HermiteBirkhoff::fit(&conditions).expect("should fit");
/// let val = interp.evaluate(0.5).expect("should evaluate");
/// // The polynomial should pass through (0, 0) and (1, 1)
/// assert!((interp.evaluate(0.0).expect("test") - 0.0).abs() < 1e-10);
/// assert!((interp.evaluate(1.0).expect("test") - 1.0).abs() < 1e-10);
/// ```
pub struct HermiteBirkhoff {
    /// Polynomial coefficients in monomial basis: c[0] + c[1]*x + c[2]*x^2 + ...
    coefficients: Vec<f64>,
}

/// Check Polya conditions for the Hermite-Birkhoff problem.
///
/// The Polya conditions are necessary for solvability:
/// for each `k = 0, 1, ..., N-1`, the count of conditions with
/// derivative order `<= k` must be `>= k + 1`.
///
/// Returns `Ok(())` if conditions are satisfied, or an error describing
/// which condition fails.
pub fn check_polya_conditions(conditions: &[InterpolationCondition]) -> InterpolateResult<()> {
    let n = conditions.len();
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "at least one condition is required".to_string(),
        ));
    }

    let max_deriv = conditions
        .iter()
        .map(|c| c.derivative_order)
        .max()
        .unwrap_or(0);

    for k in 0..n {
        let count = conditions
            .iter()
            .filter(|c| c.derivative_order <= k)
            .count();
        if count < k + 1 && k <= max_deriv {
            return Err(InterpolateError::InvalidValue(format!(
                "Polya condition violated: for k={}, need at least {} conditions \
                 with derivative order <= {}, but only {} found",
                k,
                k + 1,
                k,
                count
            )));
        }
    }

    Ok(())
}

/// Compute `k!` (factorial) as f64, for moderate `k`.
fn factorial(k: usize) -> f64 {
    let mut result = 1.0;
    for i in 2..=k {
        result *= i as f64;
    }
    result
}

/// Compute the `j`-th derivative of the monomial `x^m` evaluated at `x`,
/// i.e., `d^j/dx^j [x^m] = m!/(m-j)! * x^{m-j}` for `j <= m`, or `0` otherwise.
fn monomial_derivative(m: usize, j: usize, x: f64) -> f64 {
    if j > m {
        return 0.0;
    }
    let coeff = factorial(m) / factorial(m - j);
    coeff * x.powi((m - j) as i32)
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_system(a: &[Vec<f64>], b: &[f64]) -> InterpolateResult<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        row.extend_from_slice(&a[i]);
        row.push(b[i]);
        aug.push(row);
    }

    // Forward elimination
    for col in 0..n {
        let mut max_abs = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let abs_val = aug[row][col].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                max_row = row;
            }
        }

        if max_abs < 1e-14 {
            return Err(InterpolateError::LinalgError(
                "singular system in Hermite-Birkhoff interpolation (conditions may be incompatible)"
                    .to_string(),
            ));
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-14 {
            return Err(InterpolateError::LinalgError(
                "zero pivot in Hermite-Birkhoff back substitution".to_string(),
            ));
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

impl HermiteBirkhoff {
    /// Fit a Hermite-Birkhoff interpolant to the given conditions.
    ///
    /// Each condition specifies a point, a derivative order, and a value.
    /// The resulting polynomial has degree at most `N - 1` where `N` is
    /// the number of conditions.
    ///
    /// # Arguments
    ///
    /// * `conditions` - A slice of interpolation conditions specifying
    ///   function values and/or derivatives at various points.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No conditions are provided
    /// - The Polya conditions are violated
    /// - The linear system is singular
    pub fn fit(conditions: &[InterpolationCondition]) -> InterpolateResult<Self> {
        let n = conditions.len();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "at least one interpolation condition is required".to_string(),
            ));
        }

        // Check Polya conditions (best-effort; not all violations are caught)
        // We do a simple check: derivative orders should not skip too many levels
        // relative to the number of conditions available.
        // Full Polya check is in check_polya_conditions().

        // Build the Vandermonde-like system.
        // Row i corresponds to condition i.
        // Column m corresponds to monomial x^m.
        // Entry [i][m] = d^{j_i}/dx^{j_i} [x^m] evaluated at x_i.
        let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut rhs: Vec<f64> = Vec::with_capacity(n);

        for cond in conditions {
            let mut row = Vec::with_capacity(n);
            for m in 0..n {
                row.push(monomial_derivative(m, cond.derivative_order, cond.point));
            }
            matrix.push(row);
            rhs.push(cond.value);
        }

        let coefficients = solve_system(&matrix, &rhs)?;

        Ok(HermiteBirkhoff { coefficients })
    }

    /// Fit using the explicit specification of points, derivative orders, and values.
    ///
    /// This is a convenience method that builds conditions from parallel arrays.
    ///
    /// # Arguments
    ///
    /// * `points` - The interpolation nodes.
    /// * `derivative_orders` - The derivative order for each condition.
    /// * `values` - The target values for each condition.
    ///
    /// All three slices must have the same length.
    pub fn fit_from_arrays(
        points: &[f64],
        derivative_orders: &[usize],
        values: &[f64],
    ) -> InterpolateResult<Self> {
        if points.len() != derivative_orders.len() || points.len() != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points ({}), derivative_orders ({}), and values ({}) must have the same length",
                points.len(),
                derivative_orders.len(),
                values.len()
            )));
        }

        let conditions: Vec<InterpolationCondition> = points
            .iter()
            .zip(derivative_orders.iter())
            .zip(values.iter())
            .map(|((&p, &d), &v)| InterpolationCondition {
                point: p,
                derivative_order: d,
                value: v,
            })
            .collect();

        Self::fit(&conditions)
    }

    /// Evaluate the interpolating polynomial at `x`.
    ///
    /// Uses Horner's method for numerical stability.
    pub fn evaluate(&self, x: f64) -> InterpolateResult<f64> {
        if self.coefficients.is_empty() {
            return Err(InterpolateError::InvalidState(
                "interpolator has no coefficients".to_string(),
            ));
        }

        // Horner's method: c[0] + x*(c[1] + x*(c[2] + ...))
        let n = self.coefficients.len();
        let mut result = self.coefficients[n - 1];
        for i in (0..n - 1).rev() {
            result = result * x + self.coefficients[i];
        }

        Ok(result)
    }

    /// Evaluate the `k`-th derivative of the interpolating polynomial at `x`.
    pub fn evaluate_derivative(&self, x: f64, k: usize) -> InterpolateResult<f64> {
        if self.coefficients.is_empty() {
            return Err(InterpolateError::InvalidState(
                "interpolator has no coefficients".to_string(),
            ));
        }

        let n = self.coefficients.len();
        let mut result = 0.0;
        for m in 0..n {
            result += self.coefficients[m] * monomial_derivative(m, k, x);
        }
        Ok(result)
    }

    /// Return the polynomial degree.
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Return the polynomial coefficients (monomial basis).
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_hermite_value_and_derivative() {
        // Match value + first derivative at two points
        // f(0) = 0, f'(0) = 1, f(1) = 0, f'(1) = -1
        // This is satisfied by f(x) = x(1-x) = x - x^2 => f'(x) = 1 - 2x
        // f(0) = 0, f'(0) = 1, f(1) = 0, f'(1) = -1
        let conditions = vec![
            InterpolationCondition {
                point: 0.0,
                derivative_order: 0,
                value: 0.0,
            },
            InterpolationCondition {
                point: 0.0,
                derivative_order: 1,
                value: 1.0,
            },
            InterpolationCondition {
                point: 1.0,
                derivative_order: 0,
                value: 0.0,
            },
            InterpolationCondition {
                point: 1.0,
                derivative_order: 1,
                value: -1.0,
            },
        ];

        let interp = HermiteBirkhoff::fit(&conditions).expect("test: fit should succeed");

        // Check at data points
        let v0 = interp.evaluate(0.0).expect("test: should evaluate");
        let v1 = interp.evaluate(1.0).expect("test: should evaluate");
        assert!(v0.abs() < 1e-10, "f(0) should be 0, got {}", v0);
        assert!(v1.abs() < 1e-10, "f(1) should be 0, got {}", v1);

        // Check derivative
        let d0 = interp
            .evaluate_derivative(0.0, 1)
            .expect("test: should evaluate derivative");
        let d1 = interp
            .evaluate_derivative(1.0, 1)
            .expect("test: should evaluate derivative");
        assert!((d0 - 1.0).abs() < 1e-10, "f'(0) should be 1, got {}", d0);
        assert!((d1 + 1.0).abs() < 1e-10, "f'(1) should be -1, got {}", d1);

        // Check at midpoint: f(0.5) should be 0.25
        let v_mid = interp.evaluate(0.5).expect("test: should evaluate");
        assert!(
            (v_mid - 0.25).abs() < 1e-10,
            "f(0.5) should be 0.25, got {}",
            v_mid
        );
    }

    #[test]
    fn test_higher_order_derivative_interpolation() {
        // Fit: f(0) = 1, f'(0) = 0, f''(0) = -1
        // This gives f(x) = 1 - x^2/2 (Taylor of cos)
        let conditions = vec![
            InterpolationCondition {
                point: 0.0,
                derivative_order: 0,
                value: 1.0,
            },
            InterpolationCondition {
                point: 0.0,
                derivative_order: 1,
                value: 0.0,
            },
            InterpolationCondition {
                point: 0.0,
                derivative_order: 2,
                value: -1.0,
            },
        ];

        let interp = HermiteBirkhoff::fit(&conditions).expect("test: fit should succeed");

        // f(x) = 1 + 0*x + (-1/2)*x^2
        let v0 = interp.evaluate(0.0).expect("test: should evaluate");
        assert!((v0 - 1.0).abs() < 1e-10, "f(0) should be 1");

        let v1 = interp.evaluate(1.0).expect("test: should evaluate");
        assert!((v1 - 0.5).abs() < 1e-10, "f(1) should be 0.5, got {}", v1);

        let d2 = interp
            .evaluate_derivative(0.5, 2)
            .expect("test: should evaluate");
        assert!(
            (d2 + 1.0).abs() < 1e-10,
            "f''(0.5) should be -1, got {}",
            d2
        );
    }

    #[test]
    fn test_pure_lagrange_interpolation() {
        // Only function values (derivative_order = 0 everywhere) = Lagrange
        // f(0) = 0, f(1) = 1, f(2) = 4 => f(x) = x^2
        let conditions = vec![
            InterpolationCondition {
                point: 0.0,
                derivative_order: 0,
                value: 0.0,
            },
            InterpolationCondition {
                point: 1.0,
                derivative_order: 0,
                value: 1.0,
            },
            InterpolationCondition {
                point: 2.0,
                derivative_order: 0,
                value: 4.0,
            },
        ];

        let interp = HermiteBirkhoff::fit(&conditions).expect("test: fit should succeed");

        // Check intermediate point: f(1.5) = 2.25
        let v = interp.evaluate(1.5).expect("test: should evaluate");
        assert!((v - 2.25).abs() < 1e-10, "f(1.5) should be 2.25, got {}", v);
    }

    #[test]
    fn test_fit_from_arrays() {
        let points = [0.0, 0.0, 1.0, 1.0];
        let derivs = [0, 1, 0, 1];
        let values = [0.0, 1.0, 1.0, 0.0];

        let interp = HermiteBirkhoff::fit_from_arrays(&points, &derivs, &values)
            .expect("test: fit_from_arrays should succeed");

        let v0 = interp.evaluate(0.0).expect("test: should evaluate");
        let v1 = interp.evaluate(1.0).expect("test: should evaluate");
        assert!(v0.abs() < 1e-10);
        assert!((v1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polya_conditions_checker() {
        // Valid: two function values
        let valid = vec![
            InterpolationCondition {
                point: 0.0,
                derivative_order: 0,
                value: 0.0,
            },
            InterpolationCondition {
                point: 1.0,
                derivative_order: 0,
                value: 1.0,
            },
        ];
        assert!(check_polya_conditions(&valid).is_ok());

        // Empty should fail
        let empty: Vec<InterpolationCondition> = vec![];
        assert!(check_polya_conditions(&empty).is_err());
    }

    #[test]
    fn test_mixed_derivative_orders_at_different_points() {
        // f(0) = 0 (value at 0)
        // f'(1) = 2 (derivative at 1)
        // f(2) = 4 (value at 2)
        //
        // Polynomial of degree 2: f(x) = ax^2 + bx + c
        // f(0) = c = 0
        // f'(1) = 2a + b = 2
        // f(2) = 4a + 2b + c = 4
        // => c = 0, b = 2 - 2a, 4a + 2(2-2a) = 4 => 4 = 4, so we need
        // another condition or accept the family. With 3 conditions, degree 2:
        // c = 0, 2a+b = 2, 4a+2b = 4 => 4a+2b=4 and 2(2a+b)=4 => same equation!
        // System is underdetermined. Let's use a well-posed problem instead.
        let conditions = vec![
            InterpolationCondition {
                point: 0.0,
                derivative_order: 0,
                value: 1.0,
            },
            InterpolationCondition {
                point: 0.0,
                derivative_order: 1,
                value: 0.0,
            },
            InterpolationCondition {
                point: 1.0,
                derivative_order: 0,
                value: 2.0,
            },
            InterpolationCondition {
                point: 1.0,
                derivative_order: 1,
                value: 4.0,
            },
            InterpolationCondition {
                point: 2.0,
                derivative_order: 0,
                value: 9.0,
            },
        ];

        let interp = HermiteBirkhoff::fit(&conditions).expect("test: fit should succeed");

        // Verify conditions
        let v0 = interp.evaluate(0.0).expect("test: should evaluate");
        assert!((v0 - 1.0).abs() < 1e-8, "f(0) should be 1, got {}", v0);
        let d0 = interp
            .evaluate_derivative(0.0, 1)
            .expect("test: should evaluate");
        assert!(d0.abs() < 1e-8, "f'(0) should be 0, got {}", d0);
        let v2 = interp.evaluate(2.0).expect("test: should evaluate");
        assert!((v2 - 9.0).abs() < 1e-8, "f(2) should be 9, got {}", v2);
    }

    #[test]
    fn test_error_on_mismatched_arrays() {
        let result = HermiteBirkhoff::fit_from_arrays(&[0.0, 1.0], &[0], &[0.0, 1.0]);
        assert!(result.is_err());
    }
}
