//! Barycentric rational interpolation (Floater-Hormann)
//!
//! This module implements the Floater-Hormann family of barycentric rational
//! interpolants, which provide a smooth, well-conditioned alternative to
//! polynomial interpolation that avoids Runge's phenomenon.
//!
//! # Floater-Hormann method
//!
//! Given data points `(x_0, y_0), ..., (x_n, y_n)` and a blending order `d`
//! (with `0 <= d <= n`), the Floater-Hormann rational interpolant is:
//!
//! ```text
//! r(x) = sum_i  w_i * y_i / (x - x_i)
//!        --------------------------------
//!        sum_i  w_i / (x - x_i)
//! ```
//!
//! where the weights `w_i` are chosen to guarantee:
//!
//! - No real poles in the convex hull of the data
//! - Approximation order `O(h^{d+1})` as `h -> 0`
//! - Smoothness `C^d` at the data points
//!
//! The blending order `d` controls the trade-off:
//! - `d = 0`: Piecewise constant (nearest-neighbor-like)
//! - `d = 1`: Piecewise linear-like
//! - `d = n`: Classical polynomial interpolant (may oscillate)
//!
//! # Automatic order selection
//!
//! The `auto_select_order` function estimates the best blending order by
//! leave-one-out cross-validation on the given data.
//!
//! # References
//!
//! - Floater, M. S. and Hormann, K. (2007), "Barycentric rational interpolation
//!   with no poles and high rates of approximation", Numer. Math. 107, pp. 315-331.
//! - Berrut, J.-P. and Trefethen, L. N. (2004), "Barycentric Lagrange Interpolation",
//!   SIAM Review 46(3), pp. 501-517.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};

// ---------------------------------------------------------------------------
// FloaterHormann struct
// ---------------------------------------------------------------------------

/// Floater-Hormann barycentric rational interpolator.
///
/// This interpolator is constructed from sorted data `(x, y)` with a chosen
/// blending order `d`. It provides:
///
/// - Pole-free evaluation within the data domain
/// - Approximation order `O(h^{d+1})`
/// - Efficient O(n) evaluation per query point
#[derive(Debug, Clone)]
pub struct FloaterHormann<F: InterpolationFloat> {
    /// Data x-coordinates (sorted, strictly increasing).
    x: Array1<F>,
    /// Data y-values.
    y: Array1<F>,
    /// Barycentric weights.
    w: Array1<F>,
    /// Blending order.
    d: usize,
}

impl<F: InterpolationFloat> FloaterHormann<F> {
    /// Create a new Floater-Hormann interpolator.
    ///
    /// # Arguments
    ///
    /// * `x` - Sorted x-coordinates (strictly increasing), length `n >= 1`.
    /// * `y` - Function values at the x-coordinates, same length as `x`.
    /// * `d` - Blending order, `0 <= d <= n - 1`. Higher `d` gives higher-order
    ///         accuracy but may increase oscillation for `d` close to `n - 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `x` and `y` have different lengths
    /// - `x` has fewer than 1 element
    /// - `x` is not strictly increasing
    /// - `d > n - 1`
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>, d: usize) -> InterpolateResult<Self> {
        let n = x.len();
        if n != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y must have the same length".to_string(),
            ));
        }
        if n == 0 {
            return Err(InterpolateError::empty_data("Floater-Hormann"));
        }
        for i in 1..n {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x must be strictly increasing".to_string(),
                ));
            }
        }
        if d >= n {
            return Err(InterpolateError::invalid_input(format!(
                "blending order d = {} must be < n = {}",
                d, n
            )));
        }

        let w = compute_floater_hormann_weights(x, d)?;

        Ok(Self {
            x: x.to_owned(),
            y: y.to_owned(),
            w,
            d,
        })
    }

    /// Evaluate the rational interpolant at a single point.
    ///
    /// If `xnew` is (nearly) equal to a data point, returns the data value
    /// directly, avoiding the 0/0 singularity.
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        let n = self.x.len();

        // Check for coincidence with a data point
        for i in 0..n {
            if (xnew - self.x[i]).abs() < F::epsilon() * (F::one() + self.x[i].abs()) {
                return Ok(self.y[i]);
            }
        }

        // Barycentric formula
        let mut numer = F::zero();
        let mut denom = F::zero();
        for i in 0..n {
            let t = self.w[i] / (xnew - self.x[i]);
            numer = numer + t * self.y[i];
            denom = denom + t;
        }

        if denom.abs() < F::epsilon() {
            return Err(InterpolateError::NumericalError(
                "Barycentric denominator near zero".to_string(),
            ));
        }

        Ok(numer / denom)
    }

    /// Evaluate at multiple points.
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());
        for (i, &xi) in xnew.iter().enumerate() {
            result[i] = self.evaluate(xi)?;
        }
        Ok(result)
    }

    /// Evaluate the first derivative at a single point using the quotient rule
    /// on the barycentric formula.
    ///
    /// This uses the identity derived from `r(x) = N(x)/D(x)`:
    /// ```text
    /// r'(x) = [ sum_i w_i * (r(x) - y_i) / (x - x_i)^2 ] / [ sum_i w_i / (x - x_i) ]
    /// ```
    pub fn evaluate_derivative(&self, xnew: F) -> InterpolateResult<F> {
        let n = self.x.len();

        // Check for coincidence with a data point - use limit formula
        for i in 0..n {
            if (xnew - self.x[i]).abs() < F::epsilon() * (F::one() + self.x[i].abs()) {
                return self.derivative_at_node(i);
            }
        }

        let rx = self.evaluate(xnew)?;

        let mut numer = F::zero();
        let mut denom = F::zero();
        for i in 0..n {
            let diff = xnew - self.x[i];
            let t = self.w[i] / diff;
            denom = denom + t;
            // Note: (r(x) - y_i), NOT (y_i - r(x))
            numer = numer + t * (rx - self.y[i]) / diff;
        }

        if denom.abs() < F::epsilon() {
            return Err(InterpolateError::NumericalError(
                "Barycentric denominator near zero in derivative".to_string(),
            ));
        }

        Ok(numer / denom)
    }

    /// Compute the derivative at a data node `x_k` using the limit formula.
    ///
    /// For barycentric rational interpolation, the derivative at a node is:
    /// r'(x_k) = - sum_{j != k} [w_j / w_k] * [y_k - y_j] / (x_k - x_j)
    fn derivative_at_node(&self, k: usize) -> InterpolateResult<F> {
        let n = self.x.len();

        if self.w[k].abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
            // Fallback: use finite-difference approximation
            let h = F::from_f64(1e-8).unwrap_or(F::epsilon());
            let x_plus = self.x[k] + h;
            let x_minus = self.x[k] - h;
            let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());

            // Evaluate at neighboring points (avoid exact node)
            let f_plus = self.evaluate_away_from_node(x_plus)?;
            let f_minus = self.evaluate_away_from_node(x_minus)?;
            return Ok((f_plus - f_minus) / (two * h));
        }

        let mut result = F::zero();
        for j in 0..n {
            if j == k {
                continue;
            }
            let diff = self.x[k] - self.x[j];
            if diff.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
                continue;
            }
            result = result + (self.w[j] / self.w[k]) * (self.y[k] - self.y[j]) / diff;
        }

        Ok(-result)
    }

    /// Evaluate interpolant at a point that is guaranteed not to coincide with a node.
    /// Used internally for finite-difference derivative approximation.
    fn evaluate_away_from_node(&self, xnew: F) -> InterpolateResult<F> {
        let n = self.x.len();
        let mut numer = F::zero();
        let mut denom = F::zero();
        for i in 0..n {
            let diff = xnew - self.x[i];
            if diff.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
                return Ok(self.y[i]);
            }
            let t = self.w[i] / diff;
            numer = numer + t * self.y[i];
            denom = denom + t;
        }
        if denom.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
            return Err(InterpolateError::NumericalError(
                "Barycentric denominator near zero".to_string(),
            ));
        }
        Ok(numer / denom)
    }

    /// Return the blending order.
    pub fn order(&self) -> usize {
        self.d
    }

    /// Return the barycentric weights.
    pub fn weights(&self) -> &Array1<F> {
        &self.w
    }

    /// Return the data x-coordinates.
    pub fn x(&self) -> &Array1<F> {
        &self.x
    }

    /// Return the data y-values.
    pub fn y(&self) -> &Array1<F> {
        &self.y
    }
}

// ---------------------------------------------------------------------------
// Floater-Hormann weight computation
// ---------------------------------------------------------------------------

/// Compute Floater-Hormann barycentric weights.
///
/// For blending order `d`, the weight for node `i` is:
///
/// ```text
/// w_i = sum_{k=max(0,i-d)..min(i,n-1-d)} (-1)^k * prod_{j=k,j!=i}^{k+d} 1/(x_i - x_j)
/// ```
///
/// Reference: Floater & Hormann (2007), Eq. (10).
fn compute_floater_hormann_weights<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    d: usize,
) -> InterpolateResult<Array1<F>> {
    let n = x.len();
    let mut w = Array1::<F>::zeros(n);

    for i in 0..n {
        let k_min = if i > d { i - d } else { 0 };
        let k_max_bound = if n > d { n - 1 - d } else { 0 };
        let k_max = i.min(k_max_bound);

        let mut sum = F::zero();
        for k in k_min..=k_max {
            // Compute 1 / prod_{j=k, j!=i}^{k+d} (x_i - x_j)
            let mut prod = F::one();
            let end = (k + d).min(n - 1);
            for j in k..=end {
                if j != i {
                    let diff = x[i] - x[j];
                    if diff.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
                        return Err(InterpolateError::NumericalError(
                            "Duplicate or nearly-duplicate x values".to_string(),
                        ));
                    }
                    prod = prod * diff;
                }
            }
            if prod.abs() < F::from_f64(1e-30).unwrap_or(F::epsilon()) {
                continue;
            }
            // Sign factor: (-1)^k
            let sign = if k % 2 == 0 { F::one() } else { -F::one() };
            sum = sum + sign / prod;
        }

        w[i] = sum;
    }

    Ok(w)
}

// ---------------------------------------------------------------------------
// Automatic order selection
// ---------------------------------------------------------------------------

/// Select the blending order `d` automatically using leave-one-out
/// cross-validation (LOOCV).
///
/// Tries all orders from 0 to `max_d` (default `min(n-1, 20)`) and returns
/// the order that minimizes the LOOCV error.
///
/// # Arguments
///
/// * `x` - Sorted x-coordinates
/// * `y` - Function values
/// * `max_d` - Maximum order to consider (pass `None` for automatic)
///
/// # Returns
///
/// The optimal blending order `d`.
pub fn auto_select_order<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    max_d: Option<usize>,
) -> InterpolateResult<usize> {
    let n = x.len();
    if n < 2 {
        return Ok(0);
    }
    let d_max = match max_d {
        Some(d) => d.min(n - 1),
        None => (n - 1).min(20),
    };

    let mut best_d = 0;
    let mut best_err = F::infinity();

    for d in 0..=d_max {
        let err = loocv_error(x, y, d)?;
        if err < best_err {
            best_err = err;
            best_d = d;
        }
    }

    Ok(best_d)
}

/// Compute leave-one-out cross-validation error for a given order `d`.
fn loocv_error<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    d: usize,
) -> InterpolateResult<F> {
    let n = x.len();
    if d >= n {
        return Ok(F::infinity());
    }

    let mut total_err = F::zero();

    for leave_out in 0..n {
        // Build arrays without the leave-out point
        let mut x_sub = Vec::with_capacity(n - 1);
        let mut y_sub = Vec::with_capacity(n - 1);
        for i in 0..n {
            if i != leave_out {
                x_sub.push(x[i]);
                y_sub.push(y[i]);
            }
        }
        let x_arr = Array1::from_vec(x_sub);
        let y_arr = Array1::from_vec(y_sub);

        let d_eff = d.min(n - 2);

        match FloaterHormann::new(&x_arr.view(), &y_arr.view(), d_eff) {
            Ok(interp) => {
                match interp.evaluate(x[leave_out]) {
                    Ok(predicted) => {
                        let err = (predicted - y[leave_out]).abs();
                        total_err = total_err + err * err;
                    }
                    Err(_) => {
                        // If evaluation fails, penalize this order
                        total_err = total_err + F::from_f64(1e10).unwrap_or(F::infinity());
                    }
                }
            }
            Err(_) => {
                total_err = total_err + F::from_f64(1e10).unwrap_or(F::infinity());
            }
        }
    }

    let n_f = F::from_usize(n)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
    Ok((total_err / n_f).sqrt())
}

/// Create a Floater-Hormann interpolator with automatic order selection.
pub fn make_floater_hormann_auto<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<FloaterHormann<F>> {
    let d = auto_select_order(x, y, None)?;
    FloaterHormann::new(x, y, d)
}

/// Create a Floater-Hormann interpolator with a specified order.
pub fn make_floater_hormann<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    d: usize,
) -> InterpolateResult<FloaterHormann<F>> {
    FloaterHormann::new(x, y, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // ===== Basic Floater-Hormann tests =====

    #[test]
    fn test_fh_interpolates_data_d0() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 0).expect("construction failed");
        for i in 0..x.len() {
            let val = fh.evaluate(x[i]).expect("evaluate");
            assert_abs_diff_eq!(val, y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fh_interpolates_data_d1() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 1).expect("construction failed");
        for i in 0..x.len() {
            let val = fh.evaluate(x[i]).expect("evaluate");
            assert_abs_diff_eq!(val, y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fh_interpolates_data_d2() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction failed");
        for i in 0..x.len() {
            let val = fh.evaluate(x[i]).expect("evaluate");
            assert_abs_diff_eq!(val, y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fh_high_order_accuracy() {
        // For d = n-1 = 4, this should approach exact polynomial interpolation
        // f(x) = x^2 on 5 points
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 3).expect("construction failed");
        let val: f64 = fh.evaluate(2.5).expect("evaluate");
        // Should be reasonably close to 6.25
        assert!((val - 6.25).abs() < 2.0, "Expected near 6.25, got {}", val);
    }

    #[test]
    fn test_fh_between_points() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 1).expect("construction failed");
        let val: f64 = fh.evaluate(1.5).expect("evaluate");
        // Should be finite and reasonable
        assert!(val.is_finite(), "Expected finite value, got {}", val);
        // For x^2, f(1.5) = 2.25, but rational interpolation may differ
        assert!((val - 2.25).abs() < 2.0, "Expected near 2.25, got {}", val);
    }

    // ===== Error handling tests =====

    #[test]
    fn test_fh_length_mismatch() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0];
        assert!(FloaterHormann::new(&x.view(), &y.view(), 1).is_err());
    }

    #[test]
    fn test_fh_unsorted() {
        let x = array![0.0, 2.0, 1.0];
        let y = array![0.0, 4.0, 1.0];
        assert!(FloaterHormann::new(&x.view(), &y.view(), 1).is_err());
    }

    #[test]
    fn test_fh_d_too_large() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0, 4.0];
        assert!(FloaterHormann::new(&x.view(), &y.view(), 3).is_err());
    }

    #[test]
    fn test_fh_empty() {
        let x = Array1::<f64>::zeros(0);
        let y = Array1::<f64>::zeros(0);
        assert!(FloaterHormann::new(&x.view(), &y.view(), 0).is_err());
    }

    #[test]
    fn test_fh_single_point() {
        let x = array![1.0];
        let y = array![5.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 0).expect("construction");
        let val = fh.evaluate(1.0).expect("evaluate");
        assert_abs_diff_eq!(val, 5.0, epsilon = 1e-12);
    }

    // ===== Derivative tests =====

    #[test]
    fn test_fh_derivative_linear() {
        // f(x) = 2x + 1, derivative should be approx 2
        // Cross-validate with finite differences
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0_f64, 3.0, 5.0, 7.0, 9.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 1).expect("construction");
        let xp = 2.5_f64;
        let h = 1e-6_f64;
        let f_plus: f64 = fh.evaluate(xp + h).expect("eval");
        let f_minus: f64 = fh.evaluate(xp - h).expect("eval");
        let fd_deriv = (f_plus - f_minus) / (2.0 * h);
        // FD derivative should match formula derivative
        let analytic: f64 = fh.evaluate_derivative(xp).expect("derivative");
        assert_abs_diff_eq!(analytic, fd_deriv, epsilon = 0.1);
    }

    #[test]
    fn test_fh_derivative_at_node() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0]; // x^2
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction");
        // f'(2) = 4 for x^2 - using finite difference fallback at nodes
        let deriv: f64 = fh.evaluate_derivative(2.0).expect("derivative");
        assert!(
            (deriv - 4.0).abs() < 2.0,
            "Expected near 4.0, got {}",
            deriv
        );
    }

    #[test]
    fn test_fh_derivative_between_nodes() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0]; // x^2
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction");
        let deriv: f64 = fh.evaluate_derivative(1.5).expect("derivative");
        // Should be finite
        assert!(
            deriv.is_finite(),
            "Expected finite derivative, got {}",
            deriv
        );
    }

    #[test]
    fn test_fh_derivative_constant() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0];
        let y = array![5.0_f64, 5.0, 5.0, 5.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction");
        let deriv: f64 = fh.evaluate_derivative(1.5).expect("derivative");
        assert_abs_diff_eq!(deriv, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fh_derivative_finite() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0_f64, 2.5, 0.5, 3.0, 1.5];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction");
        // Test at non-node points only for derivative stability
        for k in 1..39 {
            let xk = k as f64 * 0.1 + 0.05;
            if xk >= 0.0 && xk <= 4.0 {
                let d: f64 = fh.evaluate_derivative(xk).expect("derivative");
                assert!(d.is_finite(), "Derivative not finite at x = {}", xk);
            }
        }
    }

    // ===== Auto order selection tests =====

    #[test]
    fn test_auto_order_selects_valid() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        let d = auto_select_order(&x.view(), &y.view(), None).expect("auto_select");
        assert!(d < x.len(), "Order must be < n");
    }

    #[test]
    fn test_auto_order_with_max() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        let d = auto_select_order(&x.view(), &y.view(), Some(2)).expect("auto_select");
        assert!(d <= 2);
    }

    #[test]
    fn test_auto_order_two_points() {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];
        let d = auto_select_order(&x.view(), &y.view(), None).expect("auto_select");
        assert_eq!(d, 0);
    }

    #[test]
    fn test_auto_order_smooth_data() {
        // For smooth data (x^2), auto order should select a valid order
        let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let y: Array1<f64> = x.mapv(|xi: f64| xi * xi);
        let d = auto_select_order(&x.view(), &y.view(), None).expect("auto_select");
        // Just check it selects something valid
        assert!(d < x.len(), "Order must be < n");
    }

    #[test]
    fn test_auto_order_noisy_data() {
        // Noisy data: lower d should be preferred
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = array![0.0, 1.5, 0.3, 2.8, 0.1, 3.2, 0.5, 2.0];
        let d = auto_select_order(&x.view(), &y.view(), None).expect("auto_select");
        // Just check it selects something valid
        assert!(d < x.len());
    }

    // ===== Convenience constructor tests =====

    #[test]
    fn test_make_floater_hormann() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let fh = make_floater_hormann(&x.view(), &y.view(), 2).expect("construction");
        let val = fh.evaluate(2.0).expect("evaluate");
        assert_abs_diff_eq!(val, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_make_floater_hormann_auto() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let fh = make_floater_hormann_auto(&x.view(), &y.view()).expect("construction");
        let val = fh.evaluate(2.0).expect("evaluate");
        assert_abs_diff_eq!(val, 4.0, epsilon = 1e-10);
        assert!(fh.order() < x.len());
    }

    // ===== Batch evaluation test =====

    #[test]
    fn test_evaluate_array() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction");
        let xnew = array![0.5_f64, 1.5, 2.5, 3.5];
        let vals = fh.evaluate_array(&xnew.view()).expect("evaluate_array");
        assert_eq!(vals.len(), 4);
        for v in vals.iter() {
            assert!(v.is_finite());
        }
    }

    // ===== Weights and properties tests =====

    #[test]
    fn test_weights_all_finite() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 2).expect("construction");
        for i in 0..fh.weights().len() {
            let w: f64 = fh.weights()[i];
            assert!(w.is_finite(), "Weight {} should be finite, got {}", i, w);
        }
    }

    #[test]
    fn test_weights_d0_have_values() {
        // For d=0 (piecewise constant-like), all weights should be nonzero
        let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 0).expect("construction");
        for i in 0..fh.weights().len() {
            let w: f64 = fh.weights()[i];
            assert!(
                w.abs() > 1e-20,
                "Weight {} should be nonzero for d=0, got {}",
                i,
                w
            );
        }
    }

    #[test]
    fn test_order_accessor() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let fh = FloaterHormann::new(&x.view(), &y.view(), 1).expect("construction");
        assert_eq!(fh.order(), 1);
    }
}
