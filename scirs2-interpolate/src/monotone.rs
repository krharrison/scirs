//! Monotone interpolation: Fritsch-Carlson and Steffen methods
//!
//! This module provides standalone monotone cubic Hermite interpolation using two
//! well-known algorithms that guarantee shape-preserving (monotonicity-preserving)
//! interpolation:
//!
//! - **Fritsch-Carlson (1980)**: The original PCHIP-like monotone method that modifies
//!   initial slope estimates to satisfy the Fritsch-Carlson monotonicity condition.
//!   Reference: Fritsch, F. N. and Carlson, R. E. (1980), "Monotone Piecewise Cubic
//!   Interpolation", SIAM J. Numer. Anal. 17(2), pp. 238--246.
//!
//! - **Steffen (1990)**: A simpler method that guarantees monotonicity and avoids
//!   overshoot by construction. Each derivative is bounded by the local secant slopes.
//!   Reference: Steffen, M. (1990), "A simple method for monotonic interpolation in
//!   one dimension", Astron. Astrophys. 239, pp. 443--450.
//!
//! Both methods produce C1 (continuous first derivative) piecewise cubic Hermite
//! interpolants. The Steffen method is slightly more conservative (smaller derivatives),
//! while Fritsch-Carlson tends to follow the data more closely.
//!
//! # Key differences from `interp1d::monotonic`
//!
//! This module provides:
//! - Derivative evaluation (`evaluate_derivative`)
//! - Integration (`integrate`)
//! - Separate, independently usable types rather than being mode-selected inside a
//!   single wrapper type

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};

// ---------------------------------------------------------------------------
// Common helper for cubic Hermite evaluation
// ---------------------------------------------------------------------------

/// Evaluate a cubic Hermite basis interpolant at a normalized parameter t in [0,1].
///
/// Given endpoint values `y0`, `y1` and endpoint derivatives `d0`, `d1`,
/// and the interval width `h`, evaluate
///
/// ```text
/// p(t) = h00(t)*y0 + h10(t)*h*d0 + h01(t)*y1 + h11(t)*h*d1
/// ```
fn hermite_eval<F: InterpolationFloat>(t: F, y0: F, y1: F, d0: F, d1: F, h: F) -> F {
    let t2 = t * t;
    let t3 = t2 * t;
    let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
    let three = F::from_f64(3.0).unwrap_or(two + F::one());

    let h00 = two * t3 - three * t2 + F::one();
    let h10 = t3 - two * t2 + t;
    let h01 = -two * t3 + three * t2;
    let h11 = t3 - t2;

    h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
}

/// Evaluate the derivative of a cubic Hermite basis interpolant.
///
/// Returns dp/dx where p is evaluated via `hermite_eval`.
fn hermite_deriv<F: InterpolationFloat>(
    t: F,
    y0: F,
    y1: F,
    d0: F,
    d1: F,
    h: F,
) -> InterpolateResult<F> {
    if h.abs() < F::epsilon() {
        return Err(InterpolateError::NumericalError(
            "Zero-width interval in Hermite derivative".to_string(),
        ));
    }
    let t2 = t * t;
    let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
    let three = F::from_f64(3.0).unwrap_or(two + F::one());
    let six = F::from_f64(6.0).unwrap_or(three + three);

    // dp/dt
    let dh00 = six * t2 - six * t;
    let dh10 = three * t2 - F::from_f64(4.0).unwrap_or(two + two) * t + F::one();
    let dh01 = -six * t2 + six * t;
    let dh11 = three * t2 - two * t;

    let dp_dt = dh00 * y0 + dh10 * h * d0 + dh01 * y1 + dh11 * h * d1;
    // dp/dx = dp/dt * dt/dx = dp/dt / h
    Ok(dp_dt / h)
}

// ===================================================================
// Fritsch-Carlson monotone interpolator
// ===================================================================

/// Fritsch-Carlson monotone piecewise cubic Hermite interpolator.
///
/// This method starts with three-point finite-difference slope estimates and then
/// adjusts them to satisfy the necessary and sufficient conditions for monotonicity
/// described in Fritsch & Carlson (1980).
#[derive(Debug, Clone)]
pub struct FritschCarlsonInterpolator<F: InterpolationFloat> {
    /// Sorted x coordinates
    x: Array1<F>,
    /// y values at the knots
    y: Array1<F>,
    /// Slopes (first derivatives) at the knots
    d: Array1<F>,
}

impl<F: InterpolationFloat> FritschCarlsonInterpolator<F> {
    /// Create a new Fritsch-Carlson monotone interpolator.
    ///
    /// # Arguments
    ///
    /// * `x` - Sorted x coordinates (strictly increasing)
    /// * `y` - Function values at x coordinates (same length as x)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `x` and `y` have different lengths
    /// - Fewer than 3 data points
    /// - `x` is not strictly increasing
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        validate_inputs(x, y, 3, "Fritsch-Carlson")?;

        let n = x.len();
        let x_arr = x.to_owned();
        let y_arr = y.to_owned();

        // Step 1: compute secant slopes delta_k = (y_{k+1} - y_k) / (x_{k+1} - x_k)
        let mut delta = Array1::<F>::zeros(n - 1);
        for k in 0..n - 1 {
            delta[k] = (y_arr[k + 1] - y_arr[k]) / (x_arr[k + 1] - x_arr[k]);
        }

        // Step 2: initial slope estimate via three-point formula
        let mut d = Array1::<F>::zeros(n);
        let two = F::from_f64(2.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;

        // Interior points: arithmetic mean of adjacent secants
        for k in 1..n - 1 {
            if delta[k - 1] * delta[k] <= F::zero() {
                d[k] = F::zero();
            } else {
                d[k] = (delta[k - 1] + delta[k]) / two;
            }
        }

        // Endpoint slopes: one-sided
        d[0] = delta[0];
        d[n - 1] = delta[n - 2];

        // Step 3: Fritsch-Carlson monotonicity adjustment
        // For each interval, if delta[k] == 0, set d[k] = d[k+1] = 0.
        // Otherwise, let alpha_k = d[k]/delta[k], beta_k = d[k+1]/delta[k].
        // If alpha_k^2 + beta_k^2 > 9, rescale d[k] and d[k+1].
        let three = F::from_f64(3.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
        let nine = F::from_f64(9.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;

        for k in 0..n - 1 {
            if delta[k].abs() < F::epsilon() {
                d[k] = F::zero();
                d[k + 1] = F::zero();
            } else {
                let alpha = d[k] / delta[k];
                let beta = d[k + 1] / delta[k];
                let phi_sq = alpha * alpha + beta * beta;
                if phi_sq > nine {
                    let tau = three / phi_sq.sqrt();
                    d[k] = tau * alpha * delta[k];
                    d[k + 1] = tau * beta * delta[k];
                }
            }
        }

        Ok(Self {
            x: x_arr,
            y: y_arr,
            d,
        })
    }

    /// Evaluate the interpolant at a single point.
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        let (idx, t, h) = self.locate(xnew)?;
        Ok(hermite_eval(
            t,
            self.y[idx],
            self.y[idx + 1],
            self.d[idx],
            self.d[idx + 1],
            h,
        ))
    }

    /// Evaluate at multiple points.
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());
        for (i, &xi) in xnew.iter().enumerate() {
            result[i] = self.evaluate(xi)?;
        }
        Ok(result)
    }

    /// Evaluate the first derivative at a single point.
    pub fn evaluate_derivative(&self, xnew: F) -> InterpolateResult<F> {
        let (idx, t, h) = self.locate(xnew)?;
        hermite_deriv(
            t,
            self.y[idx],
            self.y[idx + 1],
            self.d[idx],
            self.d[idx + 1],
            h,
        )
    }

    /// Compute the definite integral of the interpolant over `[a, b]`.
    pub fn integrate(&self, a: F, b: F) -> InterpolateResult<F> {
        if a > b {
            return Ok(-self.integrate(b, a)?);
        }
        if (a - b).abs() < F::epsilon() {
            return Ok(F::zero());
        }
        hermite_integrate(&self.x, &self.y, &self.d, a, b)
    }

    /// Return the stored slopes (derivatives at knots).
    pub fn slopes(&self) -> &Array1<F> {
        &self.d
    }

    /// Locate the segment for `xnew` and return `(index, t, h)`.
    fn locate(&self, xnew: F) -> InterpolateResult<(usize, F, F)> {
        locate_in_knots(&self.x, xnew)
    }
}

// ===================================================================
// Steffen monotone interpolator
// ===================================================================

/// Steffen monotone piecewise cubic Hermite interpolator.
///
/// Steffen's method guarantees monotonicity and prevents overshoot by
/// construction. The derivative at each interior point is the minimum of
/// the absolute values of the adjacent secant slopes (with appropriate sign),
/// bounded by a weighted average.
#[derive(Debug, Clone)]
pub struct SteffenInterpolator<F: InterpolationFloat> {
    /// Sorted x coordinates
    x: Array1<F>,
    /// y values at the knots
    y: Array1<F>,
    /// Slopes (first derivatives) at the knots
    d: Array1<F>,
}

impl<F: InterpolationFloat> SteffenInterpolator<F> {
    /// Create a new Steffen monotone interpolator.
    ///
    /// # Arguments
    ///
    /// * `x` - Sorted x coordinates (strictly increasing)
    /// * `y` - Function values at x coordinates (same length as x)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `x` and `y` have different lengths
    /// - Fewer than 3 data points
    /// - `x` is not strictly increasing
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        validate_inputs(x, y, 3, "Steffen")?;

        let n = x.len();
        let x_arr = x.to_owned();
        let y_arr = y.to_owned();

        // Secant slopes
        let mut s = Array1::<F>::zeros(n - 1);
        let mut h = Array1::<F>::zeros(n - 1);
        for k in 0..n - 1 {
            h[k] = x_arr[k + 1] - x_arr[k];
            s[k] = (y_arr[k + 1] - y_arr[k]) / h[k];
        }

        let two = F::from_f64(2.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;

        let mut d = Array1::<F>::zeros(n);

        // Interior points
        for i in 1..n - 1 {
            let p_i = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);

            // Steffen condition: |d_i| <= 2 * min(|s_{i-1}|, |s_i|)
            // and sign(d_i) must match sign of s_{i-1}) if s_{i-1}*s_i > 0
            if s[i - 1] * s[i] <= F::zero() {
                d[i] = F::zero();
            } else {
                let bound = two * F::min(s[i - 1].abs(), s[i].abs());
                d[i] = if p_i.abs() <= bound {
                    p_i
                } else {
                    bound * p_i.signum()
                };
            }
        }

        // Endpoint slopes: use first/last secant, clamped similarly
        d[0] = s[0];
        d[n - 1] = s[n - 2];

        // Optionally clamp endpoints to zero if secant changes sign
        if n >= 3 {
            // Left endpoint: check if we should reduce
            if s[0] * s[1] <= F::zero() && d[0].abs() > two * s[0].abs() {
                d[0] = two * s[0];
            }
            // Right endpoint
            if s[n - 3] * s[n - 2] <= F::zero() && d[n - 1].abs() > two * s[n - 2].abs() {
                d[n - 1] = two * s[n - 2];
            }
        }

        Ok(Self {
            x: x_arr,
            y: y_arr,
            d,
        })
    }

    /// Evaluate the interpolant at a single point.
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        let (idx, t, h) = self.locate(xnew)?;
        Ok(hermite_eval(
            t,
            self.y[idx],
            self.y[idx + 1],
            self.d[idx],
            self.d[idx + 1],
            h,
        ))
    }

    /// Evaluate at multiple points.
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());
        for (i, &xi) in xnew.iter().enumerate() {
            result[i] = self.evaluate(xi)?;
        }
        Ok(result)
    }

    /// Evaluate the first derivative at a single point.
    pub fn evaluate_derivative(&self, xnew: F) -> InterpolateResult<F> {
        let (idx, t, h) = self.locate(xnew)?;
        hermite_deriv(
            t,
            self.y[idx],
            self.y[idx + 1],
            self.d[idx],
            self.d[idx + 1],
            h,
        )
    }

    /// Compute the definite integral of the interpolant over `[a, b]`.
    pub fn integrate(&self, a: F, b: F) -> InterpolateResult<F> {
        if a > b {
            return Ok(-self.integrate(b, a)?);
        }
        if (a - b).abs() < F::epsilon() {
            return Ok(F::zero());
        }
        hermite_integrate(&self.x, &self.y, &self.d, a, b)
    }

    /// Return the stored slopes (derivatives at knots).
    pub fn slopes(&self) -> &Array1<F> {
        &self.d
    }

    /// Locate the segment for `xnew` and return `(index, t, h)`.
    fn locate(&self, xnew: F) -> InterpolateResult<(usize, F, F)> {
        locate_in_knots(&self.x, xnew)
    }
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Validate common input requirements.
fn validate_inputs<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    min_points: usize,
    method_name: &str,
) -> InterpolateResult<()> {
    if x.len() != y.len() {
        return Err(InterpolateError::invalid_input(
            "x and y arrays must have the same length".to_string(),
        ));
    }
    if x.len() < min_points {
        return Err(InterpolateError::insufficient_points(
            min_points,
            x.len(),
            method_name,
        ));
    }
    for i in 1..x.len() {
        if x[i] <= x[i - 1] {
            return Err(InterpolateError::invalid_input(
                "x values must be strictly increasing".to_string(),
            ));
        }
    }
    Ok(())
}

/// Binary-search locate a point in sorted knots.
/// Returns `(segment_index, t_normalized, h_interval_width)`.
fn locate_in_knots<F: InterpolationFloat>(
    x: &Array1<F>,
    xnew: F,
) -> InterpolateResult<(usize, F, F)> {
    let n = x.len();
    if n < 2 {
        return Err(InterpolateError::InvalidState(
            "need at least 2 knots".to_string(),
        ));
    }
    if xnew < x[0] || xnew > x[n - 1] {
        return Err(InterpolateError::OutOfBounds(format!(
            "x = {} outside [{}, {}]",
            xnew,
            x[0],
            x[n - 1]
        )));
    }
    // Clamp to last segment for the right endpoint
    if xnew == x[n - 1] {
        let idx = n - 2;
        let h = x[idx + 1] - x[idx];
        return Ok((idx, F::one(), h));
    }
    // Binary search
    let mut lo: usize = 0;
    let mut hi: usize = n - 2;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if xnew < x[mid + 1] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let h = x[lo + 1] - x[lo];
    let t = (xnew - x[lo]) / h;
    Ok((lo, t, h))
}

/// Integrate a cubic Hermite interpolant defined by knots `x`, values `y`,
/// and derivatives `d` over the interval `[a, b]` (with `a <= b`).
fn hermite_integrate<F: InterpolationFloat>(
    x: &Array1<F>,
    y: &Array1<F>,
    d: &Array1<F>,
    a: F,
    b: F,
) -> InterpolateResult<F> {
    let n = x.len();
    // Find segments containing a and b
    let (idx_a, _t_a, _h_a) = locate_in_knots(x, a)?;
    let (idx_b, _t_b, _h_b) = locate_in_knots(x, b)?;

    let mut total = F::zero();

    if idx_a == idx_b {
        // Both in the same segment
        total = integrate_hermite_segment(
            x[idx_a],
            x[idx_a + 1],
            y[idx_a],
            y[idx_a + 1],
            d[idx_a],
            d[idx_a + 1],
            a,
            b,
        )?;
    } else {
        // Partial first segment
        total = total
            + integrate_hermite_segment(
                x[idx_a],
                x[idx_a + 1],
                y[idx_a],
                y[idx_a + 1],
                d[idx_a],
                d[idx_a + 1],
                a,
                x[idx_a + 1],
            )?;
        // Full middle segments
        for k in (idx_a + 1)..idx_b {
            total = total
                + integrate_hermite_segment(
                    x[k],
                    x[k + 1],
                    y[k],
                    y[k + 1],
                    d[k],
                    d[k + 1],
                    x[k],
                    x[k + 1],
                )?;
        }
        // Partial last segment
        if idx_b < n - 1 {
            total = total
                + integrate_hermite_segment(
                    x[idx_b],
                    x[idx_b + 1],
                    y[idx_b],
                    y[idx_b + 1],
                    d[idx_b],
                    d[idx_b + 1],
                    x[idx_b],
                    b,
                )?;
        }
    }

    Ok(total)
}

/// Integrate a single Hermite cubic segment from `a` to `b` within `[x0, x1]`.
///
/// The segment polynomial is `p(t) = h00*y0 + h10*h*d0 + h01*y1 + h11*h*d1`
/// where `t = (x - x0) / h`.
///
/// The integral is computed analytically in terms of `t`.
fn integrate_hermite_segment<F: InterpolationFloat>(
    x0: F,
    x1: F,
    y0: F,
    y1: F,
    d0: F,
    d1: F,
    a: F,
    b: F,
) -> InterpolateResult<F> {
    let h = x1 - x0;
    if h.abs() < F::epsilon() {
        return Ok(F::zero());
    }
    let t_a = (a - x0) / h;
    let t_b = (b - x0) / h;

    // Antiderivative of p(t) with respect to t (then multiply by h for dx):
    //
    // P(t) = h00_int(t)*y0 + h10_int(t)*h*d0 + h01_int(t)*y1 + h11_int(t)*h*d1
    //
    // where h00_int(t) = t^4/2 - t^3 + t  (integral of 2t^3 - 3t^2 + 1)
    //       h10_int(t) = t^4/4 - 2t^3/3 + t^2/2
    //       h01_int(t) = -t^4/2 + t^3
    //       h11_int(t) = t^4/4 - t^3/3

    let antideriv = |t: F| -> InterpolateResult<F> {
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;

        let half = F::from_f64(0.5)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
        let quarter = F::from_f64(0.25)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
        let two_thirds = F::from_f64(2.0 / 3.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
        let one_third = F::from_f64(1.0 / 3.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;

        let h00_int = half * t4 - t3 + t;
        let h10_int = quarter * t4 - two_thirds * t3 + half * t2;
        let h01_int = -half * t4 + t3;
        let h11_int = quarter * t4 - one_third * t3;

        Ok(h00_int * y0 + h10_int * h * d0 + h01_int * y1 + h11_int * h * d1)
    };

    let upper = antideriv(t_b)?;
    let lower = antideriv(t_a)?;
    // Multiply by h because dx = h * dt
    Ok((upper - lower) * h)
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a Fritsch-Carlson monotone interpolator.
pub fn make_fritsch_carlson<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<FritschCarlsonInterpolator<F>> {
    FritschCarlsonInterpolator::new(x, y)
}

/// Create a Steffen monotone interpolator.
pub fn make_steffen<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<SteffenInterpolator<F>> {
    SteffenInterpolator::new(x, y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // ===== Fritsch-Carlson tests =====

    #[test]
    fn test_fc_interpolates_data_points() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        for i in 0..x.len() {
            let val = fc.evaluate(x[i]).expect("evaluate failed");
            assert_abs_diff_eq!(val, y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fc_monotone_increasing() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 3.0, 6.0, 10.0, 15.0];
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        // Check monotonicity at fine grid
        let mut prev = fc.evaluate(0.0).expect("eval");
        for k in 1..50 {
            let xk = k as f64 * 0.1;
            let curr = fc.evaluate(xk).expect("eval");
            assert!(
                curr >= prev - 1e-10,
                "Monotonicity violated at x = {}: {} < {}",
                xk,
                curr,
                prev
            );
            prev = curr;
        }
    }

    #[test]
    fn test_fc_monotone_decreasing() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![10.0, 6.0, 3.0, 1.0];
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let mut prev = fc.evaluate(0.0).expect("eval");
        for k in 1..30 {
            let xk = k as f64 * 0.1;
            let curr = fc.evaluate(xk).expect("eval");
            assert!(curr <= prev + 1e-10, "Monotonicity violated at x = {}", xk);
            prev = curr;
        }
    }

    #[test]
    fn test_fc_derivative_sign_matches_slope() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 2.0, 5.0, 9.0, 14.0]; // increasing
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        for k in 0..40 {
            let xk = k as f64 * 0.1;
            let dv = fc.evaluate_derivative(xk).expect("deriv");
            assert!(
                dv >= -1e-10,
                "Derivative should be >= 0 for increasing data at x = {}, got {}",
                xk,
                dv
            );
        }
    }

    #[test]
    fn test_fc_integration_positive() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 4.0, 8.0];
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let integral = fc.integrate(0.0, 3.0).expect("integrate");
        // All values positive, so integral must be positive
        assert!(
            integral > 0.0,
            "Integral must be positive for positive function"
        );
    }

    #[test]
    fn test_fc_integration_reversed_bounds() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let i1 = fc.integrate(0.0, 3.0).expect("int");
        let i2 = fc.integrate(3.0, 0.0).expect("int");
        assert_abs_diff_eq!(i1, -i2, epsilon = 1e-12);
    }

    #[test]
    fn test_fc_constant_data() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![5.0, 5.0, 5.0, 5.0];
        let fc =
            FritschCarlsonInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let val = fc.evaluate(1.5).expect("eval");
        assert_abs_diff_eq!(val, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fc_error_unsorted() {
        let x = array![0.0, 2.0, 1.0, 3.0];
        let y = array![0.0, 1.0, 2.0, 3.0];
        assert!(FritschCarlsonInterpolator::new(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn test_fc_error_length_mismatch() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0];
        assert!(FritschCarlsonInterpolator::new(&x.view(), &y.view()).is_err());
    }

    // ===== Steffen tests =====

    #[test]
    fn test_steffen_interpolates_data_points() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        for i in 0..x.len() {
            let val = st.evaluate(x[i]).expect("evaluate failed");
            assert_abs_diff_eq!(val, y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_steffen_monotone_increasing() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 0.5, 2.0, 5.0, 10.0, 17.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let mut prev = st.evaluate(0.0).expect("eval");
        for k in 1..50 {
            let xk = k as f64 * 0.1;
            let curr = st.evaluate(xk).expect("eval");
            assert!(
                curr >= prev - 1e-10,
                "Steffen monotonicity violated at x = {}: {} < {}",
                xk,
                curr,
                prev
            );
            prev = curr;
        }
    }

    #[test]
    fn test_steffen_no_overshoot() {
        // Data with a flat section: the interpolant should not overshoot
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 0.0, 1.0, 1.0, 1.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        for k in 0..40 {
            let xk = k as f64 * 0.1;
            let val = st.evaluate(xk).expect("eval");
            assert!(
                val >= -1e-10 && val <= 1.0 + 1e-10,
                "Steffen overshoot at x = {}: {} outside [0, 1]",
                xk,
                val
            );
        }
    }

    #[test]
    fn test_steffen_derivative_bounded() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 3.0, 6.0, 10.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        for k in 0..40 {
            let xk = k as f64 * 0.1;
            let dv = st.evaluate_derivative(xk).expect("deriv");
            // For increasing data, derivative should be non-negative
            assert!(
                dv >= -1e-10,
                "Steffen derivative should be >= 0 at x = {}, got {}",
                xk,
                dv
            );
        }
    }

    #[test]
    fn test_steffen_integration_constant() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![3.0, 3.0, 3.0, 3.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let integral = st.integrate(0.0, 3.0).expect("integrate");
        assert_abs_diff_eq!(integral, 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_steffen_integration_partial() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![2.0, 2.0, 2.0, 2.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        let integral = st.integrate(0.5, 2.5).expect("integrate");
        assert_abs_diff_eq!(integral, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_steffen_error_too_few_points() {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];
        assert!(SteffenInterpolator::new(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn test_steffen_out_of_bounds() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 2.0, 3.0];
        let st = SteffenInterpolator::new(&x.view(), &y.view()).expect("construction failed");
        assert!(st.evaluate(-0.1).is_err());
        assert!(st.evaluate(3.1).is_err());
    }

    // ===== Convenience constructor tests =====

    #[test]
    fn test_make_fritsch_carlson() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0];
        let fc = make_fritsch_carlson(&x.view(), &y.view()).expect("make_fritsch_carlson failed");
        let val: f64 = fc.evaluate(1.5).expect("eval");
        assert!(val.is_finite());
    }

    #[test]
    fn test_make_steffen() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0];
        let st = make_steffen(&x.view(), &y.view()).expect("make_steffen failed");
        let val: f64 = st.evaluate(1.5).expect("eval");
        assert!(val.is_finite());
    }
}
