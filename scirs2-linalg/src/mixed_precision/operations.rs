//! Element-wise operations and reductions in mixed precision
//!
//! Provides dot products, norms, AXPY, element-wise arithmetic, scaling,
//! and a dynamic loss scaler for mixed-precision training workflows.

use scirs2_core::ndarray::Array1;

use super::types::{BF16, F16};

// ============================================================================
// Dot products (accumulate in f32)
// ============================================================================

/// Compute the dot product of two F16 vectors, accumulating in f32.
///
/// Returns `0.0` when the vectors have different lengths (caller should
/// validate beforehand if this matters).
pub fn dot_f16(a: &Array1<F16>, b: &Array1<F16>) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut acc: f32 = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += av.to_f32() * bv.to_f32();
    }
    acc
}

/// Compute the dot product of two BF16 vectors, accumulating in f32.
pub fn dot_bf16(a: &Array1<BF16>, b: &Array1<BF16>) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut acc: f32 = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += av.to_f32() * bv.to_f32();
    }
    acc
}

// ============================================================================
// Norms
// ============================================================================

/// Compute the L2 (Euclidean) norm of an F16 vector, accumulating in f32.
pub fn norm_f16(a: &Array1<F16>) -> f32 {
    let mut acc: f32 = 0.0;
    for v in a.iter() {
        let f = v.to_f32();
        acc += f * f;
    }
    acc.sqrt()
}

/// Compute the L2 (Euclidean) norm of a BF16 vector, accumulating in f32.
pub fn norm_bf16(a: &Array1<BF16>) -> f32 {
    let mut acc: f32 = 0.0;
    for v in a.iter() {
        let f = v.to_f32();
        acc += f * f;
    }
    acc.sqrt()
}

// ============================================================================
// AXPY: y += alpha * x
// ============================================================================

/// Compute y += alpha * x where x is F16 and y is f32.
///
/// This is the classic BLAS Level-1 AXPY operation in mixed precision.
/// The F16 input `x` is promoted to f32 for the multiply-add.
pub fn axpy_f16(alpha: f32, x: &Array1<F16>, y: &mut Array1<f32>) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] += alpha * x[i].to_f32();
    }
}

/// Compute y += alpha * x where x is BF16 and y is f32.
pub fn axpy_bf16(alpha: f32, x: &Array1<BF16>, y: &mut Array1<f32>) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] += alpha * x[i].to_f32();
    }
}

// ============================================================================
// Element-wise arithmetic
// ============================================================================

/// Element-wise addition of two F16 arrays. Each pair is promoted to f32,
/// added, then converted back to F16.
pub fn elementwise_add_f16(a: &Array1<F16>, b: &Array1<F16>) -> Array1<F16> {
    let n = a.len().min(b.len());
    Array1::from_iter((0..n).map(|i| F16::from_f32(a[i].to_f32() + b[i].to_f32())))
}

/// Element-wise addition of two BF16 arrays.
pub fn elementwise_add_bf16(a: &Array1<BF16>, b: &Array1<BF16>) -> Array1<BF16> {
    let n = a.len().min(b.len());
    Array1::from_iter((0..n).map(|i| BF16::from_f32(a[i].to_f32() + b[i].to_f32())))
}

/// Element-wise multiplication of two F16 arrays.
pub fn elementwise_mul_f16(a: &Array1<F16>, b: &Array1<F16>) -> Array1<F16> {
    let n = a.len().min(b.len());
    Array1::from_iter((0..n).map(|i| F16::from_f32(a[i].to_f32() * b[i].to_f32())))
}

/// Element-wise multiplication of two BF16 arrays.
pub fn elementwise_mul_bf16(a: &Array1<BF16>, b: &Array1<BF16>) -> Array1<BF16> {
    let n = a.len().min(b.len());
    Array1::from_iter((0..n).map(|i| BF16::from_f32(a[i].to_f32() * b[i].to_f32())))
}

// ============================================================================
// Scaling
// ============================================================================

/// Scale an F16 array by a f32 scalar, returning a new F16 array.
pub fn scale_f16(alpha: f32, a: &Array1<F16>) -> Array1<F16> {
    Array1::from_iter(a.iter().map(|v| F16::from_f32(alpha * v.to_f32())))
}

/// Scale a BF16 array by a f32 scalar, returning a new BF16 array.
pub fn scale_bf16(alpha: f32, a: &Array1<BF16>) -> Array1<BF16> {
    Array1::from_iter(a.iter().map(|v| BF16::from_f32(alpha * v.to_f32())))
}

// ============================================================================
// Loss Scaler for mixed-precision training
// ============================================================================

/// Dynamic loss scaler for mixed-precision training.
///
/// Maintains a scaling factor that is applied to the loss before
/// backward pass to prevent gradient underflow in half-precision.
/// The scale grows geometrically after consecutive successful steps
/// and decreases when infinity/NaN is detected in gradients.
///
/// # Example
/// ```rust
/// use scirs2_linalg::mixed_precision::operations::LossScaler;
///
/// let mut scaler = LossScaler::new();
/// let loss = 0.5;
/// let scaled_loss = scaler.scale_loss(loss);
/// // ... compute gradients from scaled_loss ...
/// // Check for inf/nan in gradients, then:
/// scaler.update(false); // no inf found
/// ```
pub struct LossScaler {
    /// Current scaling factor
    scale: f64,
    /// Multiplicative factor to grow the scale after successful steps
    growth_factor: f64,
    /// Multiplicative factor to shrink the scale when inf/nan detected
    backoff_factor: f64,
    /// Number of consecutive successful steps before growing the scale
    growth_interval: usize,
    /// Counter of consecutive successful steps since last backoff
    consecutive_ok: usize,
}

impl LossScaler {
    /// Create a new `LossScaler` with default parameters.
    ///
    /// Defaults:
    /// - Initial scale: 2^16 = 65536
    /// - Growth factor: 2.0
    /// - Backoff factor: 0.5
    /// - Growth interval: 2000 steps
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_ok: 0,
        }
    }

    /// Create a `LossScaler` with custom parameters.
    pub fn with_params(
        initial_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: initial_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            consecutive_ok: 0,
        }
    }

    /// Return the current scale factor.
    pub fn current_scale(&self) -> f64 {
        self.scale
    }

    /// Scale the loss value by the current scale factor.
    pub fn scale_loss(&self, loss: f64) -> f64 {
        loss * self.scale
    }

    /// Unscale gradients by dividing by the current scale factor.
    ///
    /// This should be called after the backward pass to restore the
    /// original gradient magnitudes.
    pub fn unscale_gradients(&self, grads: &mut Array1<f64>) {
        if self.scale.abs() < f64::EPSILON {
            return;
        }
        let inv_scale = 1.0 / self.scale;
        grads.mapv_inplace(|v| v * inv_scale);
    }

    /// Update the scaler after a training step.
    ///
    /// - If `found_inf` is `true`: scale is reduced by `backoff_factor`
    ///   and the step counter resets.
    /// - If `found_inf` is `false`: increment the counter, and if
    ///   `growth_interval` consecutive successful steps have occurred,
    ///   grow the scale by `growth_factor`.
    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.consecutive_ok = 0;
        } else {
            self.consecutive_ok += 1;
            if self.consecutive_ok >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_ok = 0;
            }
        }
    }
}

impl Default for LossScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dot_f16() {
        let a = array![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
        let b = array![F16::from_f32(4.0), F16::from_f32(5.0), F16::from_f32(6.0)];
        let result = dot_f16(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((result - 32.0).abs() < 0.5);
    }

    #[test]
    fn test_dot_bf16() {
        let a = array![
            BF16::from_f32(1.0),
            BF16::from_f32(2.0),
            BF16::from_f32(3.0)
        ];
        let b = array![
            BF16::from_f32(4.0),
            BF16::from_f32(5.0),
            BF16::from_f32(6.0)
        ];
        let result = dot_bf16(&a, &b);
        assert!((result - 32.0).abs() < 1.0);
    }

    #[test]
    fn test_dot_f16_vs_f64_reference() {
        // Compare dot product against f64 reference
        let n = 100;
        let a_f16: Array1<F16> = Array1::from_iter((1..=n).map(|i| F16::from_f32(i as f32 * 0.1)));
        let b_f16: Array1<F16> = Array1::from_iter((1..=n).map(|i| F16::from_f32(i as f32 * 0.01)));

        let f16_dot = dot_f16(&a_f16, &b_f16);

        // f64 reference using the same f16 values
        let ref_dot: f64 = a_f16
            .iter()
            .zip(b_f16.iter())
            .map(|(a, b)| a.to_f64() * b.to_f64())
            .sum();

        let rel_err = ((f16_dot as f64) - ref_dot).abs() / ref_dot.abs().max(1e-30);
        assert!(
            rel_err < 0.01,
            "dot product relative error {rel_err} exceeds tolerance"
        );
    }

    #[test]
    fn test_norm_f16() {
        let a = array![F16::from_f32(3.0), F16::from_f32(4.0)];
        let n = norm_f16(&a);
        assert!((n - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_norm_bf16() {
        let a = array![BF16::from_f32(3.0), BF16::from_f32(4.0)];
        let n = norm_bf16(&a);
        assert!((n - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_axpy_f16() {
        let x = array![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
        let mut y = array![10.0f32, 20.0, 30.0];
        axpy_f16(2.0, &x, &mut y);
        // y = [10+2, 20+4, 30+6] = [12, 24, 36]
        assert!((y[0] - 12.0).abs() < 0.1);
        assert!((y[1] - 24.0).abs() < 0.1);
        assert!((y[2] - 36.0).abs() < 0.1);
    }

    #[test]
    fn test_elementwise_add_f16() {
        let a = array![F16::from_f32(1.0), F16::from_f32(2.0)];
        let b = array![F16::from_f32(3.0), F16::from_f32(4.0)];
        let c = elementwise_add_f16(&a, &b);
        assert!((c[0].to_f32() - 4.0).abs() < 0.01);
        assert!((c[1].to_f32() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_elementwise_mul_f16() {
        let a = array![F16::from_f32(2.0), F16::from_f32(3.0)];
        let b = array![F16::from_f32(4.0), F16::from_f32(5.0)];
        let c = elementwise_mul_f16(&a, &b);
        assert!((c[0].to_f32() - 8.0).abs() < 0.1);
        assert!((c[1].to_f32() - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_scale_f16() {
        let a = array![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(4.0)];
        let scaled = scale_f16(3.0, &a);
        assert!((scaled[0].to_f32() - 3.0).abs() < 0.1);
        assert!((scaled[1].to_f32() - 6.0).abs() < 0.1);
        assert!((scaled[2].to_f32() - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_loss_scaler_basic() {
        let scaler = LossScaler::new();
        assert_eq!(scaler.current_scale(), 65536.0);
        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);
    }

    #[test]
    fn test_loss_scaler_unscale() {
        let scaler = LossScaler::new();
        let mut grads = array![65536.0, 131072.0];
        scaler.unscale_gradients(&mut grads);
        assert!((grads[0] - 1.0).abs() < 1e-10);
        assert!((grads[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_loss_scaler_backoff_on_inf() {
        let mut scaler = LossScaler::new();
        let initial = scaler.current_scale();
        scaler.update(true); // found inf
        assert!((scaler.current_scale() - initial * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_loss_scaler_growth() {
        let mut scaler = LossScaler::with_params(1.0, 2.0, 0.5, 3);
        // 3 successful steps -> scale should double
        scaler.update(false);
        scaler.update(false);
        assert!((scaler.current_scale() - 1.0).abs() < 1e-10); // not yet
        scaler.update(false);
        assert!((scaler.current_scale() - 2.0).abs() < 1e-10); // now doubled
    }

    #[test]
    fn test_loss_scaler_reset_on_inf() {
        let mut scaler = LossScaler::with_params(4.0, 2.0, 0.5, 3);
        scaler.update(false);
        scaler.update(false);
        // 2 successful steps, then inf resets counter
        scaler.update(true);
        assert!((scaler.current_scale() - 2.0).abs() < 1e-10); // 4 * 0.5
                                                               // Need 3 more successful steps to grow
        scaler.update(false);
        scaler.update(false);
        scaler.update(false);
        assert!((scaler.current_scale() - 4.0).abs() < 1e-10); // 2 * 2
    }

    #[test]
    fn test_elementwise_add_bf16() {
        let a = array![BF16::from_f32(10.0), BF16::from_f32(-5.0)];
        let b = array![BF16::from_f32(20.0), BF16::from_f32(5.0)];
        let c = elementwise_add_bf16(&a, &b);
        assert!((c[0].to_f32() - 30.0).abs() < 1.0);
        assert!((c[1].to_f32() - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_scale_bf16() {
        let a = array![BF16::from_f32(2.0), BF16::from_f32(4.0)];
        let scaled = scale_bf16(0.5, &a);
        assert!((scaled[0].to_f32() - 1.0).abs() < 0.1);
        assert!((scaled[1].to_f32() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_dot_length_mismatch() {
        let a = array![F16::from_f32(1.0), F16::from_f32(2.0)];
        let b = array![F16::from_f32(1.0)];
        assert_eq!(dot_f16(&a, &b), 0.0);
    }
}
