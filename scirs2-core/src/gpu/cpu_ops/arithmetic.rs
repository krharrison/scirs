//! Activation function CPU fallback implementations.
//!
//! Provides forward and backward passes for ReLU, Sigmoid, Tanh, and GELU
//! used by the [`super::GpuContext`] CPU fallback operations.

use super::traits::{FloatOps, NumericOps};

// =============================================================================
// Activation functions (forward passes)
// =============================================================================

/// CPU fallback for ReLU: max(0, x)
pub(super) fn relu_cpu<T: FloatOps>(data: &[T]) -> Vec<T> {
    data.iter().map(|&x| x.max(T::zero())).collect()
}

/// CPU fallback for ReLU backward: grad * (input > 0 ? 1 : 0)
pub(super) fn relu_backward_cpu<T: FloatOps>(grad: &[T], input: &[T]) -> Vec<T> {
    grad.iter()
        .zip(input.iter())
        .map(|(&g, &x)| {
            if x.partial_lt(T::zero()) || x.to_f64() == 0.0 {
                T::zero()
            } else {
                g
            }
        })
        .collect()
}

/// CPU fallback for sigmoid: 1 / (1 + exp(-x))
pub(super) fn sigmoid_cpu<T: FloatOps>(data: &[T]) -> Vec<T> {
    data.iter()
        .map(|&x| {
            let exp_neg_x = x.neg().exp();
            T::one().div(T::one().add(exp_neg_x))
        })
        .collect()
}

/// CPU fallback for sigmoid backward: grad * sigmoid(x) * (1 - sigmoid(x))
pub(super) fn sigmoid_backward_cpu<T: FloatOps>(grad: &[T], input: &[T]) -> Vec<T> {
    grad.iter()
        .zip(input.iter())
        .map(|(&g, &x)| {
            let exp_neg_x = x.neg().exp();
            let sig = T::one().div(T::one().add(exp_neg_x));
            g.mul(sig.mul(T::one().sub(sig)))
        })
        .collect()
}

/// CPU fallback for tanh
pub(super) fn tanh_cpu<T: FloatOps>(data: &[T]) -> Vec<T> {
    data.iter().map(|&x| x.tanh_val()).collect()
}

/// CPU fallback for tanh backward: grad * (1 - tanh(x)^2)
pub(super) fn tanh_backward_cpu<T: FloatOps>(grad: &[T], input: &[T]) -> Vec<T> {
    grad.iter()
        .zip(input.iter())
        .map(|(&g, &x)| {
            let t = x.tanh_val();
            g.mul(T::one().sub(t.mul(t)))
        })
        .collect()
}

/// CPU fallback for GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
pub(super) fn gelu_cpu<T: FloatOps>(data: &[T]) -> Vec<T> {
    let sqrt2 = T::from_f64(std::f64::consts::SQRT_2);
    let half = T::from_f64(0.5);
    data.iter()
        .map(|&x| x.mul(half.mul(T::one().add(x.div(sqrt2).erf()))))
        .collect()
}

/// CPU fallback for GELU backward:
/// d/dx[GELU(x)] = 0.5*(1+erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
pub(super) fn gelu_backward_cpu<T: FloatOps>(grad: &[T], input: &[T]) -> Vec<T> {
    let sqrt2 = T::from_f64(std::f64::consts::SQRT_2);
    let half = T::from_f64(0.5);
    // 1/sqrt(2*pi)
    let inv_sqrt2pi = T::from_f64(1.0 / (2.0 * std::f64::consts::PI).sqrt());
    grad.iter()
        .zip(input.iter())
        .map(|(&g, &x)| {
            // cdf term: 0.5 * (1 + erf(x / sqrt(2)))
            let cdf = half.mul(T::one().add(x.div(sqrt2).erf()));
            // pdf term: (1/sqrt(2*pi)) * exp(-x^2 / 2)
            let neg_x_sq_half = x.mul(x).mul(half).neg();
            let pdf = inv_sqrt2pi.mul(neg_x_sq_half.exp());
            // d/dx GELU = cdf + x * pdf; gradient = grad * d/dx GELU
            g.mul(cdf.add(x.mul(pdf)))
        })
        .collect()
}
