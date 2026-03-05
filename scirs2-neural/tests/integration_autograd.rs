//! Integration tests verifying interaction between scirs2-neural and scirs2-autograd.
//!
//! These tests confirm:
//! - Neural network layers produce numerically consistent forward-pass outputs
//! - Autograd computes correct gradients via finite-difference verification
//! - Custom activation functions work with autodiff
//! - Loss function gradients are correct

use approx::assert_abs_diff_eq;
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops::{
    self as tops, convert_to_tensor, grad, matmul, relu, sigmoid, tanh,
};
use scirs2_core::ndarray::{array, Array, IxDyn};
use scirs2_neural::losses::{Loss, MeanSquaredError};
use scirs2_neural::{NeuralError, Result};

// ---------------------------------------------------------------------------
// Helper: finite-difference gradient checker
// ---------------------------------------------------------------------------

/// Numerically approximate ∂f/∂x[i] by central differences.
fn numerical_gradient<F: Fn(f64) -> f64>(f: F, x: f64, eps: f64) -> f64 {
    (f(x + eps) - f(x - eps)) / (2.0 * eps)
}

// ---------------------------------------------------------------------------
// 1. Dense forward pass produces correct matrix product
// ---------------------------------------------------------------------------

#[test]
fn test_dense_forward_pass_matches_manual_matmul() {
    // A simple 2×3 weight matrix and 3-element input
    // We bypass the Dense struct (which uses an internal RNG) and
    // validate the underlying linear-algebra contract directly
    // using autograd tensors.
    ag::run(|g: &mut ag::Context<f64>| {
        // weight: [2, 3],  bias: [2]
        let w_data = array![[1.0_f64, 0.0, -1.0], [0.5, 1.0, 0.5]];
        let b_data = array![0.1_f64, -0.1];
        let x_data = array![[1.0_f64, 2.0, 3.0]]; // shape [1, 3]

        let w = convert_to_tensor(w_data.clone(), g); // [2, 3]
        let b = convert_to_tensor(b_data.clone(), g); // [2]
        let x = convert_to_tensor(x_data.clone(), g); // [1, 3]

        // y = x @ w^T  =>  shape [1, 2]
        let wt = tops::transpose(w, &[1, 0]);
        let y_no_bias = matmul(x, wt);

        let result = y_no_bias.eval(g).expect("forward pass eval failed");

        // Manual: [1, 0, -1] dot [1, 2, 3] = 1 + 0 - 3 = -2
        //         [0.5, 1, 0.5] dot [1, 2, 3] = 0.5 + 2 + 1.5 = 4
        assert_abs_diff_eq!(result[[0, 0]], -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 4.0, epsilon = 1e-10);

        // unused variables to satisfy borrow checker
        let _ = b;
    });
}

// ---------------------------------------------------------------------------
// 2. MSE loss forward pass and gradient correctness
// ---------------------------------------------------------------------------

#[test]
fn test_mse_loss_forward_value() {
    let mse = MeanSquaredError::new();
    let preds: Array<f64, IxDyn> = array![1.0_f64, 2.0, 3.0].into_dyn();
    let targets: Array<f64, IxDyn> = array![1.5_f64, 1.8, 2.5].into_dyn();

    let loss = mse.forward(&preds, &targets).expect("mse forward failed");

    // MSE = mean([(0.5)^2, (0.2)^2, (0.5)^2]) = mean([0.25, 0.04, 0.25]) = 0.18
    assert_abs_diff_eq!(loss, 0.18, epsilon = 1e-10);
}

#[test]
fn test_mse_loss_gradient_matches_finite_difference() {
    let mse = MeanSquaredError::new();
    let targets: Array<f64, IxDyn> = array![1.5_f64, 1.8, 2.5].into_dyn();

    // Analytical gradient at predictions = [1.0, 2.0, 3.0]
    let preds: Array<f64, IxDyn> = array![1.0_f64, 2.0, 3.0].into_dyn();
    let analytical = mse.backward(&preds, &targets).expect("mse backward failed");

    let eps = 1e-5_f64;
    let n = preds.len();
    for i in 0..n {
        let mut p_plus = preds.clone();
        let mut p_minus = preds.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;

        let loss_plus = mse
            .forward(&p_plus, &targets)
            .expect("mse forward + eps failed");
        let loss_minus = mse
            .forward(&p_minus, &targets)
            .expect("mse forward - eps failed");

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        assert_abs_diff_eq!(analytical[i], fd, epsilon = 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 3. Autograd: gradient of sigmoid activation
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_sigmoid_gradient() {
    // σ(x) gradient: σ(x) * (1 - σ(x))
    ag::run(|g: &mut ag::Context<f64>| {
        let x_data = array![0.0_f64, 1.0, -1.0];
        let x_var = tops::variable(x_data.clone(), g);

        let y = sigmoid(x_var);
        let loss = tops::sum_all(y);

        let grads = grad(&[&loss], &[&x_var]);
        let grad_x = grads[0].eval(g).expect("sigmoid grad eval failed");

        // Analytical: d σ(x)/dx = σ(x)*(1-σ(x))
        let x_vals = [0.0_f64, 1.0, -1.0];
        for (i, &xi) in x_vals.iter().enumerate() {
            let s = 1.0 / (1.0 + (-xi).exp());
            let expected = s * (1.0 - s);
            assert_abs_diff_eq!(grad_x[i], expected, epsilon = 1e-6);
        }
    });
}

// ---------------------------------------------------------------------------
// 4. Autograd: gradient of tanh activation
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_tanh_gradient() {
    // d tanh(x)/dx = 1 - tanh(x)^2
    ag::run(|g: &mut ag::Context<f64>| {
        let x_data = array![-2.0_f64, 0.0, 2.0];
        let x_var = tops::variable(x_data.clone(), g);

        let y = tanh(x_var);
        let loss = tops::sum_all(y);

        let grads = grad(&[&loss], &[&x_var]);
        let grad_x = grads[0].eval(g).expect("tanh grad eval failed");

        let x_vals = [-2.0_f64, 0.0, 2.0];
        for (i, &xi) in x_vals.iter().enumerate() {
            let th = xi.tanh();
            let expected = 1.0 - th * th;
            assert_abs_diff_eq!(grad_x[i], expected, epsilon = 1e-6);
        }
    });
}

// ---------------------------------------------------------------------------
// 5. Autograd: gradient of ReLU activation
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_relu_gradient() {
    // d ReLU(x)/dx = 1 if x > 0, else 0
    ag::run(|g: &mut ag::Context<f64>| {
        let x_data = array![-1.0_f64, 0.5, 2.0, -0.1];
        let x_var = tops::variable(x_data.clone(), g);

        let y = relu(x_var);
        let loss = tops::sum_all(y);

        let grads = grad(&[&loss], &[&x_var]);
        let grad_x = grads[0].eval(g).expect("relu grad eval failed");

        // x<0 → 0, x>0 → 1
        let x_vals = [-1.0_f64, 0.5, 2.0, -0.1];
        for (i, &xi) in x_vals.iter().enumerate() {
            let expected = if xi > 0.0 { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(grad_x[i], expected, epsilon = 1e-6);
        }
    });
}

// ---------------------------------------------------------------------------
// 6. Autograd: chain rule through linear + sigmoid (simulated 1-layer net)
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_chain_rule_linear_sigmoid() {
    // y = sigmoid(w * x + b)
    // loss = (y - target)^2
    // d loss/d w  verified by finite differences
    ag::run(|g: &mut ag::Context<f64>| {
        let x_val = 2.0_f64;
        let w_val = 0.5_f64;
        let b_val = -0.3_f64;
        let target_val = 0.8_f64;

        let x = tops::scalar(x_val, g);
        let w = tops::scalar(w_val, g);
        let b = tops::scalar(b_val, g);
        let target = tops::scalar(target_val, g);

        // pre-activation
        let z = tops::add(tops::mul(w, x), b);
        let y = sigmoid(z);
        let diff = tops::sub(y, target);
        let loss = tops::mul(diff, diff);

        let grads = grad(&[&loss], &[&w]);
        let dl_dw = grads[0].eval(g).expect("chain-rule grad eval failed");

        // Finite-difference check
        let eps = 1e-5_f64;
        let loss_fn = |ww: f64| -> f64 {
            let z = ww * x_val + b_val;
            let y = 1.0 / (1.0 + (-z).exp());
            (y - target_val).powi(2)
        };
        let fd_grad = (loss_fn(w_val + eps) - loss_fn(w_val - eps)) / (2.0 * eps);

        let dl_dw_val = if dl_dw.ndim() == 0 {
            dl_dw[[]]
        } else {
            dl_dw[0]
        };
        assert_abs_diff_eq!(dl_dw_val, fd_grad, epsilon = 1e-5);
    });
}

// ---------------------------------------------------------------------------
// 7. Autograd: gradient of matrix multiply (weight gradient for linear layer)
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_matmul_weight_gradient() {
    // loss = sum(X @ W)
    // d loss / d W = X^T (broadcast)
    ag::run(|g: &mut ag::Context<f64>| {
        let x_data = array![[1.0_f64, 2.0], [3.0, 4.0]]; // [2, 2]
        let w_data = array![[0.5_f64, -0.5], [1.0, 0.0]]; // [2, 2]

        let x = convert_to_tensor(x_data.clone(), g);
        let w = tops::variable(w_data.clone(), g);

        let y = matmul(x, w);
        let loss = tops::sum_all(y);

        let grads = grad(&[&loss], &[&w]);
        let grad_w = grads[0].eval(g).expect("matmul weight grad eval failed");

        // d(sum(XW))/dW_{ij} = sum_k X_{ki}  =  column sums of X^T = row sums of X
        // For X = [[1,2],[3,4]]  → X^T = [[1,3],[2,4]]
        // d loss / d W[:,0] = X^T[:,0] = [1+3, 2+4] is wrong — let's think again.
        // loss = sum_{i,j} (X @ W)_{ij}
        // = sum_i sum_j sum_k X_{ik} W_{kj}
        // d loss / d W_{kj} = sum_i X_{ik}   (column sums of X)
        // So grad_w shape [2,2]:
        // grad_w[0,:] = X column 0 summed = 1+3=4, repeated  → [4, 4]
        // grad_w[1,:] = X column 1 summed = 2+4=6, repeated  → [6, 6]
        assert_abs_diff_eq!(grad_w[[0, 0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_w[[0, 1]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_w[[1, 0]], 6.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_w[[1, 1]], 6.0, epsilon = 1e-6);
    });
}

// ---------------------------------------------------------------------------
// 8. MSE gradient is zero at the optimum
// ---------------------------------------------------------------------------

#[test]
fn test_mse_gradient_zero_at_optimum() {
    let mse = MeanSquaredError::new();
    let vals: Array<f64, IxDyn> = array![1.0_f64, 2.0, 3.0].into_dyn();

    // predictions == targets → gradient should be zero
    let grad_at_opt = mse
        .backward(&vals, &vals)
        .expect("mse backward at optimum failed");

    for &g in grad_at_opt.iter() {
        assert_abs_diff_eq!(g, 0.0, epsilon = 1e-12);
    }
}

// ---------------------------------------------------------------------------
// 9. Autograd: higher-order chain through two activations
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_two_layer_activation_chain() {
    // f(x) = tanh(sigmoid(x))
    // df/dx = (1 - tanh(sigmoid(x))^2) * sigmoid(x) * (1 - sigmoid(x))
    ag::run(|g: &mut ag::Context<f64>| {
        let x_data = array![0.5_f64];
        let x_var = tops::variable(x_data.clone(), g);

        let h = sigmoid(x_var);
        let out = tanh(h);
        let loss = tops::sum_all(out);

        let grads = grad(&[&loss], &[&x_var]);
        let grad_val = grads[0]
            .eval(g)
            .expect("two-layer activation grad eval failed");

        // Finite difference
        let eps = 1e-5_f64;
        let x0 = 0.5_f64;
        let f = |x: f64| -> f64 {
            let s = 1.0 / (1.0 + (-x).exp());
            s.tanh()
        };
        let fd = (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps);

        let grad_scalar = if grad_val.ndim() == 0 {
            grad_val[[]]
        } else {
            grad_val[0]
        };
        assert_abs_diff_eq!(grad_scalar, fd, epsilon = 1e-5);
    });
}

// ---------------------------------------------------------------------------
// 10. Neural loss backward shape matches input shape
// ---------------------------------------------------------------------------

#[test]
fn test_neural_loss_gradient_shape() -> Result<()> {
    let mse = MeanSquaredError::new();
    let shape = [4, 3];
    let preds: Array<f64, IxDyn> = Array::from_elem(shape, 0.5_f64).into_dyn();
    let targets: Array<f64, IxDyn> = Array::from_elem(shape, 1.0_f64).into_dyn();

    let grad_arr = mse.backward(&preds, &targets)?;

    assert_eq!(
        grad_arr.shape(),
        preds.shape(),
        "gradient shape must match predictions shape"
    );
    Ok(())
}
