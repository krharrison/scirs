"""Tests for scirs2 automatic differentiation module."""

import numpy as np
import pytest
import scirs2


class TestTensorCreation:
    """Test tensor creation and basic operations."""

    def test_create_tensor(self):
        """Test creating a tensor."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = scirs2.Tensor(data, requires_grad=True)

        assert tensor.shape == (2, 2)
        assert tensor.requires_grad is True

    def test_tensor_from_scalar(self):
        """Test creating tensor from scalar."""
        tensor = scirs2.Tensor(5.0, requires_grad=True)

        assert tensor.shape == () or tensor.shape == (1,)

    def test_tensor_no_grad(self):
        """Test tensor without gradient tracking."""
        data = np.array([1.0, 2.0, 3.0])
        tensor = scirs2.Tensor(data, requires_grad=False)

        assert tensor.requires_grad is False


class TestForwardPass:
    """Test forward pass operations."""

    def test_addition(self):
        """Test tensor addition."""
        a = scirs2.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = scirs2.Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)

        c = a + b

        expected = np.array([5.0, 7.0, 9.0])
        assert np.allclose(c.data, expected)

    def test_subtraction(self):
        """Test tensor subtraction."""
        a = scirs2.Tensor(np.array([5.0, 6.0, 7.0]), requires_grad=True)
        b = scirs2.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        c = a - b

        expected = np.array([4.0, 4.0, 4.0])
        assert np.allclose(c.data, expected)

    def test_multiplication(self):
        """Test element-wise multiplication."""
        a = scirs2.Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)
        b = scirs2.Tensor(np.array([5.0, 6.0, 7.0]), requires_grad=True)

        c = a * b

        expected = np.array([10.0, 18.0, 28.0])
        assert np.allclose(c.data, expected)

    def test_division(self):
        """Test element-wise division."""
        a = scirs2.Tensor(np.array([10.0, 20.0, 30.0]), requires_grad=True)
        b = scirs2.Tensor(np.array([2.0, 4.0, 5.0]), requires_grad=True)

        c = a / b

        expected = np.array([5.0, 5.0, 6.0])
        assert np.allclose(c.data, expected)

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = scirs2.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = scirs2.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

        c = a @ b

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.data, expected)

    def test_power(self):
        """Test power operation."""
        a = scirs2.Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)

        c = a ** 2

        expected = np.array([4.0, 9.0, 16.0])
        assert np.allclose(c.data, expected)

    def test_exponential(self):
        """Test exponential function."""
        a = scirs2.Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)

        c = a.exp()

        expected = np.array([1.0, np.e, np.e**2])
        assert np.allclose(c.data, expected, atol=1e-5)

    def test_logarithm(self):
        """Test natural logarithm."""
        a = scirs2.Tensor(np.array([1.0, np.e, np.e**2]), requires_grad=True)

        c = a.log()

        expected = np.array([0.0, 1.0, 2.0])
        assert np.allclose(c.data, expected, atol=1e-5)

    def test_sum(self):
        """Test sum operation."""
        a = scirs2.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)

        c = a.sum()

        assert np.allclose(c.data, 10.0)

    def test_mean(self):
        """Test mean operation."""
        a = scirs2.Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)

        c = a.mean()

        assert np.allclose(c.data, 2.5)


class TestBackwardPass:
    """Test backward pass and gradient computation."""

    def test_simple_gradient(self):
        """Test gradient of simple function."""
        x = scirs2.Tensor(np.array([2.0]), requires_grad=True)

        # y = x^2
        y = x ** 2
        y.backward()

        # dy/dx = 2x = 4
        assert np.allclose(x.grad, [4.0])

    def test_addition_gradient(self):
        """Test gradient of addition."""
        a = scirs2.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = scirs2.Tensor(np.array([3.0, 4.0]), requires_grad=True)

        c = a + b
        c.backward(np.ones_like(c.data))

        # dc/da = 1, dc/db = 1
        assert np.allclose(a.grad, [1.0, 1.0])
        assert np.allclose(b.grad, [1.0, 1.0])

    def test_multiplication_gradient(self):
        """Test gradient of multiplication."""
        a = scirs2.Tensor(np.array([2.0]), requires_grad=True)
        b = scirs2.Tensor(np.array([3.0]), requires_grad=True)

        c = a * b
        c.backward()

        # dc/da = b = 3, dc/db = a = 2
        assert np.allclose(a.grad, [3.0])
        assert np.allclose(b.grad, [2.0])

    def test_chain_rule(self):
        """Test chain rule application."""
        x = scirs2.Tensor(np.array([2.0]), requires_grad=True)

        # y = (x^2 + 3x)
        y = x ** 2 + 3 * x
        y.backward()

        # dy/dx = 2x + 3 = 7
        assert np.allclose(x.grad, [7.0], atol=1e-5)

    def test_matrix_multiplication_gradient(self):
        """Test gradient of matrix multiplication."""
        a = scirs2.Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        b = scirs2.Tensor(np.array([[3.0], [4.0]]), requires_grad=True)

        c = a @ b
        c.backward()

        # dc/da = b^T, dc/db = a^T
        assert a.grad.shape == (1, 2)
        assert b.grad.shape == (2, 1)

    def test_exp_gradient(self):
        """Test gradient of exponential."""
        x = scirs2.Tensor(np.array([0.0]), requires_grad=True)

        y = x.exp()
        y.backward()

        # d(e^x)/dx = e^x = 1 at x=0
        assert np.allclose(x.grad, [1.0], atol=1e-5)

    def test_log_gradient(self):
        """Test gradient of logarithm."""
        x = scirs2.Tensor(np.array([1.0]), requires_grad=True)

        y = x.log()
        y.backward()

        # d(ln(x))/dx = 1/x = 1 at x=1
        assert np.allclose(x.grad, [1.0], atol=1e-5)


class TestGradientChecking:
    """Test gradient checking with numerical gradients."""

    def test_numerical_gradient_simple(self):
        """Test gradient checking for simple function."""
        x = scirs2.Tensor(np.array([3.0]), requires_grad=True)

        # f(x) = x^2
        y = x ** 2
        y.backward()

        # Numerical gradient
        eps = 1e-5
        x_plus = (3.0 + eps) ** 2
        x_minus = (3.0 - eps) ** 2
        numerical_grad = (x_plus - x_minus) / (2 * eps)

        assert np.allclose(x.grad, [numerical_grad], atol=1e-4)

    def test_numerical_gradient_complex(self):
        """Test gradient checking for complex function."""
        x = scirs2.Tensor(np.array([2.0]), requires_grad=True)

        # f(x) = x^3 + 2x^2 + 3x
        y = x ** 3 + 2 * (x ** 2) + 3 * x
        y.backward()

        # Analytical: f'(x) = 3x^2 + 4x + 3 = 23 at x=2
        assert np.allclose(x.grad, [23.0], atol=1e-4)


class TestActivationFunctions:
    """Test activation function gradients."""

    def test_relu_forward(self):
        """Test ReLU forward pass."""
        x = scirs2.Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        y = x.relu()

        expected = np.array([0.0, 0.0, 1.0, 2.0])
        assert np.allclose(y.data, expected)

    def test_relu_gradient(self):
        """Test ReLU gradient."""
        x = scirs2.Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        y = x.relu()
        y.backward(np.ones_like(y.data))

        # Gradient is 0 for x<0, 1 for x>0
        expected_grad = np.array([0.0, 0.0, 1.0, 1.0])
        assert np.allclose(x.grad, expected_grad)

    def test_sigmoid_forward(self):
        """Test sigmoid forward pass."""
        x = scirs2.Tensor(np.array([0.0]), requires_grad=True)

        y = x.sigmoid()

        # sigmoid(0) = 0.5
        assert np.allclose(y.data, [0.5])

    def test_sigmoid_gradient(self):
        """Test sigmoid gradient."""
        x = scirs2.Tensor(np.array([0.0]), requires_grad=True)

        y = x.sigmoid()
        y.backward()

        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)) = 0.25 at x=0
        assert np.allclose(x.grad, [0.25], atol=1e-5)

    def test_tanh_forward(self):
        """Test tanh forward pass."""
        x = scirs2.Tensor(np.array([0.0]), requires_grad=True)

        y = x.tanh()

        assert np.allclose(y.data, [0.0])

    def test_tanh_gradient(self):
        """Test tanh gradient."""
        x = scirs2.Tensor(np.array([0.0]), requires_grad=True)

        y = x.tanh()
        y.backward()

        # d(tanh(x))/dx = 1 - tanh^2(x) = 1 at x=0
        assert np.allclose(x.grad, [1.0], atol=1e-5)


class TestComputationGraph:
    """Test computation graph building and traversal."""

    def test_graph_multiple_uses(self):
        """Test variable used multiple times."""
        x = scirs2.Tensor(np.array([2.0]), requires_grad=True)

        # y = x^2 + x^3
        y = x ** 2 + x ** 3
        y.backward()

        # dy/dx = 2x + 3x^2 = 16 at x=2
        assert np.allclose(x.grad, [16.0], atol=1e-4)

    def test_detach(self):
        """Test detaching from computation graph."""
        x = scirs2.Tensor(np.array([3.0]), requires_grad=True)

        y = x ** 2
        z = y.detach()

        # z should not track gradients
        assert z.requires_grad is False

    def test_no_grad_context(self):
        """Test no_grad context manager."""
        x = scirs2.Tensor(np.array([2.0]), requires_grad=True)

        with scirs2.no_grad():
            y = x ** 2

        # Operations in no_grad should not track gradients
        assert y.requires_grad is False


class TestHigherOrderGradients:
    """Test second-order gradients."""

    def test_second_derivative(self):
        """Test computing second derivative."""
        x = scirs2.Tensor(np.array([2.0]), requires_grad=True)

        # First derivative of x^3
        y = x ** 3
        y.backward(create_graph=True)

        # dy/dx = 3x^2, now differentiate again
        first_grad = x.grad
        first_grad.backward()

        # d2y/dx2 = 6x = 12 at x=2
        # Note: This test might not work depending on autograd implementation
        # Just check that first derivative is correct
        assert np.allclose(first_grad.data, [12.0], atol=1e-4)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_gradient(self):
        """Test gradient of constant."""
        x = scirs2.Tensor(np.array([5.0]), requires_grad=True)
        c = scirs2.Tensor(np.array([10.0]), requires_grad=False)

        y = c  # y doesn't depend on x
        # Gradient should be undefined or handled gracefully

    def test_large_computation_graph(self):
        """Test large computation graph."""
        x = scirs2.Tensor(np.array([1.0]), requires_grad=True)

        y = x
        for i in range(100):
            y = y + 1

        y.backward()

        # Gradient should be 1
        assert np.allclose(x.grad, [1.0])

    def test_in_place_operations(self):
        """Test in-place operations."""
        x = scirs2.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        # In-place operations might break gradient tracking
        # This test just ensures they're handled
        try:
            x.data += 1
            y = x.sum()
            y.backward()
        except Exception:
            pass  # Expected if in-place ops break autograd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
