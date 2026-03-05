"""Tests for scirs2 neural network module."""

import numpy as np
import pytest
import scirs2


class TestSequentialModel:
    """Test sequential neural network model."""

    def test_sequential_creation(self):
        """Test creating sequential model."""
        model = scirs2.Sequential()

        assert model is not None

    def test_sequential_add_layer(self):
        """Test adding layers to sequential model."""
        model = scirs2.Sequential()
        model.add(scirs2.Dense(10, 5, activation="relu"))
        model.add(scirs2.Dense(5, 2, activation="sigmoid"))

        assert model.num_layers() == 2

    def test_sequential_forward_pass(self):
        """Test forward pass through sequential model."""
        model = scirs2.Sequential()
        model.add(scirs2.Dense(3, 5, activation="relu"))
        model.add(scirs2.Dense(5, 2, activation="sigmoid"))

        # Input: batch_size=2, features=3
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        output = model.forward(input_data)

        # Output should be batch_size=2, output_dim=2
        assert output.shape == (2, 2)
        # Sigmoid output should be in [0, 1]
        assert np.all((output >= 0) & (output <= 1))

    def test_sequential_compile(self):
        """Test model compilation with optimizer and loss."""
        model = scirs2.Sequential()
        model.add(scirs2.Dense(3, 5, activation="relu"))
        model.add(scirs2.Dense(5, 1, activation="linear"))

        model.compile(optimizer="sgd", loss="mse", learning_rate=0.01)

        assert model.is_compiled()


class TestDenseLayer:
    """Test dense (fully connected) layer."""

    def test_dense_creation(self):
        """Test creating dense layer."""
        layer = scirs2.Dense(input_dim=10, output_dim=5, activation="relu")

        assert layer.input_dim() == 10
        assert layer.output_dim() == 5
        assert layer.activation() == "relu"

    def test_dense_forward_shape(self):
        """Test dense layer output shape."""
        layer = scirs2.Dense(4, 8, activation="relu")

        input_data = np.random.randn(3, 4)  # batch_size=3, input_dim=4
        output = layer.forward(input_data)

        assert output.shape == (3, 8)  # batch_size=3, output_dim=8

    def test_dense_activations(self):
        """Test different activation functions."""
        activations = ["relu", "sigmoid", "tanh", "linear"]

        for act in activations:
            layer = scirs2.Dense(5, 5, activation=act)
            input_data = np.random.randn(2, 5)
            output = layer.forward(input_data)

            assert output.shape == (2, 5)

            if act == "sigmoid":
                assert np.all((output >= 0) & (output <= 1))
            elif act == "tanh":
                assert np.all((output >= -1) & (output <= 1))
            elif act == "relu":
                assert np.all(output >= 0)


class TestConvLayer:
    """Test convolutional layer."""

    def test_conv2d_creation(self):
        """Test creating 2D convolutional layer."""
        layer = scirs2.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        assert layer.in_channels() == 3
        assert layer.out_channels() == 16
        assert layer.kernel_size() == 3

    def test_conv2d_forward_shape(self):
        """Test Conv2D output shape."""
        layer = scirs2.Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1)

        # Input: batch_size=2, channels=1, height=28, width=28
        input_data = np.random.randn(2, 1, 28, 28)
        output = layer.forward(input_data)

        # With padding=1, output size should be same
        assert output.shape == (2, 8, 28, 28)

    def test_conv2d_stride(self):
        """Test convolution with stride."""
        layer = scirs2.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=2)

        input_data = np.random.randn(1, 3, 32, 32)
        output = layer.forward(input_data)

        # With stride=2, spatial dimensions should be halved
        assert output.shape[2] <= 16
        assert output.shape[3] <= 16


class TestPoolingLayers:
    """Test pooling layers."""

    def test_maxpool2d_creation(self):
        """Test creating MaxPool2D layer."""
        layer = scirs2.MaxPool2D(kernel_size=2, stride=2)

        assert layer.kernel_size() == 2
        assert layer.stride() == 2

    def test_maxpool2d_forward(self):
        """Test MaxPool2D forward pass."""
        layer = scirs2.MaxPool2D(kernel_size=2, stride=2)

        # Input: batch_size=1, channels=1, height=4, width=4
        input_data = np.array([[[[1, 2, 3, 4],
                                  [5, 6, 7, 8],
                                  [9, 10, 11, 12],
                                  [13, 14, 15, 16]]]], dtype=np.float64)

        output = layer.forward(input_data)

        # Output should be 2x2 due to pooling
        assert output.shape == (1, 1, 2, 2)
        # MaxPool should select maximum values
        assert np.max(output) == 16

    def test_avgpool2d_forward(self):
        """Test AvgPool2D forward pass."""
        layer = scirs2.AvgPool2D(kernel_size=2, stride=2)

        input_data = np.ones((1, 1, 4, 4), dtype=np.float64) * 4.0
        output = layer.forward(input_data)

        assert output.shape == (1, 1, 2, 2)
        # Average of all 4s should be 4
        assert np.allclose(output, 4.0)


class TestActivationLayers:
    """Test activation layers."""

    def test_relu_layer(self):
        """Test ReLU activation layer."""
        layer = scirs2.ReLU()

        input_data = np.array([[-1.0, 0.0, 1.0, 2.0]])
        output = layer.forward(input_data)

        assert np.allclose(output, [[0.0, 0.0, 1.0, 2.0]])

    def test_sigmoid_layer(self):
        """Test Sigmoid activation layer."""
        layer = scirs2.Sigmoid()

        input_data = np.array([[0.0]])
        output = layer.forward(input_data)

        assert np.allclose(output, [[0.5]], atol=1e-5)

    def test_tanh_layer(self):
        """Test Tanh activation layer."""
        layer = scirs2.Tanh()

        input_data = np.array([[0.0]])
        output = layer.forward(input_data)

        assert np.allclose(output, [[0.0]], atol=1e-10)

    def test_softmax_layer(self):
        """Test Softmax activation layer."""
        layer = scirs2.Softmax()

        input_data = np.array([[1.0, 2.0, 3.0]])
        output = layer.forward(input_data)

        # Softmax output should sum to 1
        assert np.allclose(np.sum(output, axis=1), [1.0])
        # All values should be positive
        assert np.all(output > 0)


class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss(self):
        """Test mean squared error loss."""
        loss = scirs2.MSELoss()

        predictions = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([[1.0, 2.0, 3.0]])

        loss_value = loss.compute(predictions, targets)

        # Perfect prediction should have zero loss
        assert np.allclose(loss_value, 0.0)

    def test_mse_loss_nonzero(self):
        """Test MSE with non-zero loss."""
        loss = scirs2.MSELoss()

        predictions = np.array([[1.0, 2.0]])
        targets = np.array([[2.0, 3.0]])

        loss_value = loss.compute(predictions, targets)

        # Loss should be positive for wrong predictions
        assert loss_value > 0

    def test_cross_entropy_loss(self):
        """Test cross-entropy loss."""
        loss = scirs2.CrossEntropyLoss()

        # Predictions (logits) and targets (class indices)
        predictions = np.array([[2.0, 1.0, 0.1]])  # Predicts class 0
        targets = np.array([0])  # True class is 0

        loss_value = loss.compute(predictions, targets)

        # Loss should be positive
        assert loss_value >= 0

    def test_binary_cross_entropy_loss(self):
        """Test binary cross-entropy loss."""
        loss = scirs2.BinaryCrossEntropyLoss()

        predictions = np.array([[0.9, 0.1, 0.8]])
        targets = np.array([[1.0, 0.0, 1.0]])

        loss_value = loss.compute(predictions, targets)

        # Good predictions should have low loss
        assert loss_value < 0.5


class TestOptimizers:
    """Test optimizer implementations."""

    def test_sgd_optimizer(self):
        """Test stochastic gradient descent optimizer."""
        optimizer = scirs2.SGD(learning_rate=0.01)

        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        gradients = np.array([[0.1, 0.1], [0.1, 0.1]])

        updated_weights = optimizer.step(weights, gradients)

        # Weights should decrease
        assert np.all(updated_weights < weights)

    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        optimizer = scirs2.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        weights = np.array([[1.0, 2.0]])
        gradients = np.array([[0.1, 0.2]])

        updated_weights = optimizer.step(weights, gradients)

        # Weights should be updated
        assert not np.allclose(updated_weights, weights)

    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer."""
        optimizer = scirs2.RMSprop(learning_rate=0.001, decay_rate=0.9)

        weights = np.array([[1.0]])
        gradients = np.array([[0.5]])

        updated_weights = optimizer.step(weights, gradients)

        assert updated_weights.shape == weights.shape


class TestTrainingLoop:
    """Test training loop functionality."""

    def test_simple_training_loop(self):
        """Test training a simple model."""
        # Create simple model for XOR problem (approximation)
        model = scirs2.Sequential()
        model.add(scirs2.Dense(2, 4, activation="relu"))
        model.add(scirs2.Dense(4, 1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="bce", learning_rate=0.1)

        # XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        # Train for a few epochs
        initial_loss = model.evaluate(X, y)

        for epoch in range(10):
            model.train_step(X, y)

        final_loss = model.evaluate(X, y)

        # Loss should decrease (even if not solving XOR perfectly)
        assert final_loss < initial_loss or final_loss < 0.5

    def test_batch_training(self):
        """Test training with mini-batches."""
        model = scirs2.Sequential()
        model.add(scirs2.Dense(5, 3, activation="relu"))
        model.add(scirs2.Dense(3, 1, activation="linear"))
        model.compile(optimizer="sgd", loss="mse", learning_rate=0.01)

        # Random data
        X = np.random.randn(20, 5)
        y = np.random.randn(20, 1)

        # Train with batches
        batch_size = 4
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            model.train_step(X_batch, y_batch)

        # Model should be able to make predictions
        predictions = model.forward(X[:5])
        assert predictions.shape == (5, 1)


class TestRegularization:
    """Test regularization techniques."""

    def test_dropout_layer(self):
        """Test dropout layer."""
        layer = scirs2.Dropout(rate=0.5)

        input_data = np.ones((10, 10))

        # During training, some units should be dropped
        layer.training(True)
        output = layer.forward(input_data)

        # During inference, no dropout
        layer.training(False)
        output_test = layer.forward(input_data)

        assert np.allclose(output_test, input_data)

    def test_batch_normalization(self):
        """Test batch normalization layer."""
        layer = scirs2.BatchNorm(num_features=5)

        # Batch of data with mean != 0, std != 1
        input_data = np.random.randn(4, 5) * 10 + 5

        output = layer.forward(input_data)

        # After batch norm, mean should be close to 0, std close to 1
        assert np.allclose(output.mean(axis=0), 0, atol=0.1)
        assert np.allclose(output.std(axis=0), 1, atol=0.5)


class TestModelArchitectures:
    """Test common neural network architectures."""

    def test_resnet_block(self):
        """Test ResNet residual block."""
        block = scirs2.ResNetBlock(channels=16, stride=1)

        input_data = np.random.randn(2, 16, 28, 28)
        output = block.forward(input_data)

        # Residual connection should preserve dimensions
        assert output.shape == input_data.shape

    def test_simple_cnn(self):
        """Test simple CNN for image classification."""
        model = scirs2.Sequential()
        model.add(scirs2.Conv2D(1, 8, kernel_size=3, padding=1))
        model.add(scirs2.ReLU())
        model.add(scirs2.MaxPool2D(kernel_size=2, stride=2))
        model.add(scirs2.Flatten())
        model.add(scirs2.Dense(8 * 14 * 14, 10))

        # MNIST-like input
        input_data = np.random.randn(2, 1, 28, 28)
        output = model.forward(input_data)

        assert output.shape == (2, 10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model(self):
        """Test empty sequential model."""
        model = scirs2.Sequential()

        assert model.num_layers() == 0

    def test_single_sample_batch(self):
        """Test training with single sample."""
        model = scirs2.Sequential()
        model.add(scirs2.Dense(3, 1, activation="linear"))
        model.compile(optimizer="sgd", loss="mse")

        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([[1.0]])

        model.train_step(X, y)

        prediction = model.forward(X)
        assert prediction.shape == (1, 1)

    def test_large_batch(self):
        """Test with large batch size."""
        model = scirs2.Sequential()
        model.add(scirs2.Dense(10, 5))

        X = np.random.randn(1000, 10)
        output = model.forward(X)

        assert output.shape == (1000, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
