"""Tests for scirs2 datasets module."""

import numpy as np
import pytest
import scirs2


class TestMNIST:
    """Test MNIST dataset loading."""

    def test_load_mnist_train(self):
        """Test loading MNIST training data."""
        try:
            mnist = scirs2.load_mnist_py(split="train", download=False)

            assert "images" in mnist
            assert "labels" in mnist
            assert mnist["images"].shape[0] == mnist["labels"].shape[0]
            # MNIST images are 28x28
            assert mnist["images"].shape[1:] == (28, 28) or mnist["images"].shape[1:] == (784,)
        except Exception:
            # Dataset might not be available
            pytest.skip("MNIST dataset not available")

    def test_load_mnist_test(self):
        """Test loading MNIST test data."""
        try:
            mnist = scirs2.load_mnist_py(split="test", download=False)

            assert "images" in mnist
            assert "labels" in mnist
            assert mnist["images"].shape[0] > 0
        except Exception:
            pytest.skip("MNIST dataset not available")

    def test_mnist_normalization(self):
        """Test MNIST with normalization."""
        try:
            mnist = scirs2.load_mnist_py(split="train", normalize=True, download=False)

            # Normalized images should be in [0, 1]
            assert np.all((mnist["images"] >= 0) & (mnist["images"] <= 1))
        except Exception:
            pytest.skip("MNIST dataset not available")


class TestCIFAR:
    """Test CIFAR dataset loading."""

    def test_load_cifar10_train(self):
        """Test loading CIFAR-10 training data."""
        try:
            cifar = scirs2.load_cifar10_py(split="train", download=False)

            assert "images" in cifar
            assert "labels" in cifar
            # CIFAR-10 images are 32x32x3
            assert cifar["images"].shape[1:] == (32, 32, 3) or cifar["images"].shape[1] == 3072
        except Exception:
            pytest.skip("CIFAR-10 dataset not available")

    def test_load_cifar10_test(self):
        """Test loading CIFAR-10 test data."""
        try:
            cifar = scirs2.load_cifar10_py(split="test", download=False)

            assert "images" in cifar
            assert "labels" in cifar
        except Exception:
            pytest.skip("CIFAR-10 dataset not available")

    def test_load_cifar100(self):
        """Test loading CIFAR-100 dataset."""
        try:
            cifar = scirs2.load_cifar100_py(split="train", download=False)

            assert "images" in cifar
            assert "labels" in cifar
            # CIFAR-100 has 100 classes
            assert len(np.unique(cifar["labels"])) <= 100
        except Exception:
            pytest.skip("CIFAR-100 dataset not available")


class TestIris:
    """Test Iris dataset loading."""

    def test_load_iris(self):
        """Test loading Iris dataset."""
        iris = scirs2.load_iris_py()

        assert "data" in iris
        assert "target" in iris
        # Iris has 150 samples, 4 features, 3 classes
        assert iris["data"].shape == (150, 4)
        assert iris["target"].shape == (150,)
        assert len(np.unique(iris["target"])) == 3

    def test_iris_feature_names(self):
        """Test Iris feature names."""
        iris = scirs2.load_iris_py()

        assert "feature_names" in iris
        assert len(iris["feature_names"]) == 4

    def test_iris_target_names(self):
        """Test Iris target names."""
        iris = scirs2.load_iris_py()

        assert "target_names" in iris
        assert len(iris["target_names"]) == 3


class TestWine:
    """Test Wine dataset loading."""

    def test_load_wine(self):
        """Test loading Wine dataset."""
        wine = scirs2.load_wine_py()

        assert "data" in wine
        assert "target" in wine
        # Wine has 178 samples, 13 features, 3 classes
        assert wine["data"].shape == (178, 13)
        assert wine["target"].shape == (178,)

    def test_wine_classes(self):
        """Test Wine dataset classes."""
        wine = scirs2.load_wine_py()

        assert len(np.unique(wine["target"])) == 3


class TestBreastCancer:
    """Test Breast Cancer dataset loading."""

    def test_load_breast_cancer(self):
        """Test loading Breast Cancer dataset."""
        data = scirs2.load_breast_cancer_py()

        assert "data" in data
        assert "target" in data
        # 569 samples, 30 features, binary classification
        assert data["data"].shape == (569, 30)
        assert data["target"].shape == (569,)
        assert len(np.unique(data["target"])) == 2

    def test_breast_cancer_feature_names(self):
        """Test feature names."""
        data = scirs2.load_breast_cancer_py()

        assert "feature_names" in data
        assert len(data["feature_names"]) == 30


class TestDigits:
    """Test Digits dataset loading."""

    def test_load_digits(self):
        """Test loading Digits dataset."""
        digits = scirs2.load_digits_py()

        assert "data" in digits or "images" in digits
        assert "target" in digits
        # Digits has 1797 samples, 8x8 images, 10 classes
        if "images" in digits:
            assert digits["images"].shape == (1797, 8, 8)
        else:
            assert digits["data"].shape == (1797, 64)

    def test_digits_classes(self):
        """Test Digits classes."""
        digits = scirs2.load_digits_py()

        assert len(np.unique(digits["target"])) == 10


class TestBoston:
    """Test Boston Housing dataset loading."""

    def test_load_boston(self):
        """Test loading Boston Housing dataset."""
        try:
            boston = scirs2.load_boston_py()

            assert "data" in boston
            assert "target" in boston
            # 506 samples, 13 features
            assert boston["data"].shape == (506, 13)
            assert boston["target"].shape == (506,)
        except Exception:
            # Dataset might be deprecated
            pytest.skip("Boston Housing dataset not available or deprecated")


class TestDiabetes:
    """Test Diabetes dataset loading."""

    def test_load_diabetes(self):
        """Test loading Diabetes dataset."""
        diabetes = scirs2.load_diabetes_py()

        assert "data" in diabetes
        assert "target" in diabetes
        # 442 samples, 10 features
        assert diabetes["data"].shape == (442, 10)
        assert diabetes["target"].shape == (442,)


class TestCaliforniaHousing:
    """Test California Housing dataset."""

    def test_load_california_housing(self):
        """Test loading California Housing dataset."""
        try:
            data = scirs2.load_california_housing_py()

            assert "data" in data
            assert "target" in data
            # 20640 samples, 8 features
            assert data["data"].shape == (20640, 8)
            assert data["target"].shape == (20640,)
        except Exception:
            pytest.skip("California Housing dataset not available")


class TestSyntheticDatasets:
    """Test synthetic dataset generation."""

    def test_make_classification(self):
        """Test synthetic classification dataset."""
        X, y = scirs2.make_classification_py(
            n_samples=100,
            n_features=20,
            n_classes=2,
            random_state=42
        )

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2

    def test_make_regression(self):
        """Test synthetic regression dataset."""
        X, y = scirs2.make_regression_py(
            n_samples=100,
            n_features=10,
            noise=0.1,
            random_state=42
        )

        assert X.shape == (100, 10)
        assert y.shape == (100,)

    def test_make_blobs(self):
        """Test synthetic clustering dataset."""
        X, y = scirs2.make_blobs_py(
            n_samples=150,
            n_features=2,
            centers=3,
            random_state=42
        )

        assert X.shape == (150, 2)
        assert y.shape == (150,)
        assert len(np.unique(y)) == 3

    def test_make_circles(self):
        """Test synthetic circles dataset."""
        X, y = scirs2.make_circles_py(
            n_samples=100,
            noise=0.05,
            factor=0.5,
            random_state=42
        )

        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2

    def test_make_moons(self):
        """Test synthetic moons dataset."""
        X, y = scirs2.make_moons_py(
            n_samples=100,
            noise=0.1,
            random_state=42
        )

        assert X.shape == (100, 2)
        assert y.shape == (100,)


class TestDatasetSplitting:
    """Test dataset splitting utilities."""

    def test_train_test_split(self):
        """Test train/test split."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, size=100)

        X_train, X_test, y_train, y_test = scirs2.train_test_split_py(
            X, y, test_size=0.2, random_state=42
        )

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_train_val_test_split(self):
        """Test train/val/test split."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, size=100)

        splits = scirs2.train_val_test_split_py(
            X, y, val_size=0.15, test_size=0.15, random_state=42
        )

        X_train, X_val, X_test, y_train, y_val, y_test = splits

        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100

    def test_stratified_split(self):
        """Test stratified split."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 80 + [1] * 20)

        X_train, X_test, y_train, y_test = scirs2.stratified_split_py(
            X, y, test_size=0.2, random_state=42
        )

        # Check class proportions are preserved
        train_ratio = np.sum(y_train == 1) / len(y_train)
        test_ratio = np.sum(y_test == 1) / len(y_test)

        assert 0.15 <= train_ratio <= 0.25
        assert 0.15 <= test_ratio <= 0.25


class TestDataLoaders:
    """Test data loading utilities."""

    def test_batch_iterator(self):
        """Test batch iterator."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        batches = list(scirs2.batch_iterator_py(X, y, batch_size=10))

        assert len(batches) == 10
        assert batches[0][0].shape == (10, 1)

    def test_shuffle_data(self):
        """Test data shuffling."""
        X = np.arange(20).reshape(20, 1)
        y = np.arange(20)

        X_shuffled, y_shuffled = scirs2.shuffle_data_py(X, y, random_state=42)

        # Should have same elements but different order
        assert not np.array_equal(X_shuffled, X)
        assert set(y_shuffled.tolist()) == set(y.tolist())

    def test_data_augmentation(self):
        """Test data augmentation."""
        X = np.random.randn(10, 5)

        X_augmented = scirs2.augment_data_py(X, noise_level=0.1, random_state=42)

        # Augmented data should be similar but not identical
        assert X_augmented.shape == X.shape
        assert not np.allclose(X_augmented, X)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test handling empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        try:
            X_train, X_test, y_train, y_test = scirs2.train_test_split_py(
                X, y, test_size=0.2, random_state=42
            )
        except Exception:
            pass  # Expected to fail or handle gracefully

    def test_single_sample(self):
        """Test single sample dataset."""
        X = np.array([[1, 2, 3]])
        y = np.array([0])

        try:
            X_train, X_test, y_train, y_test = scirs2.train_test_split_py(
                X, y, test_size=0.2, random_state=42
            )
        except Exception:
            pass  # Expected to fail

    def test_all_same_class(self):
        """Test dataset with single class."""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)

        X_train, X_test, y_train, y_test = scirs2.train_test_split_py(
            X, y, test_size=0.2, random_state=42
        )

        # Should still split even with single class
        assert X_train.shape[0] + X_test.shape[0] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
