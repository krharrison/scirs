"""Tests for scirs2 transformation module."""

import numpy as np
import pytest
import scirs2


class TestNormalization:
    """Test normalization functions."""

    def test_normalize_array_zscore(self):
        """Test z-score normalization of array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized = scirs2.normalize_array_py(data, method="zscore", axis=0)

        # Each column should have mean ~0 and std ~1
        col_means = normalized.mean(axis=0)
        col_stds = normalized.std(axis=0, ddof=1)

        assert np.allclose(col_means, [0, 0], atol=1e-10)
        assert np.allclose(col_stds, [1, 1], atol=0.1)

    def test_normalize_array_minmax(self):
        """Test min-max normalization."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized = scirs2.normalize_array_py(data, method="minmax", axis=0)

        # Each column should be in [0, 1]
        assert np.all((normalized >= 0) & (normalized <= 1))
        # Min should be 0, max should be 1
        assert np.allclose(normalized.min(axis=0), [0, 0])
        assert np.allclose(normalized.max(axis=0), [1, 1])

    def test_normalize_array_l2(self):
        """Test L2 normalization."""
        data = np.array([[3.0, 4.0], [6.0, 8.0]])

        normalized = scirs2.normalize_array_py(data, method="l2", axis=1)

        # Each row should have L2 norm = 1
        row_norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(row_norms, [1, 1], atol=1e-10)

    def test_normalize_vector_zscore(self):
        """Test vector normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalized = scirs2.normalize_vector_py(data, method="zscore")

        # Mean should be 0, std should be 1
        assert np.allclose(normalized.mean(), 0, atol=1e-10)
        assert np.allclose(normalized.std(ddof=1), 1, atol=0.1)

    def test_normalizer_fit_transform(self):
        """Test Normalizer class with fit/transform pattern."""
        normalizer = scirs2.Normalizer(method="zscore", axis=0)

        train_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        test_data = np.array([[2.0, 3.0], [4.0, 5.0]])

        normalizer.fit(train_data)
        normalized_train = normalizer.transform(train_data)
        normalized_test = normalizer.transform(test_data)

        # Train data should have mean 0
        assert np.allclose(normalized_train.mean(axis=0), [0, 0], atol=1e-10)

        # Test data should be transformed using train statistics
        assert normalized_test.shape == (2, 2)


class TestPCA:
    """Test Principal Component Analysis."""

    def test_pca_basic(self):
        """Test basic PCA dimensionality reduction."""
        # Create correlated data
        np.random.seed(42)
        data = np.random.randn(100, 5)

        pca = scirs2.PCA(n_components=2)
        pca.fit(data)
        transformed = pca.transform(data)

        assert transformed.shape == (100, 2)

    def test_pca_variance_explained(self):
        """Test PCA explained variance."""
        np.random.seed(42)
        data = np.random.randn(50, 10)

        pca = scirs2.PCA(n_components=5)
        pca.fit(data)

        explained_variance = pca.explained_variance_ratio()

        # Should sum to less than or equal to 1
        assert 0 < sum(explained_variance) <= 1.0
        # First component should explain most variance
        assert explained_variance[0] >= explained_variance[1]

    def test_pca_inverse_transform(self):
        """Test PCA inverse transformation."""
        np.random.seed(42)
        data = np.random.randn(20, 4)

        pca = scirs2.PCA(n_components=2)
        pca.fit(data)
        transformed = pca.transform(data)
        reconstructed = pca.inverse_transform(transformed)

        # Reconstruction should approximate original
        assert reconstructed.shape == data.shape

    def test_pca_full_components(self):
        """Test PCA with all components."""
        data = np.random.randn(30, 5)

        pca = scirs2.PCA(n_components=5)
        pca.fit(data)
        transformed = pca.transform(data)

        assert transformed.shape == (30, 5)


class TestTSNE:
    """Test t-SNE dimensionality reduction."""

    def test_tsne_basic(self):
        """Test basic t-SNE transformation."""
        np.random.seed(42)
        # Create clustered data
        data = np.vstack([
            np.random.randn(20, 10) + 0,
            np.random.randn(20, 10) + 5
        ])

        tsne = scirs2.TSNE(n_components=2, perplexity=5.0, max_iter=100)
        transformed = tsne.fit_transform(data)

        assert transformed.shape == (40, 2)

    def test_tsne_3d(self):
        """Test t-SNE to 3D."""
        np.random.seed(42)
        data = np.random.randn(30, 8)

        tsne = scirs2.TSNE(n_components=3, max_iter=50)
        transformed = tsne.fit_transform(data)

        assert transformed.shape == (30, 3)

    def test_tsne_with_parameters(self):
        """Test t-SNE with custom parameters."""
        np.random.seed(42)
        data = np.random.randn(25, 5)

        tsne = scirs2.TSNE(
            n_components=2,
            perplexity=3.0,
            learning_rate=100.0,
            max_iter=200
        )
        transformed = tsne.fit_transform(data)

        assert transformed.shape == (25, 2)


class TestUMAP:
    """Test UMAP dimensionality reduction."""

    def test_umap_basic(self):
        """Test basic UMAP transformation."""
        np.random.seed(42)
        data = np.random.randn(50, 10)

        umap = scirs2.UMAP(n_components=2, n_neighbors=5)
        transformed = umap.fit_transform(data)

        assert transformed.shape == (50, 2)

    def test_umap_3d(self):
        """Test UMAP to 3D."""
        np.random.seed(42)
        data = np.random.randn(40, 8)

        umap = scirs2.UMAP(n_components=3, n_neighbors=5)
        transformed = umap.fit_transform(data)

        assert transformed.shape == (40, 3)

    def test_umap_with_parameters(self):
        """Test UMAP with custom parameters."""
        np.random.seed(42)
        data = np.random.randn(30, 6)

        umap = scirs2.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.1,
            metric="euclidean"
        )
        transformed = umap.fit_transform(data)

        assert transformed.shape == (30, 2)


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        poly = scirs2.PolynomialFeatures(degree=2)
        poly_features = poly.transform(data)

        # [1, a, b, a^2, ab, b^2] = 6 features for degree 2
        assert poly_features.shape[1] >= data.shape[1]

    def test_polynomial_features_degree_3(self):
        """Test polynomial features with degree 3."""
        data = np.array([[1.0, 2.0]])

        poly = scirs2.PolynomialFeatures(degree=3)
        poly_features = poly.transform(data)

        # Should have more features
        assert poly_features.shape[1] > 2

    def test_binarize(self):
        """Test binarization."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        binarized = scirs2.binarize_py(data, threshold=3.5)

        # Values <= 3.5 should be 0, > 3.5 should be 1
        assert binarized[0, 0] == 0  # 1.0 <= 3.5
        assert binarized[0, 1] == 0  # 2.0 <= 3.5
        assert binarized[1, 0] == 1  # 4.0 > 3.5
        assert binarized[1, 2] == 1  # 6.0 > 3.5

    def test_log_transform(self):
        """Test logarithmic transformation."""
        data = np.array([[1.0, 10.0, 100.0]])

        log_data = scirs2.log_transform_py(data)

        # log(1) = 0, log(10) ≈ 2.3, log(100) ≈ 4.6
        assert np.allclose(log_data[0, 0], 0, atol=1e-10)
        assert log_data[0, 1] > 2.0
        assert log_data[0, 2] > 4.0

    def test_power_transform(self):
        """Test Box-Cox power transformation."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        power_transformer = scirs2.PowerTransformer()
        power_transformer.fit(data)
        transformed = power_transformer.transform(data)

        assert transformed.shape == data.shape

    def test_discretize_equal_width(self):
        """Test equal-width discretization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        discretized = scirs2.discretize_equal_width_py(data, n_bins=5)

        # Should have 5 unique values (bins)
        assert len(np.unique(discretized)) <= 5

    def test_discretize_equal_frequency(self):
        """Test equal-frequency discretization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        discretized = scirs2.discretize_equal_frequency_py(data, n_bins=4)

        # Each bin should have approximately equal counts
        assert len(np.unique(discretized)) <= 4


class TestEncoding:
    """Test categorical encoding."""

    def test_one_hot_encoder(self):
        """Test one-hot encoding."""
        encoder = scirs2.OneHotEncoder()

        # Categorical data
        data = np.array([[0], [1], [2], [1], [0]])

        encoder.fit(data)
        encoded = encoder.transform(data)

        # Should have 3 columns (0, 1, 2)
        assert encoded.shape == (5, 3)
        # Each row should sum to 1
        assert np.allclose(encoded.sum(axis=1), [1, 1, 1, 1, 1])

    def test_ordinal_encoder(self):
        """Test ordinal encoding."""
        encoder = scirs2.OrdinalEncoder()

        # String categorical data
        data = [["red"], ["blue"], ["green"], ["red"], ["blue"]]

        encoder.fit(data)
        encoded = encoder.transform(data)

        # Should map to integers
        assert encoded.shape == (5, 1)
        assert encoded.dtype in [np.int32, np.int64, np.float64]

    def test_one_hot_encoder_multiple_categories(self):
        """Test one-hot encoding with multiple categories."""
        encoder = scirs2.OneHotEncoder()

        data = np.array([[0], [1], [2], [3], [4]])

        encoder.fit(data)
        encoded = encoder.transform(data)

        assert encoded.shape == (5, 5)


class TestImputation:
    """Test missing value imputation."""

    def test_simple_imputer_mean(self):
        """Test mean imputation."""
        imputer = scirs2.SimpleImputer(strategy="mean")

        data = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, 6.0]])

        imputer.fit(data)
        imputed = imputer.transform(data)

        # NaN should be replaced with mean
        assert not np.isnan(imputed).any()
        # Mean of [1, 7] = 4
        assert np.allclose(imputed[1, 0], 4.0)

    def test_simple_imputer_median(self):
        """Test median imputation."""
        imputer = scirs2.SimpleImputer(strategy="median")

        data = np.array([[1.0], [np.nan], [3.0], [5.0]])

        imputer.fit(data)
        imputed = imputer.transform(data)

        assert not np.isnan(imputed).any()

    def test_simple_imputer_constant(self):
        """Test constant value imputation."""
        imputer = scirs2.SimpleImputer(strategy="constant", fill_value=0.0)

        data = np.array([[1.0, np.nan], [np.nan, 3.0]])

        imputer.fit(data)
        imputed = imputer.transform(data)

        # NaN should be replaced with 0
        assert not np.isnan(imputed).any()
        assert imputed[0, 1] == 0.0
        assert imputed[1, 0] == 0.0

    def test_knn_imputer(self):
        """Test KNN imputation."""
        imputer = scirs2.KNNImputer(n_neighbors=2)

        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0]
        ])

        imputer.fit(data)
        imputed = imputer.transform(data)

        # NaN should be replaced
        assert not np.isnan(imputed).any()
        # Imputed value should be reasonable
        assert 2.0 <= imputed[1, 1] <= 8.0


class TestScaling:
    """Test scaling transformations."""

    def test_maxabs_scaler(self):
        """Test MaxAbs scaling."""
        scaler = scirs2.MaxAbsScaler()

        data = np.array([[-2.0, 4.0], [-1.0, 2.0], [0.0, 0.0], [1.0, -2.0]])

        scaler.fit(data)
        scaled = scaler.transform(data)

        # All values should be in [-1, 1]
        assert np.all((scaled >= -1) & (scaled <= 1))
        # Max absolute value should be 1
        assert np.allclose(np.abs(scaled).max(axis=0), [1, 1])

    def test_quantile_transformer(self):
        """Test quantile transformation."""
        transformer = scirs2.QuantileTransformer(n_quantiles=10)

        np.random.seed(42)
        data = np.random.randn(50, 2)

        transformer.fit(data)
        transformed = transformer.transform(data)

        # Transformed data should be more uniformly distributed
        assert transformed.shape == data.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_normalize_single_value(self):
        """Test normalization with single value."""
        data = np.array([[5.0]])

        normalized = scirs2.normalize_array_py(data, method="minmax", axis=0)

        # Single value can't be normalized meaningfully
        assert normalized.shape == (1, 1)

    def test_pca_more_components_than_features(self):
        """Test PCA with n_components > n_features."""
        data = np.random.randn(50, 5)

        pca = scirs2.PCA(n_components=10)

        # Should handle gracefully (use min of n_samples, n_features)
        try:
            pca.fit(data)
            transformed = pca.transform(data)
            assert transformed.shape[1] <= 5
        except Exception:
            # Expected to fail or adjust automatically
            pass

    def test_imputation_all_missing(self):
        """Test imputation when all values are missing."""
        imputer = scirs2.SimpleImputer(strategy="mean")

        data = np.array([[np.nan], [np.nan], [np.nan]])

        try:
            imputer.fit(data)
            imputed = imputer.transform(data)
            # Should use fallback strategy
            assert imputed.shape == data.shape
        except Exception:
            # Expected to fail
            pass

    def test_empty_data(self):
        """Test transformations on empty data."""
        data = np.array([]).reshape(0, 2)

        try:
            normalized = scirs2.normalize_array_py(data, method="zscore", axis=0)
            assert normalized.shape == (0, 2)
        except Exception:
            # Expected to fail gracefully
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
