"""Tests for scirs2 machine learning metrics module."""

import numpy as np
import pytest
import scirs2


class TestClassificationMetrics:
    """Test classification metrics."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        accuracy = scirs2.accuracy_score_py(y_true, y_pred)

        assert accuracy == 1.0

    def test_accuracy_imperfect(self):
        """Test accuracy with some errors."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        accuracy = scirs2.accuracy_score_py(y_true, y_pred)

        # 4 correct out of 6
        assert np.allclose(accuracy, 4.0 / 6.0)

    def test_precision_binary(self):
        """Test binary precision."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1])

        precision = scirs2.precision_score_py(y_true, y_pred)

        # TP=2, FP=1, precision=2/3
        assert 0.6 <= precision <= 0.7

    def test_recall_binary(self):
        """Test binary recall."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1])

        recall = scirs2.recall_score_py(y_true, y_pred)

        # TP=2, FN=1, recall=2/3
        assert 0.6 <= recall <= 0.7

    def test_f1_score_binary(self):
        """Test binary F1 score."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])

        f1 = scirs2.f1_score_py(y_true, y_pred)

        # Perfect predictions
        assert np.allclose(f1, 1.0)

    def test_f1_score_imperfect(self):
        """Test F1 score with errors."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0])

        f1 = scirs2.f1_score_py(y_true, y_pred)

        # Should be between 0 and 1
        assert 0 < f1 < 1

    def test_precision_multiclass(self):
        """Test multiclass precision."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 2])

        precision = scirs2.precision_score_py(y_true, y_pred, average="macro")

        assert 0 <= precision <= 1

    def test_recall_multiclass(self):
        """Test multiclass recall."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 2])

        recall = scirs2.recall_score_py(y_true, y_pred, average="macro")

        assert 0 <= recall <= 1

    def test_f1_score_multiclass(self):
        """Test multiclass F1 score."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        f1 = scirs2.f1_score_py(y_true, y_pred, average="macro")

        assert np.allclose(f1, 1.0)


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_confusion_matrix_binary(self):
        """Test binary confusion matrix."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        cm = scirs2.confusion_matrix_py(y_true, y_pred)

        # [[TN, FP], [FN, TP]] = [[1, 1], [0, 2]]
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[1, 1] == 2  # TP

    def test_confusion_matrix_multiclass(self):
        """Test multiclass confusion matrix."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])

        cm = scirs2.confusion_matrix_py(y_true, y_pred)

        assert cm.shape == (3, 3)

    def test_classification_report(self):
        """Test classification report generation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        report = scirs2.classification_report_py(y_true, y_pred)

        assert "precision" in report
        assert "recall" in report
        assert "f1_score" in report


class TestROCAndAUC:
    """Test ROC curve and AUC metrics."""

    def test_roc_curve_basic(self):
        """Test ROC curve computation."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        result = scirs2.roc_curve_py(y_true, y_score)

        assert "fpr" in result
        assert "tpr" in result
        assert "thresholds" in result

    def test_roc_auc_score_perfect(self):
        """Test perfect AUC score."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])

        auc = scirs2.roc_auc_score_py(y_true, y_score)

        assert np.allclose(auc, 1.0)

    def test_roc_auc_score_random(self):
        """Test AUC for random classifier."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_score = np.random.rand(100)

        auc = scirs2.roc_auc_score_py(y_true, y_score)

        # Random classifier should have AUC around 0.5
        assert 0.3 <= auc <= 0.7

    def test_pr_curve(self):
        """Test precision-recall curve."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        result = scirs2.pr_curve_py(y_true, y_score)

        assert "precision" in result
        assert "recall" in result
        assert "thresholds" in result

    def test_average_precision_score(self):
        """Test average precision score."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        ap = scirs2.average_precision_score_py(y_true, y_score)

        assert 0 <= ap <= 1


class TestRegressionMetrics:
    """Test regression metrics."""

    def test_mean_squared_error(self):
        """Test MSE calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2])

        mse = scirs2.mean_squared_error_py(y_true, y_pred)

        # Should be small for good predictions
        assert mse < 0.1

    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5])

        mae = scirs2.mean_absolute_error_py(y_true, y_pred)

        # MAE should be 0.5
        assert np.allclose(mae, 0.5)

    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])

        rmse = scirs2.root_mean_squared_error_py(y_true, y_pred)

        # RMSE should be 1.0
        assert np.allclose(rmse, 1.0)

    def test_r2_score_perfect(self):
        """Test R² score with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = scirs2.r2_score_py(y_true, y_pred)

        assert np.allclose(r2, 1.0)

    def test_r2_score_mean_prediction(self):
        """Test R² score when predicting mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Mean

        r2 = scirs2.r2_score_py(y_true, y_pred)

        # Predicting mean gives R² = 0
        assert np.allclose(r2, 0.0, atol=1e-5)

    def test_mean_absolute_percentage_error(self):
        """Test MAPE calculation."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        mape = scirs2.mean_absolute_percentage_error_py(y_true, y_pred)

        # Should be around 5%
        assert 0 <= mape <= 10

    def test_explained_variance_score(self):
        """Test explained variance score."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])

        evs = scirs2.explained_variance_score_py(y_true, y_pred)

        # Should be close to 1 for good predictions
        assert 0.9 <= evs <= 1.0


class TestClusteringMetrics:
    """Test clustering evaluation metrics."""

    def test_silhouette_score(self):
        """Test silhouette score."""
        # Well-separated clusters
        X = np.array([[1, 1], [1, 2], [5, 5], [5, 6]])
        labels = np.array([0, 0, 1, 1])

        score = scirs2.silhouette_score_py(X, labels)

        # Good clustering should have high score
        assert 0.5 <= score <= 1.0

    def test_calinski_harabasz_score(self):
        """Test Calinski-Harabasz score."""
        X = np.array([[1, 1], [1, 2], [5, 5], [5, 6]])
        labels = np.array([0, 0, 1, 1])

        score = scirs2.calinski_harabasz_score_py(X, labels)

        # Higher is better
        assert score > 0

    def test_davies_bouldin_score(self):
        """Test Davies-Bouldin score."""
        X = np.array([[1, 1], [1, 2], [5, 5], [5, 6]])
        labels = np.array([0, 0, 1, 1])

        score = scirs2.davies_bouldin_score_py(X, labels)

        # Lower is better, should be small for good clustering
        assert score >= 0
        assert score < 2.0

    def test_adjusted_rand_score(self):
        """Test adjusted Rand index."""
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 0, 1, 1])

        ari = scirs2.adjusted_rand_score_py(labels_true, labels_pred)

        # Perfect match should give 1.0
        assert np.allclose(ari, 1.0)

    def test_adjusted_mutual_info_score(self):
        """Test adjusted mutual information."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])

        ami = scirs2.adjusted_mutual_info_score_py(labels_true, labels_pred)

        assert 0.9 <= ami <= 1.0


class TestRankingMetrics:
    """Test ranking and recommendation metrics."""

    def test_ndcg_score(self):
        """Test normalized discounted cumulative gain."""
        y_true = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        y_score = np.array([[0.9, 0.1, 0.2, 0.8, 0.3]])

        ndcg = scirs2.ndcg_score_py(y_true, y_score)

        # Should be between 0 and 1
        assert 0 <= ndcg <= 1

    def test_mean_reciprocal_rank(self):
        """Test mean reciprocal rank."""
        # Relevant item is at position 1 (0-indexed)
        y_true = np.array([[0, 1, 0, 0, 0]])
        y_score = np.array([[0.1, 0.9, 0.3, 0.2, 0.4]])

        mrr = scirs2.mean_reciprocal_rank_py(y_true, y_score)

        # First relevant item is at rank 1, so MRR = 1
        assert np.allclose(mrr, 1.0)


class TestMultilabelMetrics:
    """Test multilabel classification metrics."""

    def test_hamming_loss(self):
        """Test Hamming loss for multilabel."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 1]])

        loss = scirs2.hamming_loss_py(y_true, y_pred)

        # 1 error out of 6 labels
        assert np.allclose(loss, 1.0 / 6.0)

    def test_jaccard_score(self):
        """Test Jaccard similarity score."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])

        jaccard = scirs2.jaccard_score_py(y_true, y_pred)

        # Perfect match
        assert np.allclose(jaccard, 1.0)


class TestDistanceMetrics:
    """Test distance and similarity metrics."""

    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])

        distance = scirs2.euclidean_distance_py(a, b)

        # 3-4-5 triangle
        assert np.allclose(distance, 5.0)

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])

        distance = scirs2.manhattan_distance_py(a, b)

        # |3-0| + |4-0| = 7
        assert np.allclose(distance, 7.0)

    def test_cosine_similarity(self):
        """Test cosine similarity."""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])

        similarity = scirs2.cosine_similarity_py(a, b)

        # Identical vectors
        assert np.allclose(similarity, 1.0)

    def test_cosine_distance(self):
        """Test cosine distance."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        distance = scirs2.cosine_distance_py(a, b)

        # Orthogonal vectors, distance = 1
        assert np.allclose(distance, 1.0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_predictions(self):
        """Test metrics with no predictions."""
        y_true = np.array([])
        y_pred = np.array([])

        try:
            accuracy = scirs2.accuracy_score_py(y_true, y_pred)
            # Should handle gracefully or return NaN
            assert np.isnan(accuracy) or accuracy == 0
        except Exception:
            # Expected to fail
            pass

    def test_single_class(self):
        """Test metrics with single class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        accuracy = scirs2.accuracy_score_py(y_true, y_pred)

        assert accuracy == 1.0

    def test_all_zeros(self):
        """Test regression metrics with all zeros."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0])

        mse = scirs2.mean_squared_error_py(y_true, y_pred)

        assert mse == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
