//! SIMD-accelerated operations for clustering algorithms
//!
//! This module provides highly optimized SIMD implementations for core clustering
//! operations, leveraging the unified SIMD infrastructure from `scirs2-core`.
//! All functions automatically fall back to scalar implementations when SIMD
//! is not available or not beneficial.
//!
//! # Submodules
//!
//! - `distance`: Distance computation (Euclidean, Manhattan, pairwise, condensed)
//! - `clustering`: K-means, GMM, DBSCAN, distortion, and linkage operations

pub mod clustering;
pub mod distance;

// Re-export configuration and metric types
pub use distance::{SimdClusterConfig, SimdDistanceMetric};

// Re-export distance functions
pub use distance::{
    simd_distance, simd_euclidean_distance, simd_manhattan_distance,
    simd_pairwise_condensed_distances, simd_pairwise_distance_matrix,
    simd_squared_euclidean_distance,
};

// Re-export clustering functions
pub use clustering::{
    simd_assign_clusters, simd_batch_epsilon_neighborhood, simd_centroid_update,
    simd_compute_distortion, simd_epsilon_neighborhood, simd_gmm_log_responsibilities,
    simd_gmm_weighted_mean, simd_linkage_distances, simd_logsumexp, simd_logsumexp_rows,
};

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    /// Helper to create a simple 2-cluster test dataset.
    fn make_test_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, // cluster 0
                1.2, 1.8, // cluster 0
                0.8, 1.9, // cluster 0
                1.1, 2.1, // cluster 0
                5.0, 6.0, // cluster 1
                5.2, 5.8, // cluster 1
                4.8, 6.1, // cluster 1
                5.1, 5.9, // cluster 1
            ],
        )
        .expect("Failed to create test data")
    }

    /// Helper to create a force-simd config for testing.
    fn force_simd_config() -> SimdClusterConfig {
        SimdClusterConfig {
            force_simd: true,
            ..SimdClusterConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // Distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_euclidean_distance_basic() {
        let a = Array1::from(vec![1.0_f64, 0.0, 0.0]);
        let b = Array1::from(vec![0.0_f64, 1.0, 0.0]);
        let cfg = force_simd_config();

        let dist = simd_euclidean_distance(a.view(), b.view(), Some(&cfg))
            .expect("distance computation should succeed");
        let expected = 2.0_f64.sqrt();
        assert!(
            (dist - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            dist
        );
    }

    #[test]
    fn test_simd_euclidean_distance_same_point() {
        let a = Array1::from(vec![3.0_f64, 4.0, 5.0]);
        let cfg = force_simd_config();

        let dist = simd_euclidean_distance(a.view(), a.view(), Some(&cfg))
            .expect("distance computation should succeed");
        assert!(
            dist.abs() < 1e-10,
            "Distance to self should be 0, got {}",
            dist
        );
    }

    #[test]
    fn test_simd_euclidean_distance_length_mismatch() {
        let a = Array1::from(vec![1.0_f64, 2.0]);
        let b = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let result = simd_euclidean_distance(a.view(), b.view(), None);
        assert!(result.is_err(), "Should error on length mismatch");
    }

    #[test]
    fn test_simd_squared_euclidean_distance() {
        let a = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let b = Array1::from(vec![4.0_f64, 5.0, 6.0]);
        let cfg = force_simd_config();

        let dist = simd_squared_euclidean_distance(a.view(), b.view(), Some(&cfg))
            .expect("squared distance computation should succeed");
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9+9+9 = 27
        assert!((dist - 27.0).abs() < 1e-10, "Expected 27, got {}", dist);
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let a = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let b = Array1::from(vec![4.0_f64, 6.0, 1.0]);
        let cfg = force_simd_config();

        let dist = simd_manhattan_distance(a.view(), b.view(), Some(&cfg))
            .expect("manhattan distance should succeed");
        // |4-1| + |6-2| + |1-3| = 3 + 4 + 2 = 9
        assert!((dist - 9.0).abs() < 1e-10, "Expected 9, got {}", dist);
    }

    #[test]
    fn test_simd_distance_empty() {
        let a: Array1<f64> = Array1::zeros(0);
        let b: Array1<f64> = Array1::zeros(0);

        let dist = simd_euclidean_distance(a.view(), b.view(), None)
            .expect("empty distance should be zero");
        assert!((dist - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_simd_distance_scalar_fallback() {
        // Use a config with high threshold to force scalar path
        let cfg = SimdClusterConfig {
            simd_threshold: 1_000_000,
            force_simd: false,
            ..SimdClusterConfig::default()
        };
        let a = Array1::from(vec![1.0_f64, 0.0]);
        let b = Array1::from(vec![0.0_f64, 1.0]);

        let dist = simd_euclidean_distance(a.view(), b.view(), Some(&cfg))
            .expect("scalar fallback should succeed");
        assert!((dist - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Pairwise distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_pairwise_distance_matrix_symmetry() {
        let data = make_test_data();
        let cfg = force_simd_config();

        let dists =
            simd_pairwise_distance_matrix(data.view(), SimdDistanceMetric::Euclidean, Some(&cfg))
                .expect("pairwise distances should succeed");

        let n = data.shape()[0];
        assert_eq!(dists.shape(), &[n, n]);

        // Check symmetry and zero diagonal
        for i in 0..n {
            assert!(
                dists[[i, i]].abs() < 1e-10,
                "Diagonal should be zero at [{},{}]",
                i,
                i
            );
            for j in (i + 1)..n {
                assert!(
                    (dists[[i, j]] - dists[[j, i]]).abs() < 1e-10,
                    "Matrix should be symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_simd_pairwise_condensed_distances() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0_f64, 0.0, 3.0, 0.0, 0.0, 4.0])
            .expect("Failed to create test data");
        let cfg = force_simd_config();

        let dists = simd_pairwise_condensed_distances(
            data.view(),
            SimdDistanceMetric::Euclidean,
            Some(&cfg),
        )
        .expect("condensed distances should succeed");

        assert_eq!(dists.len(), 3); // 3*(3-1)/2 = 3
                                    // d(0,1) = 3.0, d(0,2) = 4.0, d(1,2) = 5.0
        assert!((dists[0] - 3.0).abs() < 1e-10);
        assert!((dists[1] - 4.0).abs() < 1e-10);
        assert!((dists[2] - 5.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // K-means cluster assignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_assign_clusters() {
        let data = make_test_data();
        let centroids = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 5.0, 6.0])
            .expect("Failed to create centroids");
        let cfg = force_simd_config();

        let (labels, distances) = simd_assign_clusters(
            data.view(),
            centroids.view(),
            SimdDistanceMetric::Euclidean,
            Some(&cfg),
        )
        .expect("cluster assignment should succeed");

        assert_eq!(labels.len(), 8);
        assert_eq!(distances.len(), 8);

        // First 4 points should be assigned to cluster 0
        for i in 0..4 {
            assert_eq!(labels[i], 0, "Point {} should be in cluster 0", i);
        }
        // Last 4 points should be assigned to cluster 1
        for i in 4..8 {
            assert_eq!(labels[i], 1, "Point {} should be in cluster 1", i);
        }
    }

    #[test]
    fn test_simd_assign_clusters_feature_mismatch() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create data");
        let centroids =
            Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).expect("Failed to create centroids");

        let result = simd_assign_clusters(
            data.view(),
            centroids.view(),
            SimdDistanceMetric::Euclidean,
            None,
        );
        assert!(
            result.is_err(),
            "Should error on feature dimension mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // K-means centroid update tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_centroid_update() {
        let data = make_test_data();
        let labels = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let cfg = force_simd_config();

        let centroids = simd_centroid_update(data.view(), &labels, 2, Some(&cfg))
            .expect("centroid update should succeed");

        assert_eq!(centroids.shape(), &[2, 2]);

        // Cluster 0 centroid: mean of rows 0..4
        let expected_c0_x = (1.0 + 1.2 + 0.8 + 1.1) / 4.0;
        let expected_c0_y = (2.0 + 1.8 + 1.9 + 2.1) / 4.0;
        assert!(
            (centroids[[0, 0]] - expected_c0_x).abs() < 1e-10,
            "Cluster 0 x: expected {}, got {}",
            expected_c0_x,
            centroids[[0, 0]]
        );
        assert!(
            (centroids[[0, 1]] - expected_c0_y).abs() < 1e-10,
            "Cluster 0 y: expected {}, got {}",
            expected_c0_y,
            centroids[[0, 1]]
        );

        // Cluster 1 centroid: mean of rows 4..8
        let expected_c1_x = (5.0 + 5.2 + 4.8 + 5.1) / 4.0;
        let expected_c1_y = (6.0 + 5.8 + 6.1 + 5.9) / 4.0;
        assert!(
            (centroids[[1, 0]] - expected_c1_x).abs() < 1e-10,
            "Cluster 1 x: expected {}, got {}",
            expected_c1_x,
            centroids[[1, 0]]
        );
        assert!(
            (centroids[[1, 1]] - expected_c1_y).abs() < 1e-10,
            "Cluster 1 y: expected {}, got {}",
            expected_c1_y,
            centroids[[1, 1]]
        );
    }

    #[test]
    fn test_simd_centroid_update_empty_cluster() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Failed to create data");
        // All points assigned to cluster 0; cluster 1 is empty
        let labels = Array1::from(vec![0usize, 0, 0, 0]);

        let centroids = simd_centroid_update(data.view(), &labels, 2, None)
            .expect("centroid update with empty cluster should succeed");

        // Cluster 1 should be all zeros (empty)
        assert!((centroids[[1, 0]]).abs() < 1e-15);
        assert!((centroids[[1, 1]]).abs() < 1e-15);
    }

    #[test]
    fn test_simd_centroid_update_label_mismatch() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create data");
        let labels = Array1::from(vec![0usize, 1]); // wrong length

        let result = simd_centroid_update(data.view(), &labels, 2, None);
        assert!(result.is_err(), "Should error on labels/data size mismatch");
    }

    // -----------------------------------------------------------------------
    // GMM log-responsibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_logsumexp_basic() {
        let values = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let cfg = force_simd_config();

        let result = simd_logsumexp(values.view(), Some(&cfg)).expect("logsumexp should succeed");

        // ln(e^1 + e^2 + e^3) = ln(e + e^2 + e^3)
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_simd_logsumexp_large_values() {
        // Test numerical stability with large values
        let values = Array1::from(vec![1000.0_f64, 1001.0, 1002.0]);
        let cfg = force_simd_config();

        let result = simd_logsumexp(values.view(), Some(&cfg))
            .expect("logsumexp should handle large values");

        // Should be close to 1002 + ln(e^(-2) + e^(-1) + 1)
        let expected = 1002.0 + ((-2.0_f64).exp() + (-1.0_f64).exp() + 1.0).ln();
        assert!(
            (result - expected).abs() < 1e-8,
            "Expected {}, got {} (large values)",
            expected,
            result
        );
    }

    #[test]
    fn test_simd_logsumexp_empty() {
        let values: Array1<f64> = Array1::zeros(0);
        let result = simd_logsumexp(values.view(), None);
        assert!(result.is_err(), "logsumexp of empty array should error");
    }

    #[test]
    fn test_simd_gmm_log_responsibilities() {
        // 3 samples, 2 components
        let log_prob = Array2::from_shape_vec(
            (3, 2),
            vec![
                -1.0_f64, -2.0, // sample 0
                -0.5, -3.0, // sample 1
                -2.0, -0.5, // sample 2
            ],
        )
        .expect("Failed to create log_prob");

        let log_weights = Array1::from(vec![(-0.5_f64).ln(), (-0.5_f64).ln()]);
        // Equal weights: ln(0.5) for both
        // Actually let's use proper weights summing to 1
        let w = 0.5_f64;
        let log_weights = Array1::from(vec![w.ln(), w.ln()]);
        let cfg = force_simd_config();

        let (resp, _lower_bound) =
            simd_gmm_log_responsibilities(log_prob.view(), log_weights.view(), Some(&cfg))
                .expect("GMM log responsibilities should succeed");

        assert_eq!(resp.shape(), &[3, 2]);

        // Each row should sum to approximately 1
        for i in 0..3 {
            let row_sum = resp.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-8,
                "Row {} responsibilities should sum to 1, got {}",
                i,
                row_sum
            );
        }

        // Sample 0: log_prob = [-1, -2], component 0 should have higher responsibility
        assert!(
            resp[[0, 0]] > resp[[0, 1]],
            "Sample 0 should prefer component 0"
        );

        // Sample 2: log_prob = [-2, -0.5], component 1 should have higher responsibility
        assert!(
            resp[[2, 1]] > resp[[2, 0]],
            "Sample 2 should prefer component 1"
        );
    }

    #[test]
    fn test_simd_gmm_log_responsibilities_dimension_mismatch() {
        let log_prob = Array2::from_shape_vec((2, 3), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create log_prob");
        let log_weights = Array1::from(vec![0.0_f64, 0.0]); // length 2 != 3

        let result = simd_gmm_log_responsibilities(log_prob.view(), log_weights.view(), None);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }

    // -----------------------------------------------------------------------
    // GMM weighted mean tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_gmm_weighted_mean() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0])
            .expect("Failed to create data");
        // Equal responsibilities
        let resp = Array1::from(vec![0.25_f64, 0.25, 0.25, 0.25]);
        let cfg = force_simd_config();

        let mean = simd_gmm_weighted_mean(data.view(), resp.view(), Some(&cfg))
            .expect("weighted mean should succeed");

        // Mean = (1+0+2+0)/4, (0+1+0+2)/4 = 0.75, 0.75
        assert!((mean[0] - 0.75).abs() < 1e-10);
        assert!((mean[1] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_simd_gmm_weighted_mean_nonuniform() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0_f64, 0.0, 10.0, 10.0])
            .expect("Failed to create data");
        // Heavy weight on second point
        let resp = Array1::from(vec![0.1_f64, 0.9]);
        let cfg = force_simd_config();

        let mean = simd_gmm_weighted_mean(data.view(), resp.view(), Some(&cfg))
            .expect("weighted mean should succeed");

        // Weighted mean: (0*0.1 + 10*0.9) / 1.0 = 9.0, 9.0
        assert!((mean[0] - 9.0).abs() < 1e-10);
        assert!((mean[1] - 9.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // DBSCAN neighborhood tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_epsilon_neighborhood() {
        let data = make_test_data();
        let cfg = force_simd_config();

        // Point 0 (1.0, 2.0) with eps=0.5 should find nearby points in cluster 0
        let neighbors = simd_epsilon_neighborhood(
            data.view(),
            0,
            0.5,
            SimdDistanceMetric::Euclidean,
            Some(&cfg),
        )
        .expect("neighborhood query should succeed");

        // Check that all neighbors are in the first cluster region
        for &n in &neighbors {
            assert!(n < 4, "Neighbor {} should be in cluster 0 region", n);
        }
    }

    #[test]
    fn test_simd_epsilon_neighborhood_large_eps() {
        let data = make_test_data();
        let cfg = force_simd_config();

        // Large epsilon should find all other points
        let neighbors = simd_epsilon_neighborhood(
            data.view(),
            0,
            100.0,
            SimdDistanceMetric::Euclidean,
            Some(&cfg),
        )
        .expect("large eps neighborhood should succeed");

        assert_eq!(
            neighbors.len(),
            7,
            "Large eps should include all other points"
        );
    }

    #[test]
    fn test_simd_epsilon_neighborhood_out_of_bounds() {
        let data = make_test_data();

        let result =
            simd_epsilon_neighborhood(data.view(), 100, 1.0, SimdDistanceMetric::Euclidean, None);
        assert!(result.is_err(), "Out-of-bounds query should error");
    }

    #[test]
    fn test_simd_batch_epsilon_neighborhood() {
        let data = make_test_data();
        let cfg = force_simd_config();

        let neighborhoods = simd_batch_epsilon_neighborhood(
            data.view(),
            1.0,
            SimdDistanceMetric::Euclidean,
            Some(&cfg),
        )
        .expect("batch neighborhood should succeed");

        assert_eq!(neighborhoods.len(), 8);

        // Symmetry: if j is in neighborhoods[i], then i should be in neighborhoods[j]
        for i in 0..8 {
            for &j in &neighborhoods[i] {
                assert!(
                    neighborhoods[j].contains(&i),
                    "Neighborhood should be symmetric: {} in N({}) but {} not in N({})",
                    j,
                    i,
                    i,
                    j
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Distortion (inertia) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_compute_distortion() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 1.0, 0.0, 10.0, 0.0, 11.0, 0.0])
                .expect("Failed to create data");
        let centroids = Array2::from_shape_vec((2, 2), vec![0.5_f64, 0.0, 10.5, 0.0])
            .expect("Failed to create centroids");
        let labels = Array1::from(vec![0usize, 0, 1, 1]);
        let cfg = force_simd_config();

        let distortion =
            simd_compute_distortion(data.view(), centroids.view(), &labels, Some(&cfg))
                .expect("distortion computation should succeed");

        // Cluster 0: d(0, c0)^2 + d(1, c0)^2 = 0.25 + 0.25 = 0.5
        // Cluster 1: d(2, c1)^2 + d(3, c1)^2 = 0.25 + 0.25 = 0.5
        // Total = 1.0
        assert!(
            (distortion - 1.0).abs() < 1e-10,
            "Expected distortion 1.0, got {}",
            distortion
        );
    }

    // -----------------------------------------------------------------------
    // Linkage distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_linkage_distances() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("Failed to create data");
        let cfg = force_simd_config();

        let dists = simd_linkage_distances(data.view(), SimdDistanceMetric::Euclidean, Some(&cfg))
            .expect("linkage distances should succeed");

        // 4 points => 4*3/2 = 6 distances
        assert_eq!(dists.len(), 6);

        // d(0,1) = 1.0, d(0,2) = 1.0, d(0,3) = sqrt(2),
        // d(1,2) = sqrt(2), d(1,3) = 1.0, d(2,3) = 1.0
        assert!((dists[0] - 1.0).abs() < 1e-10, "d(0,1) should be 1.0");
        assert!((dists[1] - 1.0).abs() < 1e-10, "d(0,2) should be 1.0");
        assert!(
            (dists[2] - 2.0_f64.sqrt()).abs() < 1e-10,
            "d(0,3) should be sqrt(2)"
        );
    }

    // -----------------------------------------------------------------------
    // f32 type support tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_operations_f32() {
        let a = Array1::from(vec![1.0_f32, 2.0, 3.0]);
        let b = Array1::from(vec![4.0_f32, 5.0, 6.0]);
        let cfg = force_simd_config();

        let dist = simd_euclidean_distance(a.view(), b.view(), Some(&cfg))
            .expect("f32 euclidean distance should succeed");
        let expected = 27.0_f32.sqrt();
        assert!(
            (dist - expected).abs() < 1e-5,
            "f32 distance: expected {}, got {}",
            expected,
            dist
        );

        let sq_dist = simd_squared_euclidean_distance(a.view(), b.view(), Some(&cfg))
            .expect("f32 squared euclidean distance should succeed");
        assert!(
            (sq_dist - 27.0_f32).abs() < 1e-5,
            "f32 squared distance: expected 27, got {}",
            sq_dist
        );

        let man_dist = simd_manhattan_distance(a.view(), b.view(), Some(&cfg))
            .expect("f32 manhattan distance should succeed");
        assert!(
            (man_dist - 9.0_f32).abs() < 1e-5,
            "f32 manhattan distance: expected 9, got {}",
            man_dist
        );
    }

    #[test]
    fn test_simd_centroid_update_f32() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
                .expect("Failed to create f32 data");
        let labels = Array1::from(vec![0usize, 0, 1, 1]);
        let cfg = force_simd_config();

        let centroids = simd_centroid_update(data.view(), &labels, 2, Some(&cfg))
            .expect("f32 centroid update should succeed");

        assert!((centroids[[0, 0]] - 2.0_f32).abs() < 1e-5);
        assert!((centroids[[0, 1]] - 3.0_f32).abs() < 1e-5);
        assert!((centroids[[1, 0]] - 20.0_f32).abs() < 1e-5);
        assert!((centroids[[1, 1]] - 30.0_f32).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Consistency test: SIMD vs scalar
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let data = make_test_data();

        let simd_cfg = force_simd_config();
        let scalar_cfg = SimdClusterConfig {
            simd_threshold: 1_000_000,
            force_simd: false,
            ..SimdClusterConfig::default()
        };

        // Compare Euclidean distances
        for i in 0..data.shape()[0] {
            for j in (i + 1)..data.shape()[0] {
                let simd_dist = simd_euclidean_distance(data.row(i), data.row(j), Some(&simd_cfg))
                    .expect("SIMD distance should succeed");
                let scalar_dist =
                    simd_euclidean_distance(data.row(i), data.row(j), Some(&scalar_cfg))
                        .expect("scalar distance should succeed");
                assert!(
                    (simd_dist - scalar_dist).abs() < 1e-10,
                    "SIMD and scalar Euclidean distances should match: {} vs {}",
                    simd_dist,
                    scalar_dist
                );
            }
        }

        // Compare Manhattan distances
        for i in 0..data.shape()[0] {
            for j in (i + 1)..data.shape()[0] {
                let simd_dist = simd_manhattan_distance(data.row(i), data.row(j), Some(&simd_cfg))
                    .expect("SIMD manhattan should succeed");
                let scalar_dist =
                    simd_manhattan_distance(data.row(i), data.row(j), Some(&scalar_cfg))
                        .expect("scalar manhattan should succeed");
                assert!(
                    (simd_dist - scalar_dist).abs() < 1e-10,
                    "SIMD and scalar Manhattan distances should match"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Higher-dimensional test
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_high_dimensional_distance() {
        let dim = 256;
        let a_vec: Vec<f64> = (0..dim).map(|i| i as f64 * 0.01).collect();
        let b_vec: Vec<f64> = (0..dim).map(|i| (dim - i) as f64 * 0.01).collect();
        let a = Array1::from(a_vec.clone());
        let b = Array1::from(b_vec.clone());
        let cfg = force_simd_config();

        let simd_dist = simd_euclidean_distance(a.view(), b.view(), Some(&cfg))
            .expect("high-dim SIMD distance should succeed");

        // Compute expected with scalar
        let mut expected_sq = 0.0_f64;
        for i in 0..dim {
            let d = a_vec[i] - b_vec[i];
            expected_sq += d * d;
        }
        let expected = expected_sq.sqrt();

        assert!(
            (simd_dist - expected).abs() < 1e-8,
            "High-dim distance: expected {}, got {}",
            expected,
            simd_dist
        );
    }
}
