//! Tests for Delaunay triangulation

use super::*;
use scirs2_core::ndarray::{arr2, Array2};
use scirs2_core::random::{Rng, RngExt};
use std::collections::HashSet;

#[test]
fn test_delaunay_simple() {
    let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Should have 2 triangles for 4 points in a square
    assert_eq!(tri.simplices().len(), 2);

    // Each triangle should have 3 vertices
    for simplex in tri.simplices() {
        assert_eq!(simplex.len(), 3);

        // Each vertex index should be in range
        for &idx in simplex {
            assert!(idx < points.nrows());
        }
    }

    // Check the convex hull
    let hull = tri.convex_hull();
    assert_eq!(hull.len(), 4); // All 4 points form the convex hull of the square
}

#[test]
fn test_delaunay_with_interior_point() {
    let points = arr2(&[
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5], // Interior point
    ]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // The Bowyer-Watson algorithm should produce valid simplices
    assert!(!tri.simplices().is_empty(), "Expected at least 1 triangle");

    // Verify each simplex has valid indices and structure
    for simplex in tri.simplices() {
        assert_eq!(simplex.len(), 3, "2D simplices should have 3 vertices");
        for &idx in simplex {
            assert!(idx < 4, "Vertex index {} out of bounds", idx);
        }
    }

    // Basic triangulation check: all simplices should have unique vertices
    for simplex in tri.simplices() {
        let unique: std::collections::HashSet<_> = simplex.iter().collect();
        assert_eq!(
            unique.len(),
            simplex.len(),
            "Simplex has duplicate vertices"
        );
    }
}

#[test]
fn test_delaunay_3d() {
    let points = arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Each simplex should have 4 vertices (tetrahedron in 3D)
    for simplex in tri.simplices() {
        assert_eq!(simplex.len(), 4);
    }
}

#[test]
fn test_delaunay_4d() {
    // Test 4D Delaunay triangulation
    let points = arr2(&[
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5], // Interior point
    ]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Verify basic properties
    assert_eq!(tri.ndim(), 4);
    assert_eq!(tri.npoints(), 6);
    assert!(!tri.simplices().is_empty(), "Should have simplices");

    // Each simplex should have 5 vertices (4-simplex in 4D)
    for simplex in tri.simplices() {
        assert_eq!(
            simplex.len(),
            5,
            "4D simplices should have 5 vertices (ndim+1)"
        );

        // All vertex indices should be valid
        for &idx in simplex {
            assert!(idx < 6, "Vertex index {} out of bounds", idx);
        }

        // All vertices should be unique
        let unique: HashSet<_> = simplex.iter().collect();
        assert_eq!(
            unique.len(),
            simplex.len(),
            "Simplex has duplicate vertices"
        );
    }
}

#[test]
fn test_delaunay_5d() {
    // Test 5D Delaunay triangulation with minimal points
    let points = arr2(&[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Verify basic properties
    assert_eq!(tri.ndim(), 5);
    assert_eq!(tri.npoints(), 6);
    assert!(!tri.simplices().is_empty(), "Should have simplices");

    // Each simplex should have 6 vertices (5-simplex in 5D)
    for simplex in tri.simplices() {
        assert_eq!(
            simplex.len(),
            6,
            "5D simplices should have 6 vertices (ndim+1)"
        );

        // All vertex indices should be valid
        for &idx in simplex {
            assert!(idx < 6, "Vertex index {} out of bounds", idx);
        }
    }
}

#[test]
fn test_delaunay_high_dim_with_interior() {
    // Test 4D with interior point to verify in-hypersphere test
    let points = arr2(&[
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25], // Interior point
    ]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Should produce valid triangulation
    assert!(!tri.simplices().is_empty());

    // All simplices should have correct structure
    for simplex in tri.simplices() {
        assert_eq!(simplex.len(), 5, "4D simplices should have 5 vertices");

        // Verify all indices are valid
        for &idx in simplex {
            assert!(idx < 7);
        }

        // Verify uniqueness
        let unique: HashSet<_> = simplex.iter().collect();
        assert_eq!(unique.len(), simplex.len());
    }
}

#[test]
fn test_find_simplex() {
    let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Point inside the triangle
    let inside_point = [0.3, 0.3];
    assert!(tri.find_simplex(&inside_point).is_some());

    // Point outside the triangle
    let outside_point = [1.5, 1.5];
    assert!(tri.find_simplex(&outside_point).is_none());
}

#[test]
fn test_random_points_2d() {
    // Generate some random points
    let mut rng = scirs2_core::random::rng();

    let n = 20;
    let mut points_data = Vec::with_capacity(n * 2);

    for _ in 0..n {
        points_data.push(rng.random_range(0.0..1.0));
        points_data.push(rng.random_range(0.0..1.0));
    }

    let points = Array2::from_shape_vec((n, 2), points_data).expect("Operation failed");

    let tri = Delaunay::new(&points).expect("Operation failed");

    // Basic checks
    assert_eq!(tri.ndim(), 2);
    assert_eq!(tri.npoints(), n);

    // Each simplex should have 3 valid vertex indices
    for simplex in tri.simplices() {
        assert_eq!(simplex.len(), 3);
        for &idx in simplex {
            assert!(idx < n);
        }
    }
}

#[test]
fn test_constrained_delaunay_basic() {
    let points = arr2(&[
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5], // Interior point
    ]);

    // Add constraint edges forming a square boundary
    let constraints = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

    let tri = Delaunay::new_constrained(&points, constraints.clone()).expect("Operation failed");

    // Check that constraints are stored
    assert_eq!(tri.constraints().len(), 4);
    for &constraint in &constraints {
        assert!(tri.constraints().contains(&constraint));
    }

    // Check that we have a valid triangulation
    assert!(tri.simplices().len() >= 2); // At least 2 triangles for this configuration
}

#[test]
fn test_constrained_delaunay_invalid_constraints() {
    let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);

    // Invalid constraint with out-of-bounds index
    let invalid_constraints = vec![(0, 5)];
    let result = Delaunay::new_constrained(&points, invalid_constraints);
    assert!(result.is_err());

    // Invalid constraint connecting point to itself
    let self_constraint = vec![(0, 0)];
    let result = Delaunay::new_constrained(&points, self_constraint);
    assert!(result.is_err());
}

#[test]
fn test_constrained_delaunay_3d() {
    let points_3d = arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]);

    let constraints = vec![(0, 1)];
    let result = Delaunay::new_constrained(&points_3d, constraints);
    // 3D constrained Delaunay is now supported
    assert!(result.is_ok());
    let tri = result.expect("Operation failed");
    assert!(tri.constraints().contains(&(0, 1)));
}

#[test]
fn test_edge_exists() {
    let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    let tri = Delaunay::new(&points).expect("Operation failed");

    // Check if edges exist in the triangle
    assert!(tri.edge_exists(0, 1) || tri.edge_exists(1, 0));
    assert!(tri.edge_exists(1, 2) || tri.edge_exists(2, 1));
    assert!(tri.edge_exists(0, 2) || tri.edge_exists(2, 0));
}
