//! Tests for multi-object tracking (SORT / ByteTrack) and depth completion.

use scirs2_vision::{
    depth_completion::{DepthCompleter, DepthCompletionConfig, DepthMethod},
    tracking::{BoundingBox, ByteTrackConfig, ByteTracker, SortConfig, SortTracker, TrackState},
};

// ===========================================================================
// BoundingBox
// ===========================================================================

#[test]
fn test_bbox_iou_non_overlapping() {
    let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1.0, None);
    let b = BoundingBox::new(20.0, 20.0, 30.0, 30.0, 1.0, None);
    assert_eq!(a.iou(&b), 0.0);
}

#[test]
fn test_bbox_iou_identical() {
    let a = BoundingBox::new(10.0, 10.0, 50.0, 50.0, 1.0, None);
    let b = BoundingBox::new(10.0, 10.0, 50.0, 50.0, 1.0, None);
    let iou = a.iou(&b);
    assert!(
        (iou - 1.0).abs() < 1e-5,
        "IoU of identical boxes should be 1, got {iou}"
    );
}

#[test]
fn test_bbox_area() {
    let b = BoundingBox::new(0.0, 0.0, 4.0, 5.0, 1.0, None);
    assert!((b.area() - 20.0).abs() < 1e-5);
}

#[test]
fn test_bbox_to_xywh() {
    let b = BoundingBox::new(0.0, 0.0, 10.0, 20.0, 1.0, None);
    let (cx, cy, w, h) = b.to_xywh();
    assert!((cx - 5.0).abs() < 1e-5);
    assert!((cy - 10.0).abs() < 1e-5);
    assert!((w - 10.0).abs() < 1e-5);
    assert!((h - 20.0).abs() < 1e-5);
}

#[test]
fn test_bbox_iou_partial_overlap() {
    let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1.0, None);
    let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0, 1.0, None);
    let iou = a.iou(&b);
    // Intersection = 5×5 = 25; Union = 100 + 100 - 25 = 175
    let expected = 25.0 / 175.0;
    assert!(
        (iou - expected).abs() < 1e-5,
        "Got {iou}, expected {expected}"
    );
}

// ===========================================================================
// SortConfig default
// ===========================================================================

#[test]
fn test_sort_config_default() {
    let cfg = SortConfig::default();
    assert_eq!(cfg.max_age, 3);
    assert_eq!(cfg.min_hits, 3);
    assert!((cfg.iou_threshold - 0.3).abs() < 1e-5);
}

// ===========================================================================
// SORT tracker
// ===========================================================================

#[test]
fn test_sort_single_detection_creates_track() {
    let mut tracker = SortTracker::new(SortConfig::default());
    let dets = vec![BoundingBox::new(10.0, 10.0, 60.0, 60.0, 0.9, None)];
    let _result = tracker.update(&dets, 0);
    // At least one track should be alive (tentative or confirmed).
    assert!(tracker.num_tracks() >= 1, "Expected at least one track");
}

#[test]
fn test_sort_track_persists_across_frames() {
    // Use min_hits = 1 and max_age = 5 so that a single detection gets confirmed
    // immediately and the track survives.
    let cfg = SortConfig::new(5, 1, 0.1);
    let mut tracker = SortTracker::new(cfg);

    // Send the same detection for 3 frames.
    for frame in 0..3 {
        let dets = vec![BoundingBox::new(10.0, 10.0, 60.0, 60.0, 0.9, None)];
        let result = tracker.update(&dets, frame);
        assert!(
            !result.tracks.is_empty(),
            "Frame {frame}: expected at least one confirmed track"
        );
    }
}

#[test]
fn test_sort_track_deleted_after_max_age() {
    let max_age = 2usize;
    let cfg = SortConfig::new(max_age, 1, 0.1);
    let mut tracker = SortTracker::new(cfg);

    // Create a track with a detection in frame 0.
    let dets = vec![BoundingBox::new(10.0, 10.0, 60.0, 60.0, 0.9, None)];
    tracker.update(&dets, 0);

    // Then send empty detections for max_age + 1 frames.
    for frame in 1..=(max_age + 1) {
        tracker.update(&[], frame);
    }

    // The track should have been deleted by now.
    assert_eq!(
        tracker.num_tracks(),
        0,
        "Track should be deleted after max_age frames without update"
    );
}

#[test]
fn test_sort_multiple_objects() {
    let cfg = SortConfig::new(5, 1, 0.1);
    let mut tracker = SortTracker::new(cfg);

    let dets = vec![
        BoundingBox::new(0.0, 0.0, 20.0, 20.0, 0.9, Some(0)),
        BoundingBox::new(100.0, 100.0, 120.0, 120.0, 0.8, Some(1)),
    ];
    let result = tracker.update(&dets, 0);
    // Both detections should have seeded tracks.
    assert!(tracker.num_tracks() >= 2, "Expected at least 2 tracks");
    let _ = result;
}

// ===========================================================================
// ByteTrackConfig default
// ===========================================================================

#[test]
fn test_bytetrack_config_default() {
    let cfg = ByteTrackConfig::default();
    assert!((cfg.high_thresh - 0.6).abs() < 1e-5);
    assert!((cfg.low_thresh - 0.1).abs() < 1e-5);
    assert!((cfg.match_thresh - 0.8).abs() < 1e-5);
    assert_eq!(cfg.max_age, 30);
    assert_eq!(cfg.min_hits, 3);
}

// ===========================================================================
// ByteTracker
// ===========================================================================

#[test]
fn test_bytetrack_high_confidence_detection_first_stage() {
    let cfg = ByteTrackConfig::new(0.5, 0.1, 0.3, 10, 1);
    let mut tracker = ByteTracker::new(cfg);

    // High-confidence detection.
    let dets = vec![BoundingBox::new(10.0, 10.0, 60.0, 60.0, 0.9, None)];
    let _result = tracker.update(&dets, 0);
    assert!(
        tracker.num_tracks() >= 1,
        "High-confidence detection should create a track"
    );
}

#[test]
fn test_bytetrack_low_confidence_detection_second_stage() {
    let cfg = ByteTrackConfig::new(0.6, 0.1, 0.3, 10, 1);
    let mut tracker = ByteTracker::new(cfg);

    // First create a track with a high-confidence detection.
    let high_dets = vec![BoundingBox::new(10.0, 10.0, 60.0, 60.0, 0.9, None)];
    tracker.update(&high_dets, 0);

    // Now send a low-confidence detection in the same location.
    // It should be associated with the existing track in the second stage.
    let low_dets = vec![BoundingBox::new(12.0, 12.0, 62.0, 62.0, 0.3, None)];
    let _result = tracker.update(&low_dets, 1);
    // The track should still be alive.
    assert!(tracker.num_tracks() >= 1);
}

#[test]
fn test_bytetrack_multiple_objects() {
    let cfg = ByteTrackConfig::new(0.5, 0.1, 0.3, 10, 1);
    let mut tracker = ByteTracker::new(cfg);

    let dets = vec![
        BoundingBox::new(0.0, 0.0, 30.0, 30.0, 0.9, Some(0)),
        BoundingBox::new(200.0, 200.0, 230.0, 230.0, 0.8, Some(1)),
        BoundingBox::new(400.0, 400.0, 430.0, 430.0, 0.75, Some(2)),
    ];
    for frame in 0..3 {
        tracker.update(&dets, frame);
    }
    assert!(
        tracker.num_tracks() >= 3,
        "Expected at least 3 tracks for 3 non-overlapping objects"
    );
}

// ===========================================================================
// Hungarian algorithm
// ===========================================================================

#[test]
fn test_hungarian_2x2_optimal() {
    use scirs2_vision::tracking::hungarian_assign;
    let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let assign = hungarian_assign(&cost);
    assert_eq!(assign[0], Some(0));
    assert_eq!(assign[1], Some(1));
}

// ===========================================================================
// DepthCompletionConfig default
// ===========================================================================

#[test]
fn test_depth_completion_config_default() {
    let cfg = DepthCompletionConfig::default();
    assert!((cfg.max_depth - 100.0).abs() < 1e-5);
    assert!((cfg.min_depth - 0.1).abs() < 1e-5);
    assert_eq!(cfg.iterations, 5);
}

// ===========================================================================
// NearestNeighbor – no None values after completion
// ===========================================================================

#[test]
fn test_nearest_neighbor_no_none_after_completion() {
    let h = 8;
    let w = 8;
    let mut sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    // Place a few sparse depth points.
    sparse[1][1] = Some(5.0);
    sparse[3][5] = Some(8.0);
    sparse[6][2] = Some(3.5);

    let mut cfg = DepthCompletionConfig::default();
    cfg.method = DepthMethod::NearestNeighbor;
    let completer = DepthCompleter::new(cfg);
    let result = completer.complete(&sparse, None);

    // All pixels should be filled.
    for r in 0..h {
        for c in 0..w {
            assert!(
                result.dense_depth[r][c] > 0.0,
                "Pixel ({r},{c}) should be filled, got {}",
                result.dense_depth[r][c]
            );
        }
    }
}

// ===========================================================================
// PropagationFill convergence
// ===========================================================================

#[test]
fn test_propagation_fill_convergence() {
    let h = 10;
    let w = 10;
    let mut sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    sparse[0][0] = Some(1.0);
    sparse[9][9] = Some(10.0);
    sparse[0][9] = Some(5.0);
    sparse[9][0] = Some(7.0);

    let mut cfg1 = DepthCompletionConfig::default();
    cfg1.method = DepthMethod::PropagationFill;
    cfg1.iterations = 1;

    let mut cfg5 = DepthCompletionConfig::default();
    cfg5.method = DepthMethod::PropagationFill;
    cfg5.iterations = 5;

    let r1 = DepthCompleter::new(cfg1).complete(&sparse, None);
    let r5 = DepthCompleter::new(cfg5).complete(&sparse, None);

    let filled1 = r1
        .dense_depth
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v > 0.0)
        .count();
    let filled5 = r5
        .dense_depth
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v > 0.0)
        .count();

    assert!(
        filled5 >= filled1,
        "More iterations should fill at least as many pixels: iter5={filled5} iter1={filled1}"
    );
}

// ===========================================================================
// filled_pixels > 0 for sparse input
// ===========================================================================

#[test]
fn test_filled_pixels_greater_than_zero() {
    let h = 5;
    let w = 5;
    let mut sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    sparse[2][2] = Some(5.0);

    let mut cfg = DepthCompletionConfig::default();
    cfg.method = DepthMethod::NearestNeighbor;
    let result = DepthCompleter::new(cfg).complete(&sparse, None);
    assert!(
        result.filled_pixels > 0,
        "Should have filled pixels, got {}",
        result.filled_pixels
    );
}

// ===========================================================================
// Output shape matches input shape
// ===========================================================================

#[test]
fn test_output_shape_matches_input() {
    let h = 7;
    let w = 11;
    let sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    let cfg = DepthCompletionConfig::default();
    let result = DepthCompleter::new(cfg).complete(&sparse, None);
    assert_eq!(result.dense_depth.len(), h);
    for row in &result.dense_depth {
        assert_eq!(row.len(), w);
    }
    assert_eq!(result.confidence.len(), h);
    for row in &result.confidence {
        assert_eq!(row.len(), w);
    }
}

// ===========================================================================
// Sparse points have confidence 1.0
// ===========================================================================

#[test]
fn test_sparse_points_have_confidence_one() {
    let h = 5;
    let w = 5;
    let mut sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    sparse[1][1] = Some(3.0);
    sparse[3][3] = Some(7.0);

    let mut cfg = DepthCompletionConfig::default();
    cfg.method = DepthMethod::PropagationFill;
    let result = DepthCompleter::new(cfg).complete(&sparse, None);

    // Original sparse points should have confidence exactly 1.0.
    assert!(
        (result.confidence[1][1] - 1.0).abs() < 1e-5,
        "Sparse point confidence should be 1.0, got {}",
        result.confidence[1][1]
    );
    assert!(
        (result.confidence[3][3] - 1.0).abs() < 1e-5,
        "Sparse point confidence should be 1.0, got {}",
        result.confidence[3][3]
    );
}

// ===========================================================================
// InvDistWeighted: output shape matches input
// ===========================================================================

#[test]
fn test_inv_dist_weighted_output_shape() {
    let h = 6;
    let w = 8;
    let mut sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    sparse[0][0] = Some(2.0);
    sparse[5][7] = Some(9.0);

    let mut cfg = DepthCompletionConfig::default();
    cfg.method = DepthMethod::InvDistWeighted;
    let result = DepthCompleter::new(cfg).complete(&sparse, None);
    assert_eq!(result.dense_depth.len(), h);
    assert_eq!(result.dense_depth[0].len(), w);
}

// ===========================================================================
// SurfaceNormals: output shape matches input
// ===========================================================================

#[test]
fn test_surface_normals_with_rgb() {
    let h = 5;
    let w = 5;
    let mut sparse: Vec<Vec<Option<f32>>> = vec![vec![None; w]; h];
    sparse[2][2] = Some(5.0);

    let rgb: Vec<Vec<[u8; 3]>> = (0..h)
        .map(|r| {
            (0..w)
                .map(|c| {
                    let v = ((r + c) * 25) as u8;
                    [v, v, v]
                })
                .collect()
        })
        .collect();

    let mut cfg = DepthCompletionConfig::default();
    cfg.method = DepthMethod::SurfaceNormals;
    let result = DepthCompleter::new(cfg).complete(&sparse, Some(&rgb));
    assert_eq!(result.dense_depth.len(), h);
    assert_eq!(result.dense_depth[0].len(), w);
}

// ===========================================================================
// Empty input → empty output (no panic)
// ===========================================================================

#[test]
fn test_empty_input_no_panic() {
    let sparse: Vec<Vec<Option<f32>>> = vec![];
    let cfg = DepthCompletionConfig::default();
    let result = DepthCompleter::new(cfg).complete(&sparse, None);
    assert!(result.dense_depth.is_empty());
    assert_eq!(result.filled_pixels, 0);
}

// ===========================================================================
// TrackState equality
// ===========================================================================

#[test]
fn test_track_state_equality() {
    assert_eq!(TrackState::Tentative, TrackState::Tentative);
    assert_ne!(TrackState::Confirmed, TrackState::Lost);
    assert_ne!(TrackState::Lost, TrackState::Deleted);
}
