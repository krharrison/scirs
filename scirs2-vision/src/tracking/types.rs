//! Core types for multi-object tracking (SORT / ByteTrack).

/// An axis-aligned bounding box with detection score and optional class ID.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Left edge (x-coordinate of top-left corner).
    pub x1: f32,
    /// Top edge (y-coordinate of top-left corner).
    pub y1: f32,
    /// Right edge (x-coordinate of bottom-right corner).
    pub x2: f32,
    /// Bottom edge (y-coordinate of bottom-right corner).
    pub y2: f32,
    /// Detection confidence score in `[0, 1]`.
    pub score: f32,
    /// Optional class identifier.
    pub class_id: Option<usize>,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: Option<usize>) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
        }
    }

    /// Area of the bounding box in pixels².
    pub fn area(&self) -> f32 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }

    /// Intersection-over-Union with another bounding box.
    ///
    /// Returns 0 for non-overlapping boxes and 1 for identical boxes.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
            return 0.0;
        }

        let inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
        let union_area = self.area() + other.area() - inter_area;

        if union_area <= 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }

    /// Convert to centre-x, centre-y, width, height representation.
    pub fn to_xywh(&self) -> (f32, f32, f32, f32) {
        let cx = (self.x1 + self.x2) * 0.5;
        let cy = (self.y1 + self.y2) * 0.5;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        (cx, cy, w, h)
    }
}

// ---------------------------------------------------------------------------
// TrackState
// ---------------------------------------------------------------------------

/// Life-cycle state of a tracked object.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackState {
    /// Newly created track; not yet confirmed by enough consecutive matches.
    Tentative,
    /// Track has received enough consecutive matches to be considered reliable.
    Confirmed,
    /// Track was not matched in the most recent frame but has not yet aged out.
    Lost,
    /// Track has exceeded `max_age` without a match and will be removed.
    Deleted,
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// A single tracked object with its current state and bounding box.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Track {
    /// Globally unique identifier assigned at creation.
    pub track_id: u64,
    /// Current life-cycle state.
    pub state: TrackState,
    /// Most recent (predicted or updated) bounding box.
    pub bbox: BoundingBox,
    /// Total number of frames since this track was created.
    pub age: usize,
    /// Number of consecutive frames in which this track was matched.
    pub hits: usize,
    /// Number of frames since the last successful match.
    pub time_since_update: usize,
}

// ---------------------------------------------------------------------------
// SortConfig
// ---------------------------------------------------------------------------

/// Configuration for the SORT tracker.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SortConfig {
    /// Maximum number of frames a track can go un-matched before deletion.
    pub max_age: usize,
    /// Minimum number of consecutive matches required to confirm a track.
    pub min_hits: usize,
    /// IoU threshold used during Hungarian assignment (lower = more permissive).
    pub iou_threshold: f32,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            max_age: 3,
            min_hits: 3,
            iou_threshold: 0.3,
        }
    }
}

impl SortConfig {
    /// Create a fully-specified `SortConfig`.
    pub fn new(max_age: usize, min_hits: usize, iou_threshold: f32) -> Self {
        Self {
            max_age,
            min_hits,
            iou_threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// ByteTrackConfig
// ---------------------------------------------------------------------------

/// Configuration for the ByteTrack tracker (Zhang et al. 2022).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ByteTrackConfig {
    /// Score threshold separating *high-confidence* from *low-confidence* detections.
    pub high_thresh: f32,
    /// Minimum score; detections below this are discarded entirely.
    pub low_thresh: f32,
    /// IoU threshold for both association stages.
    pub match_thresh: f32,
    /// Maximum frames a lost track survives without a match.
    pub max_age: usize,
    /// Minimum consecutive matches required to confirm a new track.
    pub min_hits: usize,
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            high_thresh: 0.6,
            low_thresh: 0.1,
            match_thresh: 0.8,
            max_age: 30,
            min_hits: 3,
        }
    }
}

impl ByteTrackConfig {
    /// Create a fully-specified `ByteTrackConfig`.
    pub fn new(
        high_thresh: f32,
        low_thresh: f32,
        match_thresh: f32,
        max_age: usize,
        min_hits: usize,
    ) -> Self {
        Self {
            high_thresh,
            low_thresh,
            match_thresh,
            max_age,
            min_hits,
        }
    }
}

// ---------------------------------------------------------------------------
// TrackerResult
// ---------------------------------------------------------------------------

/// Output produced by a tracker after processing a single frame.
pub struct TrackerResult {
    /// All currently active (non-deleted) tracks, filtered by confirmation policy.
    pub tracks: Vec<Track>,
    /// Index of the frame that produced this result.
    pub frame_id: usize,
}
