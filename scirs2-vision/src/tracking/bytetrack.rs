//! ByteTrack — two-stage IoU association tracker (Zhang et al. 2022).
//!
//! ## Association strategy
//!
//! 1. **First stage** – associate *high-confidence* detections
//!    (`score ≥ high_thresh`) with **all** active tracks using IoU cost
//!    and the Hungarian algorithm.
//!
//! 2. **Second stage** – associate *low-confidence* detections
//!    (`low_thresh ≤ score < high_thresh`) with tracks that were
//!    **not matched** in the first stage.
//!
//! Any detection below `low_thresh` is discarded entirely.
//! Unmatched detections (from either stage) above `high_thresh` seed new
//! tentative tracks.

use crate::tracking::{
    hungarian::hungarian_assign,
    kalman_box::KalmanBoxTracker,
    types::{BoundingBox, ByteTrackConfig, Track, TrackState, TrackerResult},
};

// ---------------------------------------------------------------------------
// Internal track entry
// ---------------------------------------------------------------------------

struct ByteTrack {
    kalman: KalmanBoxTracker,
    track_id: u64,
    state: TrackState,
    age: usize,
    hits: usize,
    time_since_update: usize,
    class_id: Option<usize>,
    score: f32,
}

impl ByteTrack {
    fn new(id: u64, bbox: &BoundingBox) -> Self {
        Self {
            kalman: KalmanBoxTracker::new(bbox),
            track_id: id,
            state: TrackState::Tentative,
            age: 1,
            hits: 1,
            time_since_update: 0,
            class_id: bbox.class_id,
            score: bbox.score,
        }
    }

    fn to_public(&self) -> Track {
        let mut bbox = self.kalman.get_state();
        bbox.class_id = self.class_id;
        bbox.score = self.score;
        Track {
            track_id: self.track_id,
            state: self.state.clone(),
            bbox,
            age: self.age,
            hits: self.hits,
            time_since_update: self.time_since_update,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: run one IoU-based assignment pass
// ---------------------------------------------------------------------------

/// Associate `dets` (detection indices into the outer slice) with `trk_indices`
/// (indices into `tracks`) using IoU cost and Hungarian algorithm.
///
/// Returns:
/// * `matched` – `(det_idx, trk_idx)` pairs with IoU ≥ `iou_thresh`
/// * `unmatched_dets` – detection indices not matched
/// * `unmatched_trks` – track indices not matched
fn associate(
    dets: &[usize],
    det_boxes: &[BoundingBox],
    trk_indices: &[usize],
    predicted: &[BoundingBox],
    iou_thresh: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if dets.is_empty() || trk_indices.is_empty() {
        return (Vec::new(), dets.to_vec(), trk_indices.to_vec());
    }

    // Build cost matrix: rows = detections, cols = tracks
    let cost: Vec<Vec<f32>> = dets
        .iter()
        .map(|&di| {
            trk_indices
                .iter()
                .map(|&ti| 1.0 - det_boxes[di].iou(&predicted[ti]))
                .collect()
        })
        .collect();

    let assignment = hungarian_assign(&cost);

    let mut matched = Vec::new();
    let mut unmatched_dets: Vec<usize> = Vec::new();
    let mut matched_trk_local: Vec<bool> = vec![false; trk_indices.len()];

    for (local_di, opt_local_ti) in assignment.iter().enumerate() {
        let di = dets[local_di];
        match opt_local_ti {
            Some(local_ti) if *local_ti < trk_indices.len() => {
                let local_ti = *local_ti;
                let iou = 1.0 - cost[local_di][local_ti];
                if iou >= iou_thresh {
                    matched.push((di, trk_indices[local_ti]));
                    matched_trk_local[local_ti] = true;
                } else {
                    unmatched_dets.push(di);
                }
            }
            _ => {
                unmatched_dets.push(di);
            }
        }
    }

    let unmatched_trks: Vec<usize> = trk_indices
        .iter()
        .enumerate()
        .filter(|&(li, _)| !matched_trk_local[li])
        .map(|(_, &ti)| ti)
        .collect();

    (matched, unmatched_dets, unmatched_trks)
}

// ---------------------------------------------------------------------------
// ByteTracker
// ---------------------------------------------------------------------------

/// ByteTrack multi-object tracker.
pub struct ByteTracker {
    config: ByteTrackConfig,
    tracks: Vec<ByteTrack>,
    next_id: u64,
}

impl ByteTracker {
    /// Create a new ByteTracker with the given configuration.
    pub fn new(config: ByteTrackConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Process one frame of detections.
    pub fn update(&mut self, detections: &[BoundingBox], frame_id: usize) -> TrackerResult {
        // 1. Predict all track positions.
        let predicted: Vec<BoundingBox> =
            self.tracks.iter_mut().map(|t| t.kalman.predict()).collect();

        // 2. Split detections by confidence.
        let high_dets: Vec<usize> = detections
            .iter()
            .enumerate()
            .filter(|(_, d)| d.score >= self.config.high_thresh)
            .map(|(i, _)| i)
            .collect();

        let low_dets: Vec<usize> = detections
            .iter()
            .enumerate()
            .filter(|(_, d)| d.score >= self.config.low_thresh && d.score < self.config.high_thresh)
            .map(|(i, _)| i)
            .collect();

        let all_trk: Vec<usize> = (0..self.tracks.len()).collect();

        // 3. First-stage: high-confidence detections vs. all tracks.
        let (matched1, unmatched_high, unmatched_trks1) = associate(
            &high_dets,
            detections,
            &all_trk,
            &predicted,
            self.config.match_thresh,
        );

        // Apply first-stage matches.
        let mut track_matched: Vec<bool> = vec![false; self.tracks.len()];
        for (di, ti) in &matched1 {
            self.apply_match(*di, *ti, detections);
            track_matched[*ti] = true;
        }

        // 4. Second-stage: low-confidence detections vs. unmatched tracks.
        let (matched2, _unmatched_low, unmatched_trks2) = associate(
            &low_dets,
            detections,
            &unmatched_trks1,
            &predicted,
            self.config.match_thresh,
        );

        for (di, ti) in &matched2 {
            self.apply_match(*di, *ti, detections);
            track_matched[*ti] = true;
        }

        // 5. Mark remaining unmatched tracks.
        for ti in &unmatched_trks2 {
            self.tracks[*ti].time_since_update += 1;
        }
        // Also tracks that weren't even in unmatched_trks2 but still unmatched
        for (ti, matched) in track_matched.iter().enumerate() {
            if !matched && self.tracks[ti].time_since_update == 0 {
                // predict already incremented implicitly; ensure consistency
                self.tracks[ti].time_since_update += 1;
            }
        }

        // 6. Create new tracks from unmatched high-confidence detections.
        for &di in &unmatched_high {
            let id = self.next_id;
            self.next_id += 1;
            self.tracks.push(ByteTrack::new(id, &detections[di]));
        }

        // 7. Update states and prune deleted tracks.
        for t in self.tracks.iter_mut() {
            t.age += 1;
            if t.time_since_update == 0 {
                if t.hits >= self.config.min_hits || frame_id < self.config.min_hits {
                    t.state = TrackState::Confirmed;
                } else {
                    t.state = TrackState::Tentative;
                }
            } else if t.time_since_update > self.config.max_age {
                t.state = TrackState::Deleted;
            } else {
                t.state = TrackState::Lost;
            }
        }
        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // 8. Return confirmed tracks.
        let tracks = self
            .tracks
            .iter()
            .filter(|t| {
                t.state == TrackState::Confirmed
                    || (t.state == TrackState::Tentative && t.hits >= self.config.min_hits)
            })
            .map(|t| t.to_public())
            .collect();

        TrackerResult { tracks, frame_id }
    }

    /// Apply a detection–track match.
    fn apply_match(&mut self, det_idx: usize, trk_idx: usize, dets: &[BoundingBox]) {
        self.tracks[trk_idx].kalman.update(&dets[det_idx]);
        self.tracks[trk_idx].hits += 1;
        self.tracks[trk_idx].time_since_update = 0;
        self.tracks[trk_idx].class_id = dets[det_idx].class_id;
        self.tracks[trk_idx].score = dets[det_idx].score;
    }

    /// Number of currently active tracks.
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}
