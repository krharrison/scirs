//! SORT — Simple Online and Realtime Tracking (Bewley et al. 2016).
//!
//! Each active track is maintained by a Kalman filter (`KalmanBoxTracker`).
//! Per-frame association is solved with the Hungarian algorithm on an IoU
//! cost matrix.  Track life-cycle follows the standard hit / age policy:
//!
//! * **Tentative** → not yet `min_hits` consecutive matches.
//! * **Confirmed** → at least `min_hits` matches (or frame_id < min_hits).
//! * **Lost** → no match, but `time_since_update ≤ max_age`.
//! * **Deleted** → `time_since_update > max_age`; removed on next call.

use crate::tracking::{
    hungarian::hungarian_assign,
    kalman_box::KalmanBoxTracker,
    types::{BoundingBox, SortConfig, Track, TrackState, TrackerResult},
};

// ---------------------------------------------------------------------------
// Internal track entry
// ---------------------------------------------------------------------------

struct SortTrack {
    kalman: KalmanBoxTracker,
    track_id: u64,
    state: TrackState,
    age: usize,
    hits: usize,
    time_since_update: usize,
    class_id: Option<usize>,
}

impl SortTrack {
    fn new(id: u64, bbox: &BoundingBox) -> Self {
        Self {
            kalman: KalmanBoxTracker::new(bbox),
            track_id: id,
            state: TrackState::Tentative,
            age: 1,
            hits: 1,
            time_since_update: 0,
            class_id: bbox.class_id,
        }
    }

    fn to_public(&self) -> Track {
        let mut bbox = self.kalman.get_state();
        bbox.class_id = self.class_id;
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
// SortTracker
// ---------------------------------------------------------------------------

/// SORT multi-object tracker.
pub struct SortTracker {
    config: SortConfig,
    tracks: Vec<SortTrack>,
    next_id: u64,
}

impl SortTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: SortConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Process one frame of detections and return the current set of active tracks.
    ///
    /// # Arguments
    /// * `detections` – bounding boxes detected in this frame.
    /// * `frame_id`   – monotonically increasing frame counter (used for reporting).
    pub fn update(&mut self, detections: &[BoundingBox], frame_id: usize) -> TrackerResult {
        // 1. Predict new locations for all existing tracks.
        let predicted: Vec<BoundingBox> =
            self.tracks.iter_mut().map(|t| t.kalman.predict()).collect();

        // 2. Build IoU cost matrix (rows = detections, cols = tracks).
        let n_dets = detections.len();
        let n_trks = self.tracks.len();

        let cost: Vec<Vec<f32>> = (0..n_dets)
            .map(|di| {
                (0..n_trks)
                    .map(|ti| {
                        let iou = detections[di].iou(&predicted[ti]);
                        1.0 - iou // cost = 1 – IoU
                    })
                    .collect()
            })
            .collect();

        // 3. Hungarian assignment.
        let assignment = if n_dets > 0 && n_trks > 0 {
            hungarian_assign(&cost)
        } else {
            vec![None; n_dets]
        };

        // 4a. Collect matched pairs, respecting the IoU threshold.
        let mut matched_trk: Vec<bool> = vec![false; n_trks];
        let mut unmatched_dets: Vec<usize> = Vec::new();

        for (di, opt_ti) in assignment.iter().enumerate() {
            match opt_ti {
                Some(ti) if *ti < n_trks => {
                    let ti = *ti;
                    let iou_val = 1.0 - cost[di][ti];
                    if iou_val >= self.config.iou_threshold {
                        // Valid match
                        self.tracks[ti].kalman.update(&detections[di]);
                        self.tracks[ti].hits += 1;
                        self.tracks[ti].time_since_update = 0;
                        self.tracks[ti].class_id = detections[di].class_id;
                        matched_trk[ti] = true;
                    } else {
                        unmatched_dets.push(di);
                    }
                }
                _ => {
                    unmatched_dets.push(di);
                }
            }
        }

        // 4b. Mark unmatched tracks.
        for (ti, matched) in matched_trk.iter().enumerate() {
            if !matched {
                self.tracks[ti].time_since_update += 1;
            }
        }

        // 5. Create new tracks for unmatched detections.
        for &di in &unmatched_dets {
            let id = self.next_id;
            self.next_id += 1;
            self.tracks.push(SortTrack::new(id, &detections[di]));
        }

        // 6. Update states and remove deleted tracks.
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

        // 7. Return only confirmed (or freshly matched) tracks.
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

    /// Number of currently active tracks (all states except Deleted).
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}
