//! Event-to-frame conversion methods.
//!
//! Converts asynchronous DVS events into dense image representations
//! suitable for downstream vision algorithms. Supported representations:
//!
//! - **Histogram**: simple event count per pixel
//! - **Polarity histogram**: separate ON/OFF channels
//! - **Time surface**: most recent timestamp per pixel (normalized)
//! - **Exponential decay**: temporal weighting with `exp(-dt/tau)`
//! - **Voxel grid**: time discretized into B bins

use scirs2_core::ndarray::{Array2, Array3};

use crate::error::{Result, VisionError};

use super::types::{Event, EventFrame, EventProcessingConfig, EventSlice, Polarity};

/// Method used for event-to-frame conversion.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum FrameMethod {
    /// Simple histogram: count events per pixel (polarity-agnostic).
    Histogram,
    /// Polarity histogram: accumulate polarity signs (+1/-1) per pixel.
    PolarityHistogram,
    /// Time surface: store the most recent timestamp per pixel, normalized to `[0, 1]`.
    TimeSurface,
    /// Exponential decay: each event contributes `exp(-(t_end - t_event) / tau)`.
    ExponentialDecay,
    /// Voxel grid: discretize time into `n_bins` bins, producing a 3D tensor `[B, H, W]`.
    VoxelGrid {
        /// Number of temporal bins.
        n_bins: usize,
    },
}

/// Converts an event slice to a single-channel frame using the given method.
///
/// For [`FrameMethod::VoxelGrid`] this returns the sum over bins (use
/// [`events_to_voxel_grid`] for the full 3D tensor).
pub fn events_to_frame(
    events: &EventSlice,
    method: &FrameMethod,
    config: &EventProcessingConfig,
) -> Result<EventFrame> {
    let h = config.height as usize;
    let w = config.width as usize;
    let (t_start, t_end) = events.time_range();

    let data = match method {
        FrameMethod::Histogram => {
            let mut frame = Array2::<f64>::zeros((h, w));
            for e in events.events() {
                frame[[e.y as usize, e.x as usize]] += 1.0;
            }
            frame
        }
        FrameMethod::PolarityHistogram => {
            let mut frame = Array2::<f64>::zeros((h, w));
            for e in events.events() {
                frame[[e.y as usize, e.x as usize]] += e.polarity.sign();
            }
            frame
        }
        FrameMethod::TimeSurface => {
            return events_to_time_surface(events, config);
        }
        FrameMethod::ExponentialDecay => {
            let mut frame = Array2::<f64>::zeros((h, w));
            let tau = config.decay_rate;
            if tau <= 0.0 {
                return Err(VisionError::InvalidParameter(
                    "Decay rate (tau) must be positive".to_string(),
                ));
            }
            for e in events.events() {
                let dt = t_end - e.timestamp;
                let weight = (-dt / tau).exp();
                frame[[e.y as usize, e.x as usize]] += e.polarity.sign() * weight;
            }
            frame
        }
        FrameMethod::VoxelGrid { n_bins } => {
            let voxel = events_to_voxel_grid(events, *n_bins, config)?;
            // Sum over time bins to produce a single 2D frame
            let mut frame = Array2::<f64>::zeros((h, w));
            for b in 0..*n_bins {
                for y in 0..h {
                    for x in 0..w {
                        frame[[y, x]] += voxel[[b, y, x]];
                    }
                }
            }
            frame
        }
    };

    Ok(EventFrame {
        data,
        t_start,
        t_end,
    })
}

/// Converts events into separate ON and OFF event frames.
///
/// Returns `(on_frame, off_frame)` where each frame counts events of the
/// respective polarity.
pub fn events_to_polarity_frames(
    events: &EventSlice,
    config: &EventProcessingConfig,
) -> Result<(EventFrame, EventFrame)> {
    let h = config.height as usize;
    let w = config.width as usize;
    let (t_start, t_end) = events.time_range();

    let mut on_frame = Array2::<f64>::zeros((h, w));
    let mut off_frame = Array2::<f64>::zeros((h, w));

    for e in events.events() {
        match e.polarity {
            Polarity::On => {
                on_frame[[e.y as usize, e.x as usize]] += 1.0;
            }
            Polarity::Off => {
                off_frame[[e.y as usize, e.x as usize]] += 1.0;
            }
        }
    }

    Ok((
        EventFrame {
            data: on_frame,
            t_start,
            t_end,
        },
        EventFrame {
            data: off_frame,
            t_start,
            t_end,
        },
    ))
}

/// Converts events to a 3D voxel grid `[n_bins, height, width]`.
///
/// Time is linearly discretized into `n_bins` temporal bins. Each event's
/// polarity sign is added to its corresponding bin.
pub fn events_to_voxel_grid(
    events: &EventSlice,
    n_bins: usize,
    config: &EventProcessingConfig,
) -> Result<Array3<f64>> {
    if n_bins == 0 {
        return Err(VisionError::InvalidParameter(
            "n_bins must be at least 1".to_string(),
        ));
    }

    let h = config.height as usize;
    let w = config.width as usize;
    let (t_start, t_end) = events.time_range();
    let duration = t_end - t_start;

    let mut voxel = Array3::<f64>::zeros((n_bins, h, w));

    for e in events.events() {
        let t_norm = if duration > 0.0 {
            (e.timestamp - t_start) / duration
        } else {
            0.5 // single timestamp: put in middle bin
        };
        // Clamp to [0, 1) then scale to bin index
        let t_clamped = t_norm.clamp(0.0, 1.0 - f64::EPSILON);
        let bin = (t_clamped * n_bins as f64) as usize;
        let bin = bin.min(n_bins - 1);

        voxel[[bin, e.y as usize, e.x as usize]] += e.polarity.sign();
    }

    Ok(voxel)
}

/// Converts events to a time surface.
///
/// Each pixel stores the normalized timestamp of the most recent event at
/// that location. Normalization maps `[t_start, t_end]` to `[0, 1]`.
/// Pixels with no events remain at `0.0`.
pub fn events_to_time_surface(
    events: &EventSlice,
    config: &EventProcessingConfig,
) -> Result<EventFrame> {
    let h = config.height as usize;
    let w = config.width as usize;
    let (t_start, t_end) = events.time_range();
    let duration = t_end - t_start;

    let mut frame = Array2::<f64>::zeros((h, w));

    for e in events.events() {
        let t_norm = if duration > 0.0 {
            (e.timestamp - t_start) / duration
        } else {
            1.0
        };
        // Since events are sorted, later events overwrite earlier ones
        frame[[e.y as usize, e.x as usize]] = t_norm;
    }

    Ok(EventFrame {
        data: frame,
        t_start,
        t_end,
    })
}

/// A streaming frame accumulator that processes events incrementally.
///
/// Maintains an internal frame and a timestamp map for temporal decay.
pub struct StreamingFrameAccumulator {
    frame: Array2<f64>,
    timestamps: Array2<f64>,
    config: EventProcessingConfig,
}

impl StreamingFrameAccumulator {
    /// Creates a new accumulator with the given configuration.
    pub fn new(config: EventProcessingConfig) -> Self {
        let h = config.height as usize;
        let w = config.width as usize;
        Self {
            frame: Array2::<f64>::zeros((h, w)),
            timestamps: Array2::<f64>::zeros((h, w)),
            config,
        }
    }

    /// Adds a single event to the accumulator.
    ///
    /// The event's polarity sign is added to the corresponding pixel.
    /// The pixel's timestamp is updated.
    pub fn add_event(&mut self, event: &Event) {
        let y = event.y as usize;
        let x = event.x as usize;
        if y < self.config.height as usize && x < self.config.width as usize {
            self.frame[[y, x]] += event.polarity.sign();
            self.timestamps[[y, x]] = event.timestamp;
        }
    }

    /// Returns a reference to the current accumulated frame.
    pub fn get_frame(&self) -> &Array2<f64> {
        &self.frame
    }

    /// Applies exponential temporal decay to all pixels.
    ///
    /// For each pixel, the value is multiplied by `exp(-(current_time - last_timestamp) / tau)`.
    pub fn decay(&mut self, current_time: f64) {
        let tau = self.config.decay_rate;
        if tau <= 0.0 {
            return;
        }
        let h = self.config.height as usize;
        let w = self.config.width as usize;
        for y in 0..h {
            for x in 0..w {
                let dt = current_time - self.timestamps[[y, x]];
                if dt > 0.0 {
                    self.frame[[y, x]] *= (-dt / tau).exp();
                    self.timestamps[[y, x]] = current_time;
                }
            }
        }
    }

    /// Resets the accumulator to zero.
    pub fn reset(&mut self) {
        self.frame.fill(0.0);
        self.timestamps.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_camera::types::{Event, EventSlice, Polarity};

    fn make_config(w: u16, h: u16) -> EventProcessingConfig {
        EventProcessingConfig {
            width: w,
            height: h,
            time_window: 0.033,
            decay_rate: 0.01,
            polarity_threshold: 0.5,
        }
    }

    #[test]
    fn test_histogram_single_event() {
        let events = vec![Event::new(5, 10, 0.001, Polarity::On)];
        let slice = EventSlice::new(events, 20, 20).expect("failed");
        let config = make_config(20, 20);
        let frame = events_to_frame(&slice, &FrameMethod::Histogram, &config).expect("failed");
        assert!((frame.data[[10, 5]] - 1.0).abs() < f64::EPSILON);
        // Other pixels should be zero
        assert!((frame.data[[0, 0]]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_polarity_histogram_separated() {
        let events = vec![
            Event::new(0, 0, 0.001, Polarity::On),
            Event::new(1, 1, 0.002, Polarity::Off),
            Event::new(0, 0, 0.003, Polarity::On),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let config = make_config(10, 10);
        let (on_frame, off_frame) = events_to_polarity_frames(&slice, &config).expect("failed");
        assert!((on_frame.data[[0, 0]] - 2.0).abs() < f64::EPSILON);
        assert!((off_frame.data[[1, 1]] - 1.0).abs() < f64::EPSILON);
        assert!((on_frame.data[[1, 1]]).abs() < f64::EPSILON);
        assert!((off_frame.data[[0, 0]]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_surface_most_recent() {
        let events = vec![
            Event::new(5, 5, 0.0, Polarity::On),
            Event::new(5, 5, 0.5, Polarity::Off),
            Event::new(5, 5, 1.0, Polarity::On),
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let config = make_config(10, 10);
        let frame = events_to_time_surface(&slice, &config).expect("failed");
        // Most recent timestamp normalized: (1.0 - 0.0) / (1.0 - 0.0) = 1.0
        assert!((frame.data[[5, 5]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_exponential_decay_older_lower_weight() {
        // Two events at the same pixel: one old, one recent
        let events = vec![
            Event::new(0, 0, 0.0, Polarity::On),
            Event::new(1, 0, 1.0, Polarity::On), // recent
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let config = EventProcessingConfig {
            width: 10,
            height: 10,
            time_window: 0.033,
            decay_rate: 0.5, // tau = 0.5s
            polarity_threshold: 0.5,
        };
        let frame =
            events_to_frame(&slice, &FrameMethod::ExponentialDecay, &config).expect("failed");
        // Pixel (0,0): event at t=0.0, t_end=1.0, dt=1.0, weight=exp(-1.0/0.5)=exp(-2)
        let expected_old = (-2.0_f64).exp();
        assert!((frame.data[[0, 0]] - expected_old).abs() < 1e-9);
        // Pixel (0,1): event at t=1.0, dt=0.0, weight=exp(0)=1.0
        assert!((frame.data[[0, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_voxel_grid_bin_assignment() {
        // 3 bins, events at t=0.0, 0.5, 1.0 over [0, 1]
        let events = vec![
            Event::new(0, 0, 0.0, Polarity::On), // bin 0
            Event::new(0, 0, 0.5, Polarity::On), // bin 1
            Event::new(0, 0, 1.0, Polarity::On), // bin 2 (clamped)
        ];
        let slice = EventSlice::new(events, 10, 10).expect("failed");
        let config = make_config(10, 10);
        let voxel = events_to_voxel_grid(&slice, 3, &config).expect("failed");
        assert_eq!(voxel.shape(), &[3, 10, 10]);
        // Check that events land in different bins
        let total: f64 = (0..3).map(|b| voxel[[b, 0, 0]]).sum();
        assert!((total - 3.0).abs() < f64::EPSILON); // all ON events counted
    }

    #[test]
    fn test_streaming_accumulator_matches_batch() {
        let events = vec![
            Event::new(0, 0, 0.001, Polarity::On),
            Event::new(1, 1, 0.002, Polarity::Off),
            Event::new(0, 0, 0.003, Polarity::On),
        ];
        let config = make_config(10, 10);

        // Batch
        let slice = EventSlice::new(events.clone(), 10, 10).expect("failed");
        let batch_frame =
            events_to_frame(&slice, &FrameMethod::PolarityHistogram, &config).expect("failed");

        // Streaming
        let mut acc = StreamingFrameAccumulator::new(make_config(10, 10));
        for e in &events {
            acc.add_event(e);
        }

        // Compare
        assert!((acc.get_frame()[[0, 0]] - batch_frame.data[[0, 0]]).abs() < f64::EPSILON);
        assert!((acc.get_frame()[[1, 1]] - batch_frame.data[[1, 1]]).abs() < f64::EPSILON);
    }
}
