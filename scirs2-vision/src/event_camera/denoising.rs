//! Spatiotemporal event denoising filters.
//!
//! Event cameras produce noise events due to transistor mismatch, thermal
//! fluctuations, and junction leakage. This module provides filters to remove
//! isolated noise events while preserving spatiotemporally correlated signal.
//!
//! Available filters:
//!
//! - **Nearest-neighbor filter**: keeps an event only if it has sufficient
//!   supporting events within a spatial and temporal neighborhood.
//! - **Refractory period filter**: suppresses events that arrive too quickly
//!   at the same pixel (mimics biological refractory period).
//! - **Combined filter** ([`denoise`]): applies refractory then NN filter.

use scirs2_core::ndarray::Array2;

use crate::error::{Result, VisionError};

use super::types::{Event, EventProcessingConfig, EventSlice};

/// Configuration for event denoising.
pub struct DenoisingConfig {
    /// Spatial radius in pixels for neighbor search.
    pub spatial_radius: usize,
    /// Temporal window in seconds for neighbor search.
    pub temporal_window: f64,
    /// Minimum number of supporting events to keep an event.
    pub min_support: usize,
    /// Minimum time between consecutive events at the same pixel (seconds).
    pub refractory_period: f64,
}

impl Default for DenoisingConfig {
    fn default() -> Self {
        Self {
            spatial_radius: 1,
            temporal_window: 0.005, // 5 ms
            min_support: 2,
            refractory_period: 0.001, // 1 ms
        }
    }
}

/// Nearest-neighbor filter.
///
/// Keeps an event only if at least `min_support` other events exist within
/// `spatial_radius` pixels and `temporal_window` seconds. Isolated noise
/// events (with no spatio-temporal neighbors) are removed.
pub fn nearest_neighbor_filter(
    events: &EventSlice,
    config: &DenoisingConfig,
) -> Result<EventSlice> {
    let ev = events.events();
    if ev.is_empty() {
        return Err(VisionError::InvalidParameter(
            "No events to filter".to_string(),
        ));
    }

    let radius = config.spatial_radius as i32;
    let tw = config.temporal_window;
    let min_sup = config.min_support;

    let mut kept: Vec<Event> = Vec::with_capacity(ev.len());

    // Events are sorted by timestamp (guaranteed by EventSlice::new).
    // For each event, scan nearby events in time to count spatial neighbors.
    for (i, e) in ev.iter().enumerate() {
        let mut support = 0usize;

        // Search backward
        let mut j = i;
        while j > 0 {
            j -= 1;
            let other = &ev[j];
            if e.timestamp - other.timestamp > tw {
                break;
            }
            let dx = (e.x as i32 - other.x as i32).abs();
            let dy = (e.y as i32 - other.y as i32).abs();
            if dx <= radius && dy <= radius {
                support += 1;
                if support >= min_sup {
                    break;
                }
            }
        }

        if support < min_sup {
            // Search forward
            for other in ev.iter().skip(i + 1) {
                if other.timestamp - e.timestamp > tw {
                    break;
                }
                let dx = (e.x as i32 - other.x as i32).abs();
                let dy = (e.y as i32 - other.y as i32).abs();
                if dx <= radius && dy <= radius {
                    support += 1;
                    if support >= min_sup {
                        break;
                    }
                }
            }
        }

        if support >= min_sup {
            kept.push(*e);
        }
    }

    if kept.is_empty() {
        return Err(VisionError::OperationError(
            "All events were filtered out by nearest-neighbor filter".to_string(),
        ));
    }

    EventSlice::new(kept, events.width(), events.height())
}

/// Refractory period filter.
///
/// For each pixel, suppresses events that arrive within `refractory_period`
/// seconds of the previous event at that pixel. This removes rapid-fire
/// "stuttering" noise while preserving normal event cadence.
pub fn refractory_filter(events: &EventSlice, config: &DenoisingConfig) -> Result<EventSlice> {
    let ev = events.events();
    if ev.is_empty() {
        return Err(VisionError::InvalidParameter(
            "No events to filter".to_string(),
        ));
    }

    let w = events.width() as usize;
    let h = events.height() as usize;
    let refrac = config.refractory_period;

    // Track last event timestamp per pixel
    let mut last_time = vec![f64::NEG_INFINITY; h * w];
    let mut kept: Vec<Event> = Vec::with_capacity(ev.len());

    for e in ev {
        let idx = e.y as usize * w + e.x as usize;
        if e.timestamp - last_time[idx] >= refrac {
            kept.push(*e);
            last_time[idx] = e.timestamp;
        }
    }

    if kept.is_empty() {
        return Err(VisionError::OperationError(
            "All events were filtered out by refractory filter".to_string(),
        ));
    }

    EventSlice::new(kept, events.width(), events.height())
}

/// Combined spatiotemporal denoising filter.
///
/// Applies the refractory period filter first, then the nearest-neighbor
/// filter. This two-stage approach first removes rapid-fire noise, then
/// filters remaining isolated events.
pub fn denoise(events: &EventSlice, config: &DenoisingConfig) -> Result<EventSlice> {
    let after_refrac = refractory_filter(events, config)?;
    nearest_neighbor_filter(&after_refrac, config)
}

/// Estimates which pixels are likely noise sources based on event rate.
///
/// A pixel is flagged as noisy if its event rate is above
/// `mean_rate + 3 * std_rate` (hot pixel) or if it fires exactly once in the
/// entire window (isolated transient).
///
/// Returns a boolean mask `[height, width]` where `true` indicates a noisy pixel.
pub fn estimate_noise_pixels(
    events: &EventSlice,
    config: &EventProcessingConfig,
) -> Result<Array2<bool>> {
    let h = config.height as usize;
    let w = config.width as usize;
    let (t_start, t_end) = events.time_range();
    let duration = t_end - t_start;

    if duration <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Event duration must be positive for noise estimation".to_string(),
        ));
    }

    // Count events per pixel
    let mut counts = Array2::<f64>::zeros((h, w));
    for e in events.events() {
        counts[[e.y as usize, e.x as usize]] += 1.0;
    }

    // Convert to rates
    let rates = &counts / duration;

    // Compute mean and std of non-zero rates
    let non_zero_rates: Vec<f64> = rates.iter().copied().filter(|&r| r > 0.0).collect();

    if non_zero_rates.is_empty() {
        return Ok(Array2::from_elem((h, w), false));
    }

    let n = non_zero_rates.len() as f64;
    let mean_rate = non_zero_rates.iter().sum::<f64>() / n;
    let var_rate = non_zero_rates
        .iter()
        .map(|r| (r - mean_rate).powi(2))
        .sum::<f64>()
        / n;
    let std_rate = var_rate.sqrt();

    let hot_threshold = mean_rate + 3.0 * std_rate;

    let mut noise_mask = Array2::from_elem((h, w), false);
    for y in 0..h {
        for x in 0..w {
            let rate = rates[[y, x]];
            // Hot pixel: abnormally high rate
            if rate > hot_threshold {
                noise_mask[[y, x]] = true;
            }
            // Isolated transient: exactly 1 event (likely noise in a low-activity pixel)
            if counts[[y, x]] == 1.0 && rate < mean_rate * 0.1 {
                noise_mask[[y, x]] = true;
            }
        }
    }

    Ok(noise_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_camera::types::{Event, EventSlice, Polarity};

    #[test]
    fn test_nn_filter_removes_isolated_noise() {
        // Create a cluster of "real" events and one isolated noise event
        let mut events = Vec::new();

        // Cluster at (5,5) - multiple events close in space and time
        for i in 0..5 {
            events.push(Event::new(5, 5, 0.001 * i as f64, Polarity::On));
            events.push(Event::new(6, 5, 0.001 * i as f64 + 0.0001, Polarity::On));
            events.push(Event::new(5, 6, 0.001 * i as f64 + 0.0002, Polarity::On));
        }

        // Isolated noise event far from the cluster
        events.push(Event::new(50, 50, 0.003, Polarity::Off));

        let slice = EventSlice::new(events, 100, 100).expect("failed");
        let config = DenoisingConfig {
            spatial_radius: 2,
            temporal_window: 0.005,
            min_support: 2,
            refractory_period: 0.001,
        };

        let filtered = nearest_neighbor_filter(&slice, &config).expect("failed");

        // The isolated event at (50,50) should be removed
        let has_noise = filtered.events().iter().any(|e| e.x == 50 && e.y == 50);
        assert!(!has_noise, "Isolated noise event should be removed");

        // Clustered events should be mostly preserved
        let cluster_count = filtered
            .events()
            .iter()
            .filter(|e| e.x <= 6 && e.y <= 6)
            .count();
        assert!(cluster_count > 0, "Clustered events should be preserved");
    }

    #[test]
    fn test_nn_filter_preserves_clustered_events() {
        // Dense cluster of events - all should survive
        let mut events = Vec::new();
        for y in 10..15 {
            for x in 10..15 {
                events.push(Event::new(x, y, 0.001, Polarity::On));
                events.push(Event::new(x, y, 0.002, Polarity::Off));
            }
        }

        let slice = EventSlice::new(events, 30, 30).expect("failed");
        let config = DenoisingConfig {
            spatial_radius: 2,
            temporal_window: 0.01,
            min_support: 2,
            refractory_period: 0.0, // disable refractory for this test
        };

        let filtered = nearest_neighbor_filter(&slice, &config).expect("failed");
        // Most events should survive since they're densely clustered
        assert!(
            filtered.len() > slice.len() / 2,
            "Expected most clustered events to survive, got {}/{}",
            filtered.len(),
            slice.len()
        );
    }

    #[test]
    fn test_refractory_filter_suppresses_rapid_fire() {
        // Events at the same pixel with very short intervals
        let events = vec![
            Event::new(5, 5, 0.000, Polarity::On),
            Event::new(5, 5, 0.0001, Polarity::On), // too fast, suppressed
            Event::new(5, 5, 0.0002, Polarity::On), // too fast, suppressed
            Event::new(5, 5, 0.002, Polarity::On),  // enough gap, kept
            Event::new(10, 10, 0.001, Polarity::Off), // different pixel, kept
        ];

        let slice = EventSlice::new(events, 20, 20).expect("failed");
        let config = DenoisingConfig {
            refractory_period: 0.001, // 1 ms refractory
            ..Default::default()
        };

        let filtered = refractory_filter(&slice, &config).expect("failed");

        // Events at (5,5): first at t=0 kept, next two suppressed, t=0.002 kept
        // Event at (10,10): kept
        // Total: 3 events
        assert_eq!(
            filtered.len(),
            3,
            "Expected 3 events after refractory filter, got {}",
            filtered.len()
        );
    }

    #[test]
    fn test_denoise_reduces_event_count() {
        let mut events = Vec::new();

        // Real signal: dense spatiotemporal cluster
        for step in 0..10 {
            let t = step as f64 * 0.002;
            for y in 10..15 {
                for x in 10..15 {
                    events.push(Event::new(x, y, t, Polarity::On));
                }
            }
        }

        // Noise: isolated events scattered randomly
        for i in 0..50 {
            events.push(Event::new(
                (i * 7 + 30) % 80,
                (i * 11 + 40) % 60,
                i as f64 * 0.0004,
                Polarity::Off,
            ));
        }

        let original_count = events.len();
        let slice = EventSlice::new(events, 80, 60).expect("failed");
        let config = DenoisingConfig {
            spatial_radius: 1,
            temporal_window: 0.005,
            min_support: 2,
            refractory_period: 0.001,
        };

        let filtered = denoise(&slice, &config).expect("failed");
        assert!(
            filtered.len() < original_count,
            "Denoising should reduce event count: {} -> {}",
            original_count,
            filtered.len()
        );
    }

    #[test]
    fn test_estimate_noise_pixels() {
        let mut events = Vec::new();

        // Normal pixel activity: many pixels with ~5 events each over 1 second
        for px in 0..20 {
            for step in 0..5 {
                events.push(Event::new(
                    px % 30,
                    (px / 30).min(29),
                    step as f64 * 0.2,
                    Polarity::On,
                ));
            }
        }

        // Hot pixel: extremely high event rate (1000 events in 1 second)
        for step in 0..1000 {
            events.push(Event::new(15, 15, step as f64 * 0.001, Polarity::On));
        }

        let slice = EventSlice::new(events, 30, 30).expect("failed");
        let config = EventProcessingConfig {
            width: 30,
            height: 30,
            ..Default::default()
        };

        let noise_mask = estimate_noise_pixels(&slice, &config).expect("failed");
        // The hot pixel at (15,15) should be flagged (rate ~1000/s vs ~5/s for normal)
        assert!(
            noise_mask[[15, 15]],
            "Hot pixel (15,15) should be flagged as noisy"
        );
    }
}
