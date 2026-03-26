//! Event camera (Dynamic Vision Sensor) processing module.
//!
//! This module provides support for event-based vision processing, including:
//!
//! - **Event types**: Core data structures for DVS events and event slices
//! - **Event-to-frame conversion**: Histogram, time surface, exponential decay, voxel grid
//! - **Event-based optical flow**: Local plane fitting, time surface matching, contrast maximization
//! - **Spatiotemporal denoising**: Nearest-neighbor, refractory period, and combined filters
//!
//! # Overview
//!
//! Event cameras (Dynamic Vision Sensors) output asynchronous events whenever a pixel
//! detects a brightness change. Each event encodes the pixel coordinates, a timestamp
//! (microsecond precision), and a polarity (brightness increase or decrease). This
//! module provides algorithms to process these events for downstream vision tasks.

pub mod conversion;
pub mod denoising;
pub mod optical_flow;
pub mod types;

pub use conversion::{
    events_to_frame, events_to_polarity_frames, events_to_time_surface, events_to_voxel_grid,
    FrameMethod, StreamingFrameAccumulator,
};
pub use denoising::{
    denoise, estimate_noise_pixels, nearest_neighbor_filter, refractory_filter, DenoisingConfig,
};
pub use optical_flow::{
    contrast_maximization, local_plane_fitting, time_surface_flow, FlowField, OpticalFlowConfig,
};
pub use types::{Event, EventFrame, EventProcessingConfig, EventSlice, Polarity};
