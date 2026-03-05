//! Real-time streaming analytics and data processing.
//!
//! This module provides building blocks for constructing streaming analytics
//! pipelines directly on data read from I/O sources:
//!
//! - **aggregators**: Incremental, numerically-stable streaming aggregators
//!   (count, sum, mean/variance, min/max, histogram, top-K).
//! - **windows**: Tumbling, sliding, and session window processors for
//!   partitioning a time-ordered event stream into bounded batches.
//! - **joins**: Hash join and interval join for merging two concurrent streams.
//! - **watermarks**: Event-time progress tracking for out-of-order streams.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::analytics::{TumblingWindow, MeanAgg, StreamAggregator};
//! use scirs2_io::analytics::windows::Record;
//!
//! let mut window = TumblingWindow::new(5.0);
//! let mut agg = MeanAgg::new();
//!
//! for i in 0u64..6 {
//!     let record = Record::new(i as f64, i as f64);
//!     if let Some(completed) = window.add(record) {
//!         for r in &completed {
//!             agg.update(r.value);
//!         }
//!     }
//! }
//! ```

pub mod aggregators;
pub mod joins;
pub mod watermarks;
pub mod windows;

pub use aggregators::{
    CountAgg, HistogramAgg, MeanAgg, MinMaxAgg, StreamAggregator, SumAgg, TopKAgg,
};
pub use joins::{HashJoin, IntervalJoin, SortMergeJoin, StreamJoin};
pub use watermarks::{EventTimeTracker, MultiStreamWatermark, Watermark};
pub use windows::{Record, SessionWindow, SlidingWindow, TumblingWindow, WindowSpec};
