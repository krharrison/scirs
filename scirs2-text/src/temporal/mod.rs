//! Temporal text mining and event extraction.
//!
//! This module provides a comprehensive, rule-based pipeline for extracting
//! temporal information from English text, aligning events on a timeline, and
//! generating narrative summaries.
//!
//! # Sub-modules
//!
//! | Sub-module | Purpose |
//! |---|---|
//! | [`temporal_patterns`] | Compiled regex patterns for temporal expressions |
//! | [`temporal_relations`] | TIMEX3-style extraction, normalisation, relation classification |
//! | [`event_extraction`] | ACE-style event detection with trigger/argument extraction |
//! | [`timeline`] | Timeline construction, constraint propagation, narrative generation |
//!
//! # Quick-start Example
//!
//! ```rust
//! use scirs2_text::temporal::{
//!     event_extraction::extract_events,
//!     temporal_relations::extract_time_expressions,
//!     timeline::{build_timeline, timeline_to_narrative},
//!     temporal_patterns::all_patterns,
//! };
//!
//! let text = "The merger was announced in March 2019. The deal closed on 2019-09-01.";
//!
//! // 1. Extract temporal patterns.
//! let patterns = all_patterns(text);
//! assert!(!patterns.is_empty());
//!
//! // 2. Extract TIMEX3-like time expressions.
//! let timex = extract_time_expressions(text);
//! assert!(!timex.is_empty());
//!
//! // 3. Extract events.
//! let events = extract_events(text);
//!
//! // 4. Build and narrate timeline.
//! let timeline = build_timeline(events, &[]);
//! let narrative = timeline_to_narrative(&timeline);
//! println!("{}", narrative);
//! ```

pub mod event_extraction;
pub mod temporal_patterns;
pub mod temporal_relations;
pub mod timeline;

// ---------------------------------------------------------------------------
// Top-level re-exports for ergonomic access
// ---------------------------------------------------------------------------

// Temporal patterns
pub use temporal_patterns::{
    AbsoluteDatePattern, AnchorPattern, DurationPattern, FrequencyPattern, PatternMatch,
    RelativeTimePattern, TemporalPatternBank, all_patterns, month_name_to_number, word_to_number,
};

// Temporal relations
pub use temporal_relations::{
    TimeExpression, TimeType, TemporalRelation,
    extract_time_expressions, normalize_timex, temporal_ordering,
};

// Event extraction
pub use event_extraction::{
    Argument, ArgumentRole, Event, EventExtractor, EventType,
    argument_detection, event_patterns, extract_events,
};

// Timeline
pub use timeline::{
    NarrativeArc, TemporalConstraint, TimeAnchor, Timeline, TimelineNode,
    build_timeline, extract_narrative_arcs, resolve_temporal_ambiguity, timeline_to_narrative,
};
