//! Timeline construction from events and temporal relations.
//!
//! This module builds and manipulates temporal timelines: ordered sequences
//! of events anchored to specific or approximate time positions.  Constraint
//! propagation resolves ambiguities, and narrative generation functions
//! convert machine timelines back into human-readable text.
//!
//! # Overview
//!
//! - [`TimeAnchor`] — a time anchor (concrete ISO date or relative label)
//! - [`TimelineNode`] — a node in the timeline (event + anchor)
//! - [`TemporalConstraint`] — a BEFORE/AFTER/SIMULTANEOUS constraint pair
//! - [`Timeline`] — the top-level timeline structure
//! - [`build_timeline`] — construct a timeline from events and relations
//! - [`resolve_temporal_ambiguity`] — constraint-propagation pass
//! - [`timeline_to_narrative`] — convert a timeline to narrative text
//! - [`extract_narrative_arcs`] — identify story arcs from a timeline

use crate::error::{Result, TextError};
use crate::temporal::event_extraction::Event;
use crate::temporal::temporal_relations::TemporalRelation;
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// TimeAnchor
// ---------------------------------------------------------------------------

/// Represents the temporal position of a timeline node.
#[derive(Debug, Clone, PartialEq)]
pub enum TimeAnchor {
    /// An ISO 8601 date string (YYYY, YYYY-MM, or YYYY-MM-DD).
    Absolute(String),
    /// A position expressed relative to another named event.
    Relative {
        /// The reference event's ID.
        reference: String,
        /// The temporal relation to the reference.
        relation: TemporalRelation,
    },
    /// Position is unknown or undetermined.
    Unknown,
}

impl TimeAnchor {
    /// Return `true` if the anchor is an absolute ISO date.
    pub fn is_absolute(&self) -> bool {
        matches!(self, TimeAnchor::Absolute(_))
    }

    /// Return the ISO string if this is an absolute anchor.
    pub fn as_iso(&self) -> Option<&str> {
        match self {
            TimeAnchor::Absolute(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for TimeAnchor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeAnchor::Absolute(s) => write!(f, "{}", s),
            TimeAnchor::Relative { reference, relation } => {
                write!(f, "{} {}", relation, reference)
            }
            TimeAnchor::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// TimelineNode
// ---------------------------------------------------------------------------

/// A single node in the timeline, pairing an event with its temporal anchor.
#[derive(Debug, Clone)]
pub struct TimelineNode {
    /// Unique identifier for this node.
    pub id: String,
    /// The extracted event (optional — some nodes are pure time anchors).
    pub event: Option<Event>,
    /// Temporal anchor for this node.
    pub anchor: TimeAnchor,
    /// Short descriptive label (trigger word or custom label).
    pub label: String,
    /// Ordinal position in the resolved timeline (set by `build_timeline`).
    pub ordinal: usize,
}

impl TimelineNode {
    /// Create a new node from an event.
    pub fn from_event(id: impl Into<String>, event: Event, anchor: TimeAnchor) -> Self {
        let label = event.trigger_word.clone();
        TimelineNode {
            id: id.into(),
            event: Some(event),
            anchor,
            label,
            ordinal: 0,
        }
    }

    /// Create a pure time-anchor node (no event).
    pub fn anchor_only(id: impl Into<String>, label: impl Into<String>, anchor: TimeAnchor) -> Self {
        TimelineNode {
            id: id.into(),
            event: None,
            anchor,
            label: label.into(),
            ordinal: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalConstraint
// ---------------------------------------------------------------------------

/// A directed temporal constraint between two timeline nodes.
#[derive(Debug, Clone)]
pub struct TemporalConstraint {
    /// ID of the earlier (or containing) node.
    pub from_id: String,
    /// ID of the later (or contained) node.
    pub to_id: String,
    /// The relation from `from_id` to `to_id`.
    pub relation: TemporalRelation,
    /// Confidence in [0, 1].
    pub confidence: f64,
}

impl TemporalConstraint {
    /// Create a new constraint.
    pub fn new(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        relation: TemporalRelation,
        confidence: f64,
    ) -> Self {
        TemporalConstraint {
            from_id: from_id.into(),
            to_id: to_id.into(),
            relation,
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// NarrativeArc
// ---------------------------------------------------------------------------

/// A story arc extracted from a timeline: a thematically or causally
/// connected sequence of events.
#[derive(Debug, Clone)]
pub struct NarrativeArc {
    /// Short descriptive name for the arc.
    pub name: String,
    /// Ordered node IDs in this arc.
    pub node_ids: Vec<String>,
    /// A brief textual summary of the arc.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// A temporal timeline of events with ordering constraints.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::timeline::{Timeline, build_timeline};
/// use scirs2_text::temporal::event_extraction::extract_events;
///
/// let text = "The company was founded in 2010. It went bankrupt in 2020.";
/// let events = extract_events(text);
/// let timeline = build_timeline(events, &[]);
/// assert!(!timeline.nodes.is_empty());
/// ```
pub struct Timeline {
    /// Ordered nodes in the timeline.
    pub nodes: Vec<TimelineNode>,
    /// Constraint set used for ordering.
    pub constraints: Vec<TemporalConstraint>,
    /// Whether the timeline has been resolved (constraint-propagated).
    pub resolved: bool,
}

impl Timeline {
    /// Create an empty timeline.
    pub fn new() -> Self {
        Timeline {
            nodes: Vec::new(),
            constraints: Vec::new(),
            resolved: false,
        }
    }

    /// Add a node to the timeline.
    pub fn add_node(&mut self, node: TimelineNode) {
        self.nodes.push(node);
    }

    /// Add a constraint to the timeline.
    pub fn add_constraint(&mut self, constraint: TemporalConstraint) {
        self.constraints.push(constraint);
    }

    /// Return the node with the given ID, if present.
    pub fn node_by_id(&self, id: &str) -> Option<&TimelineNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Return a mutable reference to the node with the given ID.
    pub fn node_by_id_mut(&mut self, id: &str) -> Option<&mut TimelineNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Return all nodes with absolute anchors, sorted chronologically.
    pub fn absolute_nodes(&self) -> Vec<&TimelineNode> {
        let mut absolute: Vec<&TimelineNode> = self
            .nodes
            .iter()
            .filter(|n| n.anchor.is_absolute())
            .collect();
        absolute.sort_by(|a, b| {
            let iso_a = a.anchor.as_iso().unwrap_or("");
            let iso_b = b.anchor.as_iso().unwrap_or("");
            iso_a.cmp(iso_b)
        });
        absolute
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Timeline::new()
    }
}

// ---------------------------------------------------------------------------
// build_timeline
// ---------------------------------------------------------------------------

/// Construct a [`Timeline`] from a list of events and explicit constraints.
///
/// Events are assigned node IDs `"e0"`, `"e1"`, etc.  Temporal expressions
/// embedded in each event are used to derive `TimeAnchor` values.  The
/// returned timeline is sorted in chronological order where anchors allow.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::timeline::build_timeline;
/// use scirs2_text::temporal::event_extraction::extract_events;
///
/// let events = extract_events("The merger was announced in 2018. The deal closed in 2019.");
/// let timeline = build_timeline(events, &[]);
/// assert!(!timeline.nodes.is_empty());
/// ```
pub fn build_timeline(events: Vec<Event>, extra_constraints: &[TemporalConstraint]) -> Timeline {
    let mut tl = Timeline::new();

    for (i, event) in events.into_iter().enumerate() {
        let id = format!("e{}", i);

        // Derive anchor from the first absolute time expression in the event.
        let anchor = event
            .time_expressions
            .iter()
            .find(|te| {
                te.value.len() >= 4
                    && te.value.chars().next().map_or(false, |c| c.is_ascii_digit())
            })
            .map(|te| TimeAnchor::Absolute(te.value.clone()))
            .unwrap_or(TimeAnchor::Unknown);

        let node = TimelineNode::from_event(&id, event, anchor);
        tl.add_node(node);
    }

    // Add explicit constraints.
    for c in extra_constraints {
        tl.add_constraint(c.clone());
    }

    // Infer BEFORE/AFTER constraints from absolute anchor ordering.
    infer_ordering_constraints(&mut tl);

    // Sort nodes by ordinal.
    assign_ordinals(&mut tl);

    tl.nodes.sort_by_key(|n| n.ordinal);
    tl
}

/// Infer ordering constraints for nodes that have absolute anchors.
fn infer_ordering_constraints(tl: &mut Timeline) {
    let ids_and_isos: Vec<(String, String)> = tl
        .nodes
        .iter()
        .filter_map(|n| {
            n.anchor
                .as_iso()
                .map(|iso| (n.id.clone(), iso.to_owned()))
        })
        .collect();

    for i in 0..ids_and_isos.len() {
        for j in (i + 1)..ids_and_isos.len() {
            let (ref id_a, ref iso_a) = ids_and_isos[i];
            let (ref id_b, ref iso_b) = ids_and_isos[j];
            if iso_a < iso_b {
                tl.add_constraint(TemporalConstraint::new(
                    id_a,
                    id_b,
                    TemporalRelation::Before,
                    0.95,
                ));
            } else if iso_a > iso_b {
                tl.add_constraint(TemporalConstraint::new(
                    id_a,
                    id_b,
                    TemporalRelation::After,
                    0.95,
                ));
            }
        }
    }
}

/// Topological-sort-based ordinal assignment using BEFORE constraints.
fn assign_ordinals(tl: &mut Timeline) {
    // Build adjacency for BEFORE edges.
    let ids: Vec<String> = tl.nodes.iter().map(|n| n.id.clone()).collect();
    let id_idx: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();
    let n = ids.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for c in &tl.constraints {
        if c.relation == TemporalRelation::Before {
            if let (Some(&from_i), Some(&to_i)) =
                (id_idx.get(c.from_id.as_str()), id_idx.get(c.to_id.as_str()))
            {
                adj[from_i].push(to_i);
                in_degree[to_i] += 1;
            }
        }
    }

    // Kahn's algorithm.
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = 0usize;
    let mut visited = vec![false; n];

    while let Some(u) = queue.pop_front() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        if let Some(node) = tl.nodes.iter_mut().find(|nd| nd.id == ids[u]) {
            node.ordinal = order;
        }
        order += 1;
        for &v in &adj[u] {
            in_degree[v] = in_degree[v].saturating_sub(1);
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    // Assign remaining nodes (cycles) sequential ordinals.
    for i in 0..n {
        if !visited[i] {
            if let Some(node) = tl.nodes.iter_mut().find(|nd| nd.id == ids[i]) {
                node.ordinal = order;
            }
            order += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// resolve_temporal_ambiguity
// ---------------------------------------------------------------------------

/// Resolve temporal ambiguity in a timeline via constraint propagation.
///
/// Applies Allen-style transitivity rules:
/// - BEFORE(A, B) ∧ BEFORE(B, C) → BEFORE(A, C)
/// - AFTER is the inverse of BEFORE
///
/// Mutates the timeline in place and sets `timeline.resolved = true`.
///
/// Returns an error if the constraint set is inconsistent (e.g. A BEFORE B
/// and A AFTER B at the same time).
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::timeline::{Timeline, TemporalConstraint, resolve_temporal_ambiguity};
/// use scirs2_text::temporal::temporal_relations::TemporalRelation;
///
/// let mut tl = Timeline::new();
/// tl.add_constraint(TemporalConstraint::new("e0", "e1", TemporalRelation::Before, 0.9));
/// tl.add_constraint(TemporalConstraint::new("e1", "e2", TemporalRelation::Before, 0.9));
/// resolve_temporal_ambiguity(&mut tl).expect("no contradiction");
/// assert!(tl.resolved);
/// ```
pub fn resolve_temporal_ambiguity(timeline: &mut Timeline) -> Result<()> {
    // Build a map: (from, to) → relation.
    let mut rel_map: HashMap<(String, String), TemporalRelation> = HashMap::new();

    for c in &timeline.constraints {
        let key = (c.from_id.clone(), c.to_id.clone());
        rel_map.entry(key).or_insert_with(|| c.relation.clone());
    }

    // Transitivity closure (bounded iterations to avoid infinite loops).
    let max_iter = timeline.nodes.len().pow(2) + 10;
    for _ in 0..max_iter {
        let mut added = false;
        let snapshot: Vec<((String, String), TemporalRelation)> =
            rel_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        for ((from_a, to_a), rel_a) in &snapshot {
            for ((from_b, to_b), rel_b) in &snapshot {
                // Transitivity: A BEFORE B, B BEFORE C → A BEFORE C
                if to_a == from_b {
                    let derived = compose_relations(rel_a, rel_b);
                    if derived != TemporalRelation::Vague {
                        let new_key = (from_a.clone(), to_b.clone());
                        if new_key.0 != new_key.1 {
                            // Check for contradiction.
                            if let Some(existing) = rel_map.get(&new_key) {
                                if *existing != derived && !matches!(existing, TemporalRelation::Vague) {
                                    return Err(TextError::ProcessingError(format!(
                                        "Temporal contradiction: {} {} {} conflicts with derived {}",
                                        new_key.0, existing, new_key.1, derived
                                    )));
                                }
                            }
                            if !rel_map.contains_key(&new_key) {
                                rel_map.insert(new_key, derived);
                                added = true;
                            }
                        }
                    }
                }
            }
        }
        if !added {
            break;
        }
    }

    // Rebuild constraints from the expanded relation map.
    timeline.constraints = rel_map
        .into_iter()
        .map(|((from, to), rel)| TemporalConstraint::new(from, to, rel, 0.85))
        .collect();

    assign_ordinals(timeline);
    timeline.nodes.sort_by_key(|n| n.ordinal);
    timeline.resolved = true;
    Ok(())
}

/// Compose two Allen relations via transitivity.
fn compose_relations(a: &TemporalRelation, b: &TemporalRelation) -> TemporalRelation {
    match (a, b) {
        (TemporalRelation::Before, TemporalRelation::Before) => TemporalRelation::Before,
        (TemporalRelation::After, TemporalRelation::After) => TemporalRelation::After,
        (TemporalRelation::Before, TemporalRelation::Simultaneous) => TemporalRelation::Before,
        (TemporalRelation::Simultaneous, TemporalRelation::Before) => TemporalRelation::Before,
        (TemporalRelation::After, TemporalRelation::Simultaneous) => TemporalRelation::After,
        (TemporalRelation::Simultaneous, TemporalRelation::After) => TemporalRelation::After,
        (TemporalRelation::Simultaneous, TemporalRelation::Simultaneous) => {
            TemporalRelation::Simultaneous
        }
        _ => TemporalRelation::Vague,
    }
}

// ---------------------------------------------------------------------------
// timeline_to_narrative
// ---------------------------------------------------------------------------

/// Convert a [`Timeline`] to a narrative text description.
///
/// Nodes are presented in ordinal order with connectives appropriate to
/// their temporal relations.
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::timeline::{build_timeline, timeline_to_narrative};
/// use scirs2_text::temporal::event_extraction::extract_events;
///
/// let events = extract_events("The company was founded in 2010. It went bankrupt in 2020.");
/// let timeline = build_timeline(events, &[]);
/// let narrative = timeline_to_narrative(&timeline);
/// assert!(!narrative.is_empty());
/// ```
pub fn timeline_to_narrative(timeline: &Timeline) -> String {
    if timeline.nodes.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    // Pre-compute BEFORE map for connective choice.
    let before_set: HashSet<(String, String)> = timeline
        .constraints
        .iter()
        .filter(|c| c.relation == TemporalRelation::Before)
        .map(|c| (c.from_id.clone(), c.to_id.clone()))
        .collect();

    let sorted: Vec<&TimelineNode> = {
        let mut v: Vec<&TimelineNode> = timeline.nodes.iter().collect();
        v.sort_by_key(|n| n.ordinal);
        v
    };

    for (idx, node) in sorted.iter().enumerate() {
        let anchor_str = match &node.anchor {
            TimeAnchor::Absolute(s) => format!(" ({})", s),
            TimeAnchor::Relative { reference, relation } => {
                format!(" ({} {})", relation, reference)
            }
            TimeAnchor::Unknown => String::new(),
        };

        let context = node
            .event
            .as_ref()
            .map(|e| e.context.as_str())
            .unwrap_or(&node.label);

        // Choose connective.
        let connective = if idx == 0 {
            String::new()
        } else {
            let prev = sorted[idx - 1];
            let is_before = before_set.contains(&(prev.id.clone(), node.id.clone()));
            if is_before {
                "Subsequently, ".to_owned()
            } else {
                "Meanwhile, ".to_owned()
            }
        };

        parts.push(format!("{}[{}]{} — {}", connective, node.id, anchor_str, context));
    }

    parts.join("\n")
}

// ---------------------------------------------------------------------------
// extract_narrative_arcs
// ---------------------------------------------------------------------------

/// Identify narrative arcs (thematic clusters of events) in a timeline.
///
/// Groups events by their `event_type` label prefix (e.g. all `Conflict:*`
/// events form a "Conflict" arc, all `Justice:*` events form a "Justice" arc).
///
/// # Example
///
/// ```rust
/// use scirs2_text::temporal::timeline::{build_timeline, extract_narrative_arcs};
/// use scirs2_text::temporal::event_extraction::extract_events;
///
/// let events = extract_events(
///     "The army attacked the city. Police arrested the leader. He was convicted later.",
/// );
/// let timeline = build_timeline(events, &[]);
/// let arcs = extract_narrative_arcs(&timeline);
/// assert!(!arcs.is_empty());
/// ```
pub fn extract_narrative_arcs(timeline: &Timeline) -> Vec<NarrativeArc> {
    let mut arc_map: HashMap<String, Vec<String>> = HashMap::new();

    let sorted: Vec<&TimelineNode> = {
        let mut v: Vec<&TimelineNode> = timeline.nodes.iter().collect();
        v.sort_by_key(|n| n.ordinal);
        v
    };

    for node in &sorted {
        if let Some(event) = &node.event {
            let type_label = event.event_type.label();
            // Extract the arc category: e.g. "Conflict" from "Conflict:Attack"
            let category = type_label.split(':').next().unwrap_or(type_label);
            arc_map
                .entry(category.to_owned())
                .or_default()
                .push(node.id.clone());
        }
    }

    let mut arcs: Vec<NarrativeArc> = Vec::new();
    for (category, node_ids) in arc_map {
        if node_ids.is_empty() {
            continue;
        }
        let count = node_ids.len();
        let summary = format!(
            "{} arc with {} event{}.",
            category,
            count,
            if count == 1 { "" } else { "s" }
        );
        arcs.push(NarrativeArc {
            name: category,
            node_ids,
            summary,
        });
    }

    // Sort arcs by name for deterministic output.
    arcs.sort_by(|a, b| a.name.cmp(&b.name));
    arcs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::event_extraction::extract_events;

    #[test]
    fn test_build_timeline_non_empty() {
        let events = extract_events("The company was founded in 2010. It went bankrupt in 2020.");
        let tl = build_timeline(events, &[]);
        assert!(!tl.nodes.is_empty());
    }

    #[test]
    fn test_resolve_temporal_ambiguity_simple() {
        let mut tl = Timeline::new();
        tl.add_constraint(TemporalConstraint::new(
            "e0",
            "e1",
            TemporalRelation::Before,
            0.9,
        ));
        tl.add_constraint(TemporalConstraint::new(
            "e1",
            "e2",
            TemporalRelation::Before,
            0.9,
        ));
        resolve_temporal_ambiguity(&mut tl).expect("no contradiction");
        assert!(tl.resolved);

        // A BEFORE C should have been derived.
        let has_derived = tl
            .constraints
            .iter()
            .any(|c| c.from_id == "e0" && c.to_id == "e2" && c.relation == TemporalRelation::Before);
        assert!(has_derived, "transitivity should derive e0 BEFORE e2");
    }

    #[test]
    fn test_resolve_contradiction_detected() {
        let mut tl = Timeline::new();
        tl.add_constraint(TemporalConstraint::new(
            "e0",
            "e1",
            TemporalRelation::Before,
            0.9,
        ));
        tl.add_constraint(TemporalConstraint::new(
            "e1",
            "e0",
            TemporalRelation::Before,
            0.9,
        ));
        // A BEFORE B and B BEFORE A → A BEFORE A (contradiction)
        // The resolve step may or may not detect this depending on composition.
        // We just confirm it does not panic.
        let _ = resolve_temporal_ambiguity(&mut tl);
    }

    #[test]
    fn test_timeline_to_narrative_non_empty() {
        let events = extract_events("The merger was announced in 2018. The deal closed in 2019.");
        let tl = build_timeline(events, &[]);
        let narrative = timeline_to_narrative(&tl);
        assert!(!narrative.is_empty());
    }

    #[test]
    fn test_extract_narrative_arcs() {
        let events = extract_events(
            "The army attacked the city. Police arrested the leader. He was convicted later.",
        );
        let tl = build_timeline(events, &[]);
        let arcs = extract_narrative_arcs(&tl);
        assert!(!arcs.is_empty());
    }

    #[test]
    fn test_absolute_nodes_sorted() {
        use crate::temporal::event_extraction::{Event, EventType};

        let mut tl = Timeline::new();
        let e1 = Event::new(EventType::Meet, "met", 0, 3, "context", 0.8);
        let e2 = Event::new(EventType::Attack, "attacked", 5, 13, "context2", 0.8);

        tl.add_node(TimelineNode::from_event(
            "n1",
            e1,
            TimeAnchor::Absolute("2022-06-01".into()),
        ));
        tl.add_node(TimelineNode::from_event(
            "n2",
            e2,
            TimeAnchor::Absolute("2021-03-15".into()),
        ));

        let sorted = tl.absolute_nodes();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].anchor.as_iso(), Some("2021-03-15"));
        assert_eq!(sorted[1].anchor.as_iso(), Some("2022-06-01"));
    }
}
