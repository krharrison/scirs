//! Arc-Eager transition system (Nivre 2003).
//!
//! Transitions:
//! * **SHIFT**      – push the front of the buffer onto the stack.
//! * **REDUCE**     – pop the stack (only when top has a head).
//! * **LEFT-ARC(l)** – add arc `buffer[0] → stack[top]`, pop stack.
//! * **RIGHT-ARC(l)** – add arc `stack[top] → buffer[0]`, push buffer[0].
//!
//! The arc-eager system can attach arcs sooner than arc-standard,
//! often requiring fewer transitions and enabling O(n) parsing.

use super::graph::{DepLabel, DependencyGraph};

// ---------------------------------------------------------------------------
// Transition type
// ---------------------------------------------------------------------------

/// Transitions in the arc-eager system.
#[derive(Debug, Clone)]
pub enum ArcEagerTransition {
    /// Move the front of the buffer onto the stack.
    Shift,
    /// Pop the stack top (only valid when top already has a head).
    Reduce,
    /// Add arc from buffer front to stack top and pop the stack.
    LeftArc(DepLabel),
    /// Add arc from stack top to buffer front and push buffer front.
    RightArc(DepLabel),
}

impl std::fmt::Display for ArcEagerTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shift          => write!(f, "SHIFT"),
            Self::Reduce         => write!(f, "REDUCE"),
            Self::LeftArc(l)    => write!(f, "LEFT-ARC({})", l),
            Self::RightArc(l)   => write!(f, "RIGHT-ARC({})", l),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parser state for the arc-eager system.
///
/// Token indices are 1-based (0 = virtual ROOT).
#[derive(Debug, Clone)]
pub struct ArcEagerConfig {
    /// Parser stack of token indices (0 = virtual ROOT).
    pub stack: Vec<usize>,
    /// Remaining unprocessed token indices.
    pub buffer: Vec<usize>,
    /// Accumulated dependency arcs as `(head, dependent, label)` triples.
    pub arcs: Vec<(usize, usize, DepLabel)>,
    /// Whether each token already has a head assigned.
    pub has_head: Vec<bool>,
    /// Total number of real tokens (excluding virtual ROOT).
    pub n_tokens: usize,
}

impl ArcEagerConfig {
    /// Initialise with ROOT on the stack and all tokens in the buffer.
    pub fn new(n_tokens: usize) -> Self {
        Self {
            stack: vec![0],
            buffer: (1..=n_tokens).collect(),
            arcs: Vec::new(),
            has_head: vec![false; n_tokens + 1], // 0 = ROOT never needs a head
            n_tokens,
        }
    }

    /// The configuration is terminal when the buffer is empty.
    pub fn is_terminal(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Top of the stack, if any.
    pub fn stack_top(&self) -> Option<usize> {
        self.stack.last().copied()
    }

    /// Front of the buffer, if any.
    pub fn buffer_front(&self) -> Option<usize> {
        self.buffer.first().copied()
    }

    /// Apply a transition; returns `true` on success.
    pub fn apply(&mut self, t: &ArcEagerTransition) -> bool {
        match t {
            ArcEagerTransition::Shift => {
                match self.buffer_front() {
                    None => false,
                    Some(w) => {
                        self.buffer.remove(0);
                        self.stack.push(w);
                        true
                    }
                }
            }

            ArcEagerTransition::Reduce => {
                match self.stack_top() {
                    None | Some(0) => false, // cannot reduce ROOT
                    Some(top) => {
                        if !self.has_head[top] {
                            // Precondition: stack top must already have a head.
                            return false;
                        }
                        self.stack.pop();
                        true
                    }
                }
            }

            ArcEagerTransition::LeftArc(label) => {
                let top = match self.stack_top() { None => return false, Some(t) => t };
                let front = match self.buffer_front() { None => return false, Some(f) => f };
                if top == 0 { return false; } // ROOT cannot be a dependent
                if self.has_head[top] { return false; } // already has a head
                // arc: front → top
                self.arcs.push((front, top, label.clone()));
                self.has_head[top] = true;
                self.stack.pop();
                true
            }

            ArcEagerTransition::RightArc(label) => {
                let top = match self.stack_top() { None => return false, Some(t) => t };
                let front = match self.buffer_front() { None => return false, Some(f) => f };
                // arc: top → front
                self.arcs.push((top, front, label.clone()));
                self.has_head[front] = true;
                // Push front onto stack (it stays in buffer conceptually, but we
                // immediately shift it).
                self.buffer.remove(0);
                self.stack.push(front);
                true
            }
        }
    }

    /// Enumerate precondition-valid transitions.
    pub fn legal_transitions(&self) -> Vec<ArcEagerTransition> {
        let mut legal = Vec::new();
        if !self.buffer.is_empty() {
            legal.push(ArcEagerTransition::Shift);
        }
        if let Some(top) = self.stack_top() {
            if top != 0 && self.has_head[top] {
                legal.push(ArcEagerTransition::Reduce);
            }
            if let Some(_front) = self.buffer_front() {
                if top != 0 && !self.has_head[top] {
                    legal.push(ArcEagerTransition::LeftArc(DepLabel::Dep));
                }
                legal.push(ArcEagerTransition::RightArc(DepLabel::Dep));
            }
        }
        legal
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Arc-Eager dependency parser with a rule-based oracle.
///
/// The arc-eager system can attach arcs sooner than arc-standard,
/// often requiring fewer transitions and enabling O(n) parsing with a learned model.
/// This implementation uses a rule-based heuristic oracle.
pub struct ArcEagerParser;

impl ArcEagerParser {
    /// Create a new arc-eager parser instance.
    pub fn new() -> Self {
        Self
    }

    /// Parse `tokens` with their `pos_tags`.
    ///
    /// Uses the arc-eager transition system with a heuristic oracle.
    pub fn parse(&self, tokens: &[String], pos_tags: &[String]) -> DependencyGraph {
        let n = tokens.len();
        if n == 0 {
            return DependencyGraph::new(Vec::new(), Vec::new());
        }

        let mut config = ArcEagerConfig::new(n);
        let mut graph  = DependencyGraph::new(tokens.to_vec(), pos_tags.to_vec());

        let max_steps = 4 * n + 10;
        for _ in 0..max_steps {
            if config.is_terminal() { break; }
            let trans = self.oracle(&config, pos_tags);
            if !config.apply(&trans) {
                // Fallback chain.
                if !config.buffer.is_empty() {
                    config.apply(&ArcEagerTransition::Shift);
                } else if let Some(top) = config.stack_top() {
                    if top != 0 {
                        // Force right-arc to drain.
                        config.arcs.push((0, top, DepLabel::Root));
                        config.has_head[top] = true;
                        config.stack.pop();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Populate graph.
        for (head, dep, label) in &config.arcs {
            graph.add_arc(*head, *dep, label.clone(), 1.0);
        }

        // Attach any un-headed token to ROOT.
        for i in 1..=n {
            if graph.head_of(i).is_none() {
                graph.add_arc(0, i, DepLabel::Root, 0.5);
            }
        }

        graph
    }

    fn oracle(&self, config: &ArcEagerConfig, pos_tags: &[String]) -> ArcEagerTransition {
        let top   = config.stack_top();
        let front = config.buffer_front();

        let pos = |idx: Option<usize>| -> &str {
            match idx {
                None | Some(0) => "ROOT",
                Some(i) => pos_tags.get(i - 1).map(|s| s.as_str()).unwrap_or("_"),
            }
        };

        let pt = pos(top);
        let pf = pos(front);

        // If top has a head and it no longer needs more dependents, reduce.
        if let Some(t) = top {
            if t != 0 && config.has_head[t] && front.is_none() {
                return ArcEagerTransition::Reduce;
            }
        }

        if top.is_none() || matches!(top, Some(0)) {
            return ArcEagerTransition::Shift;
        }

        // Punct on front attaches as left-arc to top.
        if is_punct(pf) {
            if !config.has_head[front.unwrap_or(0)] {
                return ArcEagerTransition::LeftArc(DepLabel::Punct);
            }
        }

        // Det/Adj in buffer, noun on stack → left-arc.
        if (is_det(pf) || is_adj(pf)) && is_noun(pt) {
            // Not quite right for arc-eager; prefer shift to get more context.
        }

        // Det on stack, noun in buffer → right-arc wouldn't help; shift.
        if is_det(pt) && is_noun(pf) {
            return ArcEagerTransition::Shift;
        }

        // Verb on stack, noun in buffer → right-arc (object).
        if is_verb(pt) && is_noun(pf) {
            return ArcEagerTransition::RightArc(DepLabel::Obj);
        }

        // Noun on stack, verb in buffer → need more context; shift.
        if is_noun(pt) && is_verb(pf) {
            return ArcEagerTransition::Shift;
        }

        // Stack top has a head → try reduce.
        if let Some(t) = top {
            if t != 0 && config.has_head[t] {
                return ArcEagerTransition::Reduce;
            }
        }

        // Default: shift.
        if !config.buffer.is_empty() {
            ArcEagerTransition::Shift
        } else {
            ArcEagerTransition::RightArc(DepLabel::Dep)
        }
    }
}

impl Default for ArcEagerParser {
    fn default() -> Self {
        Self::new()
    }
}

// POS helpers (mirrors arc_standard).
fn is_punct(pos: &str) -> bool {
    matches!(pos, "PUNCT" | "." | "," | ":" | ";" | "!" | "?")
        || pos.starts_with("PUNCT")
}
fn is_det(pos: &str) -> bool   { matches!(pos, "DT" | "det" | "DET") }
fn is_noun(pos: &str) -> bool  {
    pos.starts_with("NN") || matches!(pos, "noun" | "NOUN" | "PROPN")
}
fn is_adj(pos: &str) -> bool   { pos.starts_with("JJ") || matches!(pos, "adj" | "ADJ") }
fn is_verb(pos: &str) -> bool  { pos.starts_with('V') || matches!(pos, "verb" | "VERB") }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_eager_parse() {
        let tokens = ["The", "cat", "sat"].map(String::from).to_vec();
        let pos    = ["DT", "NN", "VBD"].map(String::from).to_vec();
        let parser = ArcEagerParser::new();
        let graph  = parser.parse(&tokens, &pos);
        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(graph.head_of(i).is_some(), "token {} has no head", i);
        }
    }

    #[test]
    fn test_arc_eager_empty() {
        let parser = ArcEagerParser::new();
        let g = parser.parse(&[], &[]);
        assert_eq!(g.n_tokens, 0);
    }

    #[test]
    fn test_config_shift() {
        let mut cfg = ArcEagerConfig::new(3);
        assert!(!cfg.is_terminal());
        cfg.apply(&ArcEagerTransition::Shift);
        assert_eq!(cfg.stack.len(), 2); // ROOT + 1
        assert_eq!(cfg.buffer.len(), 2);
    }

    #[test]
    fn test_config_left_arc() {
        let mut cfg = ArcEagerConfig::new(3);
        cfg.apply(&ArcEagerTransition::Shift); // stack=[0,1], buffer=[2,3]
        // LEFT-ARC: buffer[0]=2 → stack[top]=1
        let ok = cfg.apply(&ArcEagerTransition::LeftArc(DepLabel::Det));
        assert!(ok);
        assert!(cfg.has_head[1]);
        assert_eq!(cfg.stack, vec![0]); // 1 was popped
    }

    #[test]
    fn test_config_right_arc() {
        let mut cfg = ArcEagerConfig::new(3);
        cfg.apply(&ArcEagerTransition::Shift); // stack=[0,1], buffer=[2,3]
        // RIGHT-ARC: stack[top]=1 → buffer[0]=2
        let ok = cfg.apply(&ArcEagerTransition::RightArc(DepLabel::Obj));
        assert!(ok);
        assert!(cfg.has_head[2]);
        assert_eq!(cfg.stack.last(), Some(&2)); // 2 pushed onto stack
    }

    #[test]
    fn test_reduce_precondition() {
        let mut cfg = ArcEagerConfig::new(2);
        cfg.apply(&ArcEagerTransition::Shift); // stack=[0,1]
        // Cannot reduce 1 because it has no head yet.
        let ok = cfg.apply(&ArcEagerTransition::Reduce);
        assert!(!ok);
    }
}
