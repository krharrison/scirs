//! Arc-Standard transition-based dependency parser (Nivre 2004).
//!
//! Transitions:
//! * **SHIFT** – move the front of the buffer to the top of the stack.
//! * **LEFT-ARC(l)** – add arc `stack[top] → stack[top-1]` with label `l`,
//!   then pop `stack[top-1]`.
//! * **RIGHT-ARC(l)** – add arc `stack[top-1] → stack[top]` with label `l`,
//!   then pop `stack[top]`.
//!
//! This implementation includes a rule-based oracle that approximates a
//! left-to-right greedy parser without a learned model.

use super::graph::{DepLabel, DependencyGraph};

// ---------------------------------------------------------------------------
// Transition system
// ---------------------------------------------------------------------------

/// A transition in the arc-standard system.
#[derive(Debug, Clone)]
pub enum Transition {
    /// Move the front of the buffer to the top of the stack.
    Shift,
    /// Add arc from stack top to second-top with the given label, pop second-top.
    LeftArc(DepLabel),
    /// Add arc from second-top to stack top with the given label, pop stack top.
    RightArc(DepLabel),
}

impl std::fmt::Display for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shift          => write!(f, "SHIFT"),
            Self::LeftArc(l)    => write!(f, "LEFT-ARC({})", l),
            Self::RightArc(l)   => write!(f, "RIGHT-ARC({})", l),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser configuration
// ---------------------------------------------------------------------------

/// Complete parser state: stack, buffer, and accumulated arcs.
///
/// Token indices are 1-based (0 = virtual ROOT, 1..=n_tokens = real tokens).
#[derive(Debug, Clone)]
pub struct ArcStandardConfig {
    /// Parser stack of token indices (0 = virtual ROOT).
    pub stack: Vec<usize>,
    /// Remaining unprocessed token indices.
    pub buffer: Vec<usize>,
    /// Accumulated dependency arcs as `(head, dependent, label)` triples.
    pub arcs: Vec<(usize, usize, DepLabel)>,
    /// Total number of real tokens (excluding virtual ROOT).
    pub n_tokens: usize,
}

impl ArcStandardConfig {
    /// Initialise a configuration with ROOT on the stack and all tokens in the buffer.
    pub fn new(n_tokens: usize) -> Self {
        let buffer: Vec<usize> = (1..=n_tokens).collect();
        Self {
            stack: vec![0],
            buffer,
            arcs: Vec::new(),
            n_tokens,
        }
    }

    /// The configuration is terminal when the buffer is empty and only ROOT
    /// remains on the stack.
    pub fn is_terminal(&self) -> bool {
        self.buffer.is_empty() && self.stack.len() == 1
    }

    /// Apply a transition and return `true` on success.
    pub fn apply(&mut self, t: &Transition) -> bool {
        match t {
            Transition::Shift => {
                if self.buffer.is_empty() {
                    return false;
                }
                let word = self.buffer.remove(0);
                self.stack.push(word);
                true
            }

            Transition::LeftArc(label) => {
                let n = self.stack.len();
                if n < 2 {
                    return false;
                }
                let dep  = self.stack[n - 2];
                let head = self.stack[n - 1];
                if dep == 0 {
                    // Cannot make ROOT a dependent.
                    return false;
                }
                self.arcs.push((head, dep, label.clone()));
                self.stack.remove(n - 2);
                true
            }

            Transition::RightArc(label) => {
                let n = self.stack.len();
                if n < 2 {
                    return false;
                }
                let head = self.stack[n - 2];
                let dep  = self.stack[n - 1];
                self.arcs.push((head, dep, label.clone()));
                self.stack.pop();
                true
            }
        }
    }

    /// Enumerate legal transitions from this configuration.
    pub fn legal_transitions(&self) -> Vec<Transition> {
        let mut legal = Vec::new();
        if !self.buffer.is_empty() {
            legal.push(Transition::Shift);
        }
        if self.stack.len() >= 2 {
            // Allow all UD labels; the oracle will choose a specific one.
            for label in DepLabel::all_basic() {
                legal.push(Transition::LeftArc(label.clone()));
                legal.push(Transition::RightArc(label.clone()));
            }
        }
        legal
    }

    /// Stack item at depth `d` from the top (0 = top), if it exists.
    pub fn stack_top(&self, d: usize) -> Option<usize> {
        let n = self.stack.len();
        if d < n { Some(self.stack[n - 1 - d]) } else { None }
    }

    /// Front of the buffer, if non-empty.
    pub fn buffer_front(&self) -> Option<usize> {
        self.buffer.first().copied()
    }
}

// Extend DepLabel with a helper to enumerate the basic labels.
impl DepLabel {
    pub(crate) fn all_basic() -> Vec<DepLabel> {
        vec![
            DepLabel::Root, DepLabel::Subj, DepLabel::Obj, DepLabel::Iobj,
            DepLabel::Nmod, DepLabel::Amod, DepLabel::Advmod, DepLabel::Aux,
            DepLabel::Det, DepLabel::Case, DepLabel::Punct, DepLabel::Conj,
            DepLabel::Cc, DepLabel::Mark, DepLabel::Dep,
        ]
    }
}

// ---------------------------------------------------------------------------
// Rule-based oracle
// ---------------------------------------------------------------------------

/// A simple rule-based arc-standard parser that uses POS patterns as its
/// oracle.  It is not learned but provides a reasonable approximation for
/// basic English sentences.
pub struct ArcStandardParser;

impl ArcStandardParser {
    /// Create a new arc-standard parser instance.
    pub fn new() -> Self {
        Self
    }

    /// Parse `tokens` with their `pos_tags` and return a `DependencyGraph`.
    pub fn parse(&self, tokens: &[String], pos_tags: &[String]) -> DependencyGraph {
        let n = tokens.len();
        if n == 0 {
            return DependencyGraph::new(Vec::new(), Vec::new());
        }

        let mut config = ArcStandardConfig::new(n);
        let mut graph  = DependencyGraph::new(tokens.to_vec(), pos_tags.to_vec());

        // Run oracle until terminal or stuck.
        let max_steps = 3 * n + 10;
        for _ in 0..max_steps {
            if config.is_terminal() {
                break;
            }
            let trans = self.oracle(&config, pos_tags);
            if !config.apply(&trans) {
                // Fallback: try SHIFT.
                if !config.buffer.is_empty() {
                    config.apply(&Transition::Shift);
                } else if config.stack.len() >= 2 {
                    config.apply(&Transition::RightArc(DepLabel::Dep));
                } else {
                    break;
                }
            }
        }

        // Drain any remaining buffer with shifts and right-arcs.
        while !config.buffer.is_empty() {
            config.apply(&Transition::Shift);
        }
        while config.stack.len() >= 2 {
            config.apply(&Transition::RightArc(DepLabel::Dep));
        }

        // Populate graph from accumulated arcs.
        for (head, dep, label) in &config.arcs {
            graph.add_arc(*head, *dep, label.clone(), 1.0);
        }

        // Any token that still has no head gets attached to ROOT.
        for i in 1..=n {
            if graph.head_of(i).is_none() {
                graph.add_arc(0, i, DepLabel::Root, 0.5);
            }
        }

        graph
    }

    /// Heuristic oracle: choose a transition based on POS context.
    fn oracle(&self, config: &ArcStandardConfig, pos_tags: &[String]) -> Transition {
        let s0 = config.stack_top(0); // top of stack
        let s1 = config.stack_top(1); // second from top

        // Helper closure: get POS for a token index (1-based, 0 = ROOT).
        let pos = |idx: Option<usize>| -> &str {
            match idx {
                None | Some(0) => "ROOT",
                Some(i) => pos_tags.get(i - 1).map(|s| s.as_str()).unwrap_or("_"),
            }
        };

        let p0 = pos(s0);
        let p1 = pos(s1);

        if s1.is_none() {
            // Only ROOT on stack — must SHIFT if possible.
            return Transition::Shift;
        }

        // ---- POS-based heuristic rules ----

        // Punctuation on top: attach to the word beneath it.
        if is_punct(p0) {
            return Transition::LeftArc(DepLabel::Punct);
        }

        // Determiner left of a noun.
        if is_det(p1) && is_noun(p0) {
            return Transition::LeftArc(DepLabel::Det);
        }

        // Adjective left of a noun.
        if is_adj(p1) && is_noun(p0) {
            return Transition::LeftArc(DepLabel::Amod);
        }

        // Adverb left of verb/adjective.
        if is_adv(p1) && (is_verb(p0) || is_adj(p0)) {
            return Transition::LeftArc(DepLabel::Advmod);
        }

        // Auxiliary left of a verb.
        if is_aux(p1) && is_verb(p0) {
            return Transition::LeftArc(DepLabel::Aux);
        }

        // Preposition/subordinator on top: shift.
        if is_prep(p0) || is_sub(p0) {
            if !config.buffer.is_empty() {
                return Transition::Shift;
            }
        }

        // Noun on top following a verb: right-arc as subject.
        if is_verb(p1) && is_noun(p0) {
            return Transition::RightArc(DepLabel::Obj);
        }

        // Verb on top with noun beneath: right-arc as object.
        if is_noun(p1) && is_verb(p0) {
            return Transition::LeftArc(DepLabel::Subj);
        }

        // Coordinating conjunction on stack.
        if is_cc(p0) {
            if !config.buffer.is_empty() {
                return Transition::Shift;
            }
            return Transition::RightArc(DepLabel::Cc);
        }

        // Default: shift if buffer is non-empty, else reduce.
        if !config.buffer.is_empty() {
            Transition::Shift
        } else {
            Transition::RightArc(DepLabel::Dep)
        }
    }
}

impl Default for ArcStandardParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// POS category helpers
// ---------------------------------------------------------------------------

fn is_punct(pos: &str) -> bool {
    matches!(pos, "PUNCT" | "." | "," | ":" | ";" | "!" | "?")
        || pos.starts_with("PUNCT")
}

fn is_det(pos: &str) -> bool {
    matches!(pos, "DT" | "det" | "DET")
}

fn is_noun(pos: &str) -> bool {
    pos.starts_with("NN")
        || matches!(pos, "noun" | "NOUN" | "PROPN" | "NNP" | "NNPS" | "NNS")
}

fn is_adj(pos: &str) -> bool {
    pos.starts_with("JJ")
        || matches!(pos, "adj" | "ADJ")
}

fn is_adv(pos: &str) -> bool {
    pos.starts_with("RB")
        || matches!(pos, "adv" | "ADV")
}

fn is_verb(pos: &str) -> bool {
    pos.starts_with('V')
        || matches!(pos, "verb" | "VERB" | "AUX")
}

fn is_aux(pos: &str) -> bool {
    matches!(pos, "MD" | "aux" | "AUX")
}

fn is_prep(pos: &str) -> bool {
    matches!(pos, "IN" | "prep" | "ADP")
}

fn is_sub(pos: &str) -> bool {
    matches!(pos, "IN" | "mark" | "SCONJ")
}

fn is_cc(pos: &str) -> bool {
    matches!(pos, "CC" | "cc" | "CCONJ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_terminal() {
        let mut cfg = ArcStandardConfig::new(2);
        assert!(!cfg.is_terminal());
        cfg.apply(&Transition::Shift);
        cfg.apply(&Transition::Shift);
        cfg.apply(&Transition::RightArc(DepLabel::Subj));
        // Buffer empty, stack has [0, ...] — should be terminal after one more right-arc.
        cfg.apply(&Transition::RightArc(DepLabel::Root));
        assert!(cfg.is_terminal());
    }

    #[test]
    fn test_left_arc_cannot_attach_root_as_dep() {
        let mut cfg = ArcStandardConfig::new(2);
        // stack = [0], buffer = [1,2]
        cfg.apply(&Transition::Shift); // stack = [0,1]
        // LEFT-ARC would make 0 (ROOT) a dependent — should fail.
        let before_arcs = cfg.arcs.len();
        let ok = cfg.apply(&Transition::LeftArc(DepLabel::Dep));
        assert!(!ok);
        assert_eq!(cfg.arcs.len(), before_arcs);
    }

    #[test]
    fn test_parse_simple_sentence() {
        let tokens  = ["The", "cat", "sat"].map(String::from).to_vec();
        let pos     = ["DT", "NN", "VBD"].map(String::from).to_vec();
        let parser  = ArcStandardParser::new();
        let graph   = parser.parse(&tokens, &pos);

        // Every token should have exactly one head.
        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(
                graph.head_of(i).is_some(),
                "token {} has no head",
                i
            );
        }
    }

    #[test]
    fn test_parse_empty() {
        let parser = ArcStandardParser::new();
        let g = parser.parse(&[], &[]);
        assert_eq!(g.n_tokens, 0);
    }

    #[test]
    fn test_legal_transitions() {
        let cfg = ArcStandardConfig::new(3);
        let legal = cfg.legal_transitions();
        // Only SHIFT is legal with stack = [ROOT] and buffer non-empty.
        assert!(legal.iter().any(|t| matches!(t, Transition::Shift)));
    }
}
