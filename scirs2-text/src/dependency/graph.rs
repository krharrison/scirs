//! Dependency graph representation.
//!
//! Provides `DependencyGraph`, `DependencyArc`, and `DepLabel` types
//! for encoding dependency parse trees in Universal Dependencies style.

use std::collections::VecDeque;

/// Dependency relation labels following Universal Dependencies (UD) conventions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DepLabel {
    /// Root of the sentence.
    Root,
    /// Nominal subject (nsubj).
    Subj,
    /// Direct object.
    Obj,
    /// Indirect object.
    Iobj,
    /// Clausal subject.
    Csubj,
    /// Clausal complement.
    Ccomp,
    /// Open clausal complement.
    Xcomp,
    /// Nominal modifier.
    Nmod,
    /// Adjectival modifier.
    Amod,
    /// Adverbial modifier.
    Advmod,
    /// Auxiliary verb.
    Aux,
    /// Determiner.
    Det,
    /// Case marker (preposition, postposition).
    Case,
    /// Punctuation.
    Punct,
    /// Conjunct.
    Conj,
    /// Coordinating conjunction.
    Cc,
    /// Subordinating conjunction or complementizer.
    Mark,
    /// Unspecified dependency.
    Dep,
    /// Non-standard or extended label.
    Other(String),
}

impl std::fmt::Display for DepLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Root   => write!(f, "root"),
            Self::Subj   => write!(f, "nsubj"),
            Self::Obj    => write!(f, "obj"),
            Self::Iobj   => write!(f, "iobj"),
            Self::Csubj  => write!(f, "csubj"),
            Self::Ccomp  => write!(f, "ccomp"),
            Self::Xcomp  => write!(f, "xcomp"),
            Self::Nmod   => write!(f, "nmod"),
            Self::Amod   => write!(f, "amod"),
            Self::Advmod => write!(f, "advmod"),
            Self::Aux    => write!(f, "aux"),
            Self::Det    => write!(f, "det"),
            Self::Case   => write!(f, "case"),
            Self::Punct  => write!(f, "punct"),
            Self::Conj   => write!(f, "conj"),
            Self::Cc     => write!(f, "cc"),
            Self::Mark   => write!(f, "mark"),
            Self::Dep    => write!(f, "dep"),
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

impl DepLabel {
    /// Parse a UD label string into a `DepLabel`.
    pub fn from_str(s: &str) -> Self {
        match s {
            "root"   => Self::Root,
            "nsubj"  => Self::Subj,
            "obj"    => Self::Obj,
            "iobj"   => Self::Iobj,
            "csubj"  => Self::Csubj,
            "ccomp"  => Self::Ccomp,
            "xcomp"  => Self::Xcomp,
            "nmod"   => Self::Nmod,
            "amod"   => Self::Amod,
            "advmod" => Self::Advmod,
            "aux"    => Self::Aux,
            "det"    => Self::Det,
            "case"   => Self::Case,
            "punct"  => Self::Punct,
            "conj"   => Self::Conj,
            "cc"     => Self::Cc,
            "mark"   => Self::Mark,
            "dep"    => Self::Dep,
            other    => Self::Other(other.to_string()),
        }
    }
}

/// A directed dependency arc: `head` → `dependent`.
///
/// Index 0 is the virtual ROOT node; token indices start at 1 in the underlying
/// graph (as in CoNLL-U), but `DependencyGraph` stores them 0-based internally
/// with `head == 0` meaning ROOT.
#[derive(Debug, Clone)]
pub struct DependencyArc {
    /// Head token index (0 = ROOT).
    pub head: usize,
    /// Dependent token index (1-based w.r.t. the sentence, stored 1-based here too).
    pub dependent: usize,
    /// Grammatical relation label.
    pub label: DepLabel,
    /// Model confidence score for the arc.
    pub score: f64,
}

/// Dependency parse tree represented as an adjacency structure.
///
/// Token indices in `arcs` use 1-based indexing (1 … n_tokens); index 0 is the
/// virtual ROOT.  Methods such as `head_of` and `dependents_of` accept 1-based
/// token indices.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Surface forms of the tokens (0-indexed).
    pub tokens: Vec<String>,
    /// Part-of-speech tags parallel to `tokens`.
    pub pos_tags: Vec<String>,
    /// One arc per non-root token (may contain additional arcs for multi-head graphs).
    pub arcs: Vec<DependencyArc>,
    /// Number of real tokens (not counting the virtual ROOT).
    pub n_tokens: usize,
}

impl DependencyGraph {
    /// Create an empty graph for the given token/POS sequences.
    pub fn new(tokens: Vec<String>, pos_tags: Vec<String>) -> Self {
        let n = tokens.len();
        Self {
            tokens,
            pos_tags,
            arcs: Vec::new(),
            n_tokens: n,
        }
    }

    /// Add a dependency arc.
    ///
    /// * `head`      – head index (0 = ROOT, 1..=n_tokens for real tokens).
    /// * `dependent` – dependent index (1..=n_tokens).
    /// * `label`     – grammatical relation.
    /// * `score`     – confidence / weight.
    pub fn add_arc(&mut self, head: usize, dependent: usize, label: DepLabel, score: f64) {
        self.arcs.push(DependencyArc { head, dependent, label, score });
    }

    /// Return the head of token `i` (1-based), or `None` if it is the root.
    pub fn head_of(&self, i: usize) -> Option<usize> {
        self.arcs.iter().find(|a| a.dependent == i).map(|a| a.head)
    }

    /// Return the label of the arc whose dependent is `i` (1-based), if any.
    pub fn label_of(&self, i: usize) -> Option<&DepLabel> {
        self.arcs.iter().find(|a| a.dependent == i).map(|a| &a.label)
    }

    /// Return all dependents of token `i` (1-based), including ROOT (i=0).
    pub fn dependents_of(&self, i: usize) -> Vec<usize> {
        self.arcs
            .iter()
            .filter(|a| a.head == i)
            .map(|a| a.dependent)
            .collect()
    }

    /// Return the shortest path between two tokens as `(token_idx, upward)` pairs.
    ///
    /// `upward == true` means the step goes from child to parent.
    /// Both `src` and `dst` are 1-based token indices.
    pub fn path(&self, src: usize, dst: usize) -> Vec<(usize, bool)> {
        // Build an undirected adjacency list over the range [0..=n_tokens].
        let size = self.n_tokens + 1; // index 0 = ROOT
        let mut adj: Vec<Vec<(usize, bool)>> = vec![Vec::new(); size];
        for arc in &self.arcs {
            if arc.head < size && arc.dependent < size {
                adj[arc.head].push((arc.dependent, false)); // downward
                adj[arc.dependent].push((arc.head, true));  // upward
            }
        }

        // BFS from `src` to `dst`.
        let mut visited = vec![false; size];
        let mut prev: Vec<Option<(usize, bool)>> = vec![None; size];
        let mut queue = VecDeque::new();
        if src < size {
            queue.push_back(src);
            visited[src] = true;
        }

        'bfs: while let Some(curr) = queue.pop_front() {
            if curr == dst {
                break 'bfs;
            }
            for &(next, up) in &adj[curr] {
                if !visited[next] {
                    visited[next] = true;
                    prev[next] = Some((curr, up));
                    queue.push_back(next);
                }
            }
        }

        // Reconstruct path by tracing predecessors back from `dst`.
        let mut path = Vec::new();
        let mut curr = dst;
        while let Some((p, up)) = prev[curr] {
            path.push((curr, up));
            curr = p;
        }
        path.reverse();
        path
    }

    /// Return `true` if no two arcs cross (i.e., the parse is projective).
    ///
    /// Arc `(h, d)` and arc `(h', d')` cross when exactly one of `h'`, `d'` lies
    /// strictly between `min(h, d)` and `max(h, d)`.
    pub fn is_projective(&self) -> bool {
        for a1 in &self.arcs {
            for a2 in &self.arcs {
                if std::ptr::eq(a1, a2) {
                    continue;
                }
                let lo1 = a1.head.min(a1.dependent);
                let hi1 = a1.head.max(a1.dependent);
                let lo2 = a2.head.min(a2.dependent);
                let hi2 = a2.head.max(a2.dependent);
                // Crossing: lo1 < lo2 < hi1 < hi2  or  lo2 < lo1 < hi2 < hi1
                if (lo1 < lo2 && lo2 < hi1 && hi1 < hi2)
                    || (lo2 < lo1 && lo1 < hi2 && hi2 < hi1)
                {
                    return false;
                }
            }
        }
        true
    }

    /// Labeled Attachment Score (LAS) against a gold-standard graph.
    ///
    /// Returns the fraction of tokens for which both the head and the label
    /// in `self` match those in `gold`.
    pub fn las(&self, gold: &DependencyGraph) -> f64 {
        if self.n_tokens == 0 {
            return 0.0;
        }
        let correct = self.arcs.iter().filter(|pred| {
            gold.arcs.iter().any(|g| {
                g.head == pred.head && g.dependent == pred.dependent && g.label == pred.label
            })
        }).count();
        correct as f64 / self.n_tokens as f64
    }

    /// Unlabeled Attachment Score (UAS) against a gold-standard graph.
    pub fn uas(&self, gold: &DependencyGraph) -> f64 {
        if self.n_tokens == 0 {
            return 0.0;
        }
        let correct = self.arcs.iter().filter(|pred| {
            gold.arcs.iter().any(|g| {
                g.head == pred.head && g.dependent == pred.dependent
            })
        }).count();
        correct as f64 / self.n_tokens as f64
    }

    /// Emit the graph in CoNLL-U format.
    ///
    /// Columns: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    pub fn to_conllu(&self) -> String {
        let mut out = String::new();
        for i in 1..=self.n_tokens {
            let arc = self.arcs.iter().find(|a| a.dependent == i);
            let (head, label) = arc
                .map(|a| (a.head, a.label.to_string()))
                .unwrap_or((0, "dep".to_string()));
            let form = self.tokens.get(i - 1).map(|s| s.as_str()).unwrap_or("_");
            let pos  = self.pos_tags.get(i - 1).map(|s| s.as_str()).unwrap_or("_");
            out += &format!("{}\t{}\t_\t{}\t_\t_\t{}\t{}\t_\t_\n", i, form, pos, head, label);
        }
        out
    }

    /// Parse a CoNLL-U string back into a `DependencyGraph`.
    pub fn from_conllu(conllu: &str) -> Self {
        let mut tokens = Vec::new();
        let mut pos_tags = Vec::new();
        let mut arcs = Vec::new();

        for line in conllu.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 8 {
                continue;
            }
            // Skip multi-word tokens (e.g. "1-2")
            if cols[0].contains('-') || cols[0].contains('.') {
                continue;
            }
            let dep_idx: usize = cols[0].parse().unwrap_or(0);
            let form = cols[1].to_string();
            let pos  = cols[3].to_string();
            let head: usize = cols[6].parse().unwrap_or(0);
            let label = DepLabel::from_str(cols[7]);
            tokens.push(form);
            pos_tags.push(pos);
            arcs.push(DependencyArc { head, dependent: dep_idx, label, score: 1.0 });
        }

        let n = tokens.len();
        Self { tokens, pos_tags, arcs, n_tokens: n }
    }

    /// Return the subtree rooted at token `root_idx` (1-based) as a sorted list
    /// of token indices.
    pub fn subtree(&self, root_idx: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![root_idx];
        while let Some(node) = stack.pop() {
            result.push(node);
            for child in self.dependents_of(node) {
                stack.push(child);
            }
        }
        result.sort_unstable();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> DependencyGraph {
        // "The cat sat" — DT NN VBD
        // ROOT -> sat(3), sat -> cat(2), cat -> The(1)
        let tokens  = vec!["The".into(), "cat".into(), "sat".into()];
        let pos     = vec!["DT".into(), "NN".into(), "VBD".into()];
        let mut g   = DependencyGraph::new(tokens, pos);
        g.add_arc(0, 3, DepLabel::Root,  1.0); // ROOT -> sat
        g.add_arc(3, 2, DepLabel::Subj,  1.0); // sat  -> cat
        g.add_arc(2, 1, DepLabel::Det,   1.0); // cat  -> The
        g
    }

    #[test]
    fn test_projectivity() {
        let g = simple_graph();
        assert!(g.is_projective());
    }

    #[test]
    fn test_non_projective() {
        // Construct a crossing arc: (1,3) and (2,4)
        let tokens = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let pos    = vec!["NN".into(); 4];
        let mut g  = DependencyGraph::new(tokens, pos);
        g.add_arc(1, 3, DepLabel::Dep, 1.0); // arc from 1 to 3
        g.add_arc(2, 4, DepLabel::Dep, 1.0); // arc from 2 to 4 — crosses (1,3)
        assert!(!g.is_projective());
    }

    #[test]
    fn test_head_of_and_dependents_of() {
        let g = simple_graph();
        assert_eq!(g.head_of(3), Some(0)); // sat's head is ROOT
        assert_eq!(g.head_of(2), Some(3)); // cat's head is sat
        assert_eq!(g.head_of(1), Some(2)); // The's head is cat
        let deps = g.dependents_of(3);
        assert!(deps.contains(&2));
    }

    #[test]
    fn test_conllu_roundtrip() {
        let g      = simple_graph();
        let conllu = g.to_conllu();
        let g2     = DependencyGraph::from_conllu(&conllu);
        assert_eq!(g2.n_tokens, g.n_tokens);
        assert_eq!(g2.tokens, g.tokens);
    }

    #[test]
    fn test_las_uas() {
        let gold = simple_graph();
        let pred = simple_graph();
        assert!((pred.las(&gold) - 1.0).abs() < 1e-9);
        assert!((pred.uas(&gold) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_subtree() {
        let g = simple_graph();
        // Subtree of "sat" (3) should include all tokens.
        let sub = g.subtree(3);
        assert!(sub.contains(&1));
        assert!(sub.contains(&2));
        assert!(sub.contains(&3));
    }

    #[test]
    fn test_path() {
        let g = simple_graph();
        // Path from "The"(1) to "sat"(3): 1->2->3
        let path = g.path(1, 3);
        assert!(!path.is_empty());
    }
}
