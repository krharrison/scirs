//! Semantic Parsing
//!
//! This module provides semantic parsing capabilities including frame semantics,
//! dependency parsing, constituency parsing, semantic role labeling, simplified
//! AMR graph construction, and logical form representations.
//!
//! ## Overview
//!
//! Semantic parsing bridges the gap between surface syntax and meaning
//! representation. This module implements:
//!
//! - [`SemanticFrame`]: Frame-semantic representation (predicate + argument slots)
//! - [`DependencyParser`]: Transition-based arc-eager dependency parser (rule-based)
//! - [`ConstituencyParser`]: CYK-based phrase structure parser with PCFG
//! - [`SemanticRoleLabeler`]: Simple SRL using dependency parse + patterns
//! - [`AMRLite`]: Simplified Abstract Meaning Representation graph builder
//! - [`LogicalForm`]: First-order logic-like representation
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::semantic_parsing::{DependencyParser, SemanticRoleLabeler};
//!
//! let mut parser = DependencyParser::new();
//! let sentence = "The cat sat on the mat";
//! let parse = parser.parse(sentence).expect("parse failed");
//! println!("Root: {:?}", parse.root());
//!
//! let srl = SemanticRoleLabeler::new();
//! let roles = srl.label(&parse).expect("srl failed");
//! for r in &roles {
//!     println!("{:?}", r);
//! }
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// SemanticFrame
// ────────────────────────────────────────────────────────────────────────────

/// A slot in a semantic frame (argument role + filler)
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSlot {
    /// Role name (e.g. "Agent", "Theme", "Location")
    pub role: String,
    /// Text filler for the slot
    pub filler: String,
    /// Token span (start, end) in the original sentence
    pub span: (usize, usize),
}

impl FrameSlot {
    /// Create a new frame slot
    pub fn new(role: impl Into<String>, filler: impl Into<String>, span: (usize, usize)) -> Self {
        Self {
            role: role.into(),
            filler: filler.into(),
            span,
        }
    }
}

/// Frame-semantic representation: a predicate together with its argument slots.
#[derive(Debug, Clone)]
pub struct SemanticFrame {
    /// The predicate (event / relation word)
    pub predicate: String,
    /// Frame name (e.g. "Motion", "Causation")
    pub frame_name: String,
    /// Argument slots
    pub slots: Vec<FrameSlot>,
}

impl SemanticFrame {
    /// Construct a new semantic frame
    pub fn new(predicate: impl Into<String>, frame_name: impl Into<String>) -> Self {
        Self {
            predicate: predicate.into(),
            frame_name: frame_name.into(),
            slots: Vec::new(),
        }
    }

    /// Add an argument slot
    pub fn add_slot(&mut self, slot: FrameSlot) {
        self.slots.push(slot);
    }

    /// Retrieve all slots matching a given role name
    pub fn slots_for_role(&self, role: &str) -> Vec<&FrameSlot> {
        self.slots.iter().filter(|s| s.role == role).collect()
    }
}

impl fmt::Display for SemanticFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})[", self.predicate, self.frame_name)?;
        for (i, slot) in self.slots.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", slot.role, slot.filler)?;
        }
        write!(f, "]")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dependency Parse structures
// ────────────────────────────────────────────────────────────────────────────

/// Universal dependency relation labels (simplified)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DepRel {
    /// Root of the sentence
    Root,
    /// Nominal subject
    NSubj,
    /// Direct object
    DObj,
    /// Indirect object
    IObj,
    /// Prepositional object
    PObj,
    /// Adjectival modifier
    AdjMod,
    /// Adverbial modifier
    AdvMod,
    /// Determiner
    Det,
    /// Prepositional modifier
    Prep,
    /// Compound noun
    Compound,
    /// Copula
    Cop,
    /// Auxiliary verb
    Aux,
    /// Punctuation
    Punct,
    /// Clausal subject
    CSubj,
    /// Clausal complement
    CComp,
    /// Open clausal complement
    XComp,
    /// Numeric modifier
    NumMod,
    /// Possessive modifier
    Poss,
    /// Relative clause
    RelCl,
    /// Other / unknown
    Other(String),
}

impl fmt::Display for DepRel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DepRel::Root => "root",
            DepRel::NSubj => "nsubj",
            DepRel::DObj => "dobj",
            DepRel::IObj => "iobj",
            DepRel::PObj => "pobj",
            DepRel::AdjMod => "amod",
            DepRel::AdvMod => "advmod",
            DepRel::Det => "det",
            DepRel::Prep => "prep",
            DepRel::Compound => "compound",
            DepRel::Cop => "cop",
            DepRel::Aux => "aux",
            DepRel::Punct => "punct",
            DepRel::CSubj => "csubj",
            DepRel::CComp => "ccomp",
            DepRel::XComp => "xcomp",
            DepRel::NumMod => "nummod",
            DepRel::Poss => "poss",
            DepRel::RelCl => "relcl",
            DepRel::Other(s) => s.as_str(),
        };
        write!(f, "{}", s)
    }
}

impl DepRel {
    /// Parse a string into a dependency relation
    pub fn from_str(s: &str) -> Self {
        match s {
            "root" => DepRel::Root,
            "nsubj" => DepRel::NSubj,
            "dobj" => DepRel::DObj,
            "iobj" => DepRel::IObj,
            "pobj" => DepRel::PObj,
            "amod" => DepRel::AdjMod,
            "advmod" => DepRel::AdvMod,
            "det" => DepRel::Det,
            "prep" => DepRel::Prep,
            "compound" => DepRel::Compound,
            "cop" => DepRel::Cop,
            "aux" => DepRel::Aux,
            "punct" => DepRel::Punct,
            "csubj" => DepRel::CSubj,
            "ccomp" => DepRel::CComp,
            "xcomp" => DepRel::XComp,
            "nummod" => DepRel::NumMod,
            "poss" => DepRel::Poss,
            "relcl" => DepRel::RelCl,
            other => DepRel::Other(other.to_string()),
        }
    }
}

/// A single token with its dependency annotation
#[derive(Debug, Clone)]
pub struct DepToken {
    /// 1-based index in the sentence
    pub index: usize,
    /// Surface form
    pub word: String,
    /// Coarse POS tag (NOUN, VERB, ADJ, ADV, DET, PREP, PRON, CONJ, PUNCT, NUM, OTHER)
    pub pos: String,
    /// Index of head token (0 = root pseudo-token)
    pub head: usize,
    /// Dependency relation to the head
    pub dep_rel: DepRel,
}

impl DepToken {
    /// Returns true if this token is the root
    pub fn is_root(&self) -> bool {
        self.dep_rel == DepRel::Root
    }
}

/// A complete dependency parse tree
#[derive(Debug, Clone)]
pub struct DependencyParse {
    /// All tokens in sentence order
    pub tokens: Vec<DepToken>,
}

impl DependencyParse {
    /// Return the root token
    pub fn root(&self) -> Option<&DepToken> {
        self.tokens.iter().find(|t| t.dep_rel == DepRel::Root)
    }

    /// Return all dependents of a given head index
    pub fn dependents_of(&self, head_idx: usize) -> Vec<&DepToken> {
        self.tokens.iter().filter(|t| t.head == head_idx).collect()
    }

    /// Return the head token for a given token index (1-based)
    pub fn head_of(&self, token_idx: usize) -> Option<&DepToken> {
        let token = self.tokens.get(token_idx.saturating_sub(1))?;
        if token.head == 0 {
            return None;
        }
        self.tokens.get(token.head.saturating_sub(1))
    }

    /// Return number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if parse has no tokens
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Arc-eager transition system (rule-based DependencyParser)
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the dependency parser
#[derive(Debug, Clone)]
pub struct DependencyParserConfig {
    /// Whether to use heuristic verb detection
    pub use_verb_heuristics: bool,
    /// Whether to detect prepositional phrases
    pub detect_preps: bool,
}

impl Default for DependencyParserConfig {
    fn default() -> Self {
        Self {
            use_verb_heuristics: true,
            detect_preps: true,
        }
    }
}

/// State of the arc-eager transition-based parser
struct ParserState {
    /// Stack of token indices (1-based)
    stack: Vec<usize>,
    /// Buffer (remaining token indices, front = next to process)
    buffer: Vec<usize>,
    /// Assigned heads: token_index -> head_index
    heads: Vec<usize>,
    /// Assigned dep relations
    rels: Vec<DepRel>,
    /// The token list (cloned from sentence)
    tokens: Vec<String>,
    /// POS tags
    pos_tags: Vec<String>,
}

impl ParserState {
    fn new(words: Vec<String>, pos_tags: Vec<String>) -> Self {
        let n = words.len();
        let buffer: Vec<usize> = (1..=n).collect();
        Self {
            stack: vec![0], // root pseudo-token
            buffer,
            heads: vec![0; n + 1],  // 1-based; index 0 = root placeholder
            rels: (0..=n).map(|_| DepRel::Other("_".to_string())).collect(),
            tokens: words,
            pos_tags,
        }
    }

    fn stack_top(&self) -> Option<usize> {
        self.stack.last().copied()
    }

    fn buffer_front(&self) -> Option<usize> {
        self.buffer.first().copied()
    }

    fn word_of(&self, idx: usize) -> Option<&str> {
        if idx == 0 {
            Some("ROOT")
        } else {
            self.tokens.get(idx - 1).map(|s| s.as_str())
        }
    }

    fn pos_of(&self, idx: usize) -> Option<&str> {
        if idx == 0 {
            Some("ROOT")
        } else {
            self.pos_tags.get(idx - 1).map(|s| s.as_str())
        }
    }

    /// Arc-left: buf[0] -> stack_top; pop stack
    fn arc_left(&mut self, rel: DepRel) {
        if let (Some(s), Some(b)) = (self.stack_top(), self.buffer_front()) {
            if s != 0 {
                self.heads[s] = b;
                self.rels[s] = rel;
                self.stack.pop();
            }
        }
    }

    /// Arc-right: stack_top -> buf[0]; push buf[0] onto stack, remove from buffer
    fn arc_right(&mut self, rel: DepRel) {
        if let (Some(s), Some(b)) = (self.stack_top(), self.buffer_front()) {
            self.heads[b] = s;
            self.rels[b] = rel;
            self.buffer.remove(0);
            self.stack.push(b);
        }
    }

    /// Shift: move buf[0] to top of stack
    fn shift(&mut self) {
        if let Some(b) = self.buffer_front() {
            self.buffer.remove(0);
            self.stack.push(b);
        }
    }

    /// Reduce: pop stack
    fn reduce(&mut self) {
        self.stack.pop();
    }
}

/// Transition-based arc-eager dependency parser (rule-based, pure Rust)
///
/// Implements a simplified oracle-free parser using heuristic rules modelled
/// after standard NLP patterns for English.  Not as accurate as a trained
/// neural parser but is 100 % pure Rust with no external model files.
pub struct DependencyParser {
    config: DependencyParserConfig,
}

impl Default for DependencyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyParser {
    /// Create a new dependency parser with default configuration
    pub fn new() -> Self {
        Self {
            config: DependencyParserConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DependencyParserConfig) -> Self {
        Self { config }
    }

    /// Parse a sentence into a dependency tree
    pub fn parse(&self, sentence: &str) -> Result<DependencyParse> {
        let raw_tokens: Vec<String> = sentence
            .split_whitespace()
            .map(|t| t.to_string())
            .collect();

        if raw_tokens.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot parse empty sentence".to_string(),
            ));
        }

        let pos_tags: Vec<String> = raw_tokens.iter().map(|w| self.coarse_pos(w)).collect();
        let n = raw_tokens.len();
        let mut state = ParserState::new(raw_tokens.clone(), pos_tags.clone());

        // Run the arc-eager transition system with rule-based decisions
        let max_steps = n * n + n + 10;
        let mut steps = 0;
        while (!state.buffer.is_empty() || state.stack.len() > 1) && steps < max_steps {
            steps += 1;
            let action = self.oracle_action(&state);
            match action {
                ParserAction::Shift => state.shift(),
                ParserAction::Reduce => state.reduce(),
                ParserAction::ArcLeft(rel) => state.arc_left(rel),
                ParserAction::ArcRight(rel) => state.arc_right(rel),
            }
        }

        // Assign root relation to stack top if no heads assigned yet
        for idx in 1..=n {
            if state.heads[idx] == 0 {
                state.heads[idx] = 0;
                state.rels[idx] = DepRel::Root;
            }
        }

        let tokens: Vec<DepToken> = (1..=n)
            .map(|i| DepToken {
                index: i,
                word: raw_tokens[i - 1].clone(),
                pos: pos_tags[i - 1].clone(),
                head: state.heads[i],
                dep_rel: state.rels[i].clone(),
            })
            .collect();

        Ok(DependencyParse { tokens })
    }

    // Determine the oracle action for the current parser state using heuristic rules
    fn oracle_action(&self, state: &ParserState) -> ParserAction {
        let s = state.stack_top();
        let b = state.buffer_front();

        match (s, b) {
            (None, _) | (_, None) => {
                if state.stack.len() > 1 {
                    ParserAction::Reduce
                } else {
                    ParserAction::Shift
                }
            }
            (Some(si), Some(bi)) => {
                let s_pos = state.pos_of(si).unwrap_or("OTHER");
                let b_pos = state.pos_of(bi).unwrap_or("OTHER");
                let s_word = state.word_of(si).unwrap_or("").to_lowercase();
                let b_word = state.word_of(bi).unwrap_or("").to_lowercase();

                // Rule: determiners/adjectives attach to the next noun (arc-right)
                if (s_pos == "DET" || s_pos == "ADJ" || s_pos == "NUM")
                    && (b_pos == "NOUN" || b_pos == "PROPN")
                {
                    let rel = if s_pos == "DET" {
                        DepRel::Det
                    } else if s_pos == "NUM" {
                        DepRel::NumMod
                    } else {
                        DepRel::AdjMod
                    };
                    return ParserAction::ArcRight(rel);
                }

                // Rule: nominal subject (NOUN/PRON before VERB)
                if (s_pos == "NOUN" || s_pos == "PRON" || s_pos == "PROPN") && b_pos == "VERB" {
                    return ParserAction::ArcLeft(DepRel::NSubj);
                }

                // Rule: aux/modal before main verb
                if s_pos == "AUX" && b_pos == "VERB" {
                    return ParserAction::ArcLeft(DepRel::Aux);
                }

                // Rule: adverb attached to verb
                if s_pos == "ADV" && b_pos == "VERB" {
                    return ParserAction::ArcLeft(DepRel::AdvMod);
                }

                // Rule: verb -> noun (direct object)
                if s_pos == "VERB" && (b_pos == "NOUN" || b_pos == "PRON" || b_pos == "PROPN") {
                    return ParserAction::ArcRight(DepRel::DObj);
                }

                // Rule: preposition chains
                if s_pos == "PREP" && (b_pos == "NOUN" || b_pos == "PRON" || b_pos == "PROPN") {
                    return ParserAction::ArcRight(DepRel::PObj);
                }
                if (s_pos == "VERB" || s_pos == "NOUN" || s_pos == "ADJ") && b_pos == "PREP" {
                    return ParserAction::ArcRight(DepRel::Prep);
                }

                // Copula + predicate
                if b_pos == "VERB" && (s_word == "is" || s_word == "are" || s_word == "was" || s_word == "were") {
                    return ParserAction::ArcLeft(DepRel::Cop);
                }

                // Adverb modifying adjective/adverb
                if s_pos == "ADV" && (b_pos == "ADJ" || b_pos == "ADV") {
                    return ParserAction::ArcLeft(DepRel::AdvMod);
                }

                // Punctuation
                if b_pos == "PUNCT" {
                    // Attach punct to stack top
                    if si != 0 {
                        return ParserAction::ArcRight(DepRel::Punct);
                    }
                }

                // Compound nouns
                if s_pos == "NOUN" && b_pos == "NOUN" {
                    return ParserAction::ArcRight(DepRel::Compound);
                }

                // Default: reduce if stack item has a head; otherwise shift
                if si != 0 && state.heads[si] != 0 {
                    ParserAction::Reduce
                } else {
                    // Root attachment when stack has only root
                    if si == 0 && b_pos == "VERB" {
                        return ParserAction::ArcRight(DepRel::Root);
                    }
                    ParserAction::Shift
                }
            }
        }
    }

    /// Assign a coarse POS tag to a word using simple morphological heuristics
    fn coarse_pos(&self, word: &str) -> String {
        let lower = word.to_lowercase();
        let clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();

        // Punctuation
        if word.chars().all(|c| !c.is_alphanumeric()) {
            return "PUNCT".to_string();
        }
        // Numbers
        if word.chars().any(|c| c.is_numeric()) && word.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
            return "NUM".to_string();
        }
        // Determiners
        let determiners = ["the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"];
        if determiners.contains(&clean.as_str()) {
            return "DET".to_string();
        }
        // Pronouns
        let pronouns = ["i", "me", "we", "us", "you", "he", "him", "she", "her", "it", "they", "them", "who", "what", "which"];
        if pronouns.contains(&clean.as_str()) {
            return "PRON".to_string();
        }
        // Prepositions
        let preps = ["in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "of", "up", "out", "off", "over", "under", "around", "near"];
        if preps.contains(&clean.as_str()) {
            return "PREP".to_string();
        }
        // Conjunctions
        let conjs = ["and", "or", "but", "nor", "so", "yet", "for", "because", "although", "while", "when", "if", "that", "than", "since", "until", "unless", "after", "before", "as"];
        if conjs.contains(&clean.as_str()) {
            return "CONJ".to_string();
        }
        // Auxiliaries / modals
        let aux = ["is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "can", "could", "must"];
        if aux.contains(&clean.as_str()) {
            return "AUX".to_string();
        }
        // Adverbs: many end in -ly
        if clean.ends_with("ly") && clean.len() > 3 {
            return "ADV".to_string();
        }
        // Adjectives: -ful, -less, -ous, -ive, -able, -ible, -al, -ary
        for suffix in &["ful", "less", "ous", "ive", "able", "ible", "al", "ary", "ic"] {
            if clean.ends_with(suffix) && clean.len() > suffix.len() + 1 {
                return "ADJ".to_string();
            }
        }
        // Verbs: -ing, -ed, -en (past participle), common verb endings
        if clean.ends_with("ing") && clean.len() > 4 {
            return "VERB".to_string();
        }
        if clean.ends_with("ed") && clean.len() > 3 {
            return "VERB".to_string();
        }
        // Proper nouns: starts with capital letter
        if word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && clean.len() > 1 {
            return "PROPN".to_string();
        }
        // Default to NOUN
        "NOUN".to_string()
    }
}

/// Actions in the arc-eager transition system
#[derive(Debug)]
enum ParserAction {
    Shift,
    Reduce,
    ArcLeft(DepRel),
    ArcRight(DepRel),
}

// ────────────────────────────────────────────────────────────────────────────
// ConstituencyParser – CYK with PCFG
// ────────────────────────────────────────────────────────────────────────────

/// A PCFG rule: LHS -> RHS with probability
#[derive(Debug, Clone)]
pub struct PcfgRule {
    /// Left-hand side non-terminal
    pub lhs: String,
    /// Right-hand side symbols (1 or 2 for CNF)
    pub rhs: Vec<String>,
    /// Log-probability of the rule
    pub log_prob: f64,
}

impl PcfgRule {
    /// Create a new rule with probability
    pub fn new(lhs: impl Into<String>, rhs: Vec<String>, prob: f64) -> Self {
        let lp = if prob > 0.0 { prob.ln() } else { f64::NEG_INFINITY };
        Self { lhs: lhs.into(), rhs, log_prob: lp }
    }
}

/// A phrase-structure constituent
#[derive(Debug, Clone)]
pub struct ParseNode {
    /// Non-terminal or terminal label
    pub label: String,
    /// Span (start, end) in the token list (exclusive end)
    pub span: (usize, usize),
    /// Child nodes
    pub children: Vec<ParseNode>,
    /// Log-probability of this sub-tree
    pub log_prob: f64,
}

impl ParseNode {
    /// Create a leaf (terminal) node
    pub fn leaf(label: impl Into<String>, position: usize, log_prob: f64) -> Self {
        ParseNode {
            label: label.into(),
            span: (position, position + 1),
            children: Vec::new(),
            log_prob,
        }
    }

    /// Pretty-print the parse tree in bracket notation
    pub fn to_bracket_string(&self) -> String {
        if self.children.is_empty() {
            format!("({})", self.label)
        } else {
            let children_str: String =
                self.children.iter().map(|c| c.to_bracket_string()).collect::<Vec<_>>().join(" ");
            format!("({} {})", self.label, children_str)
        }
    }
}

/// CYK-based constituency parser using a PCFG in Chomsky Normal Form (CNF)
pub struct ConstituencyParser {
    /// Grammar rules indexed by rhs1 (binary rules)
    binary_rules: HashMap<(String, String), Vec<PcfgRule>>,
    /// Unary rules indexed by terminal/rhs
    unary_rules: HashMap<String, Vec<PcfgRule>>,
    /// Start symbol
    start_symbol: String,
}

impl Default for ConstituencyParser {
    fn default() -> Self {
        Self::with_default_grammar()
    }
}

impl ConstituencyParser {
    /// Create a parser with a provided PCFG
    pub fn new(rules: Vec<PcfgRule>, start_symbol: impl Into<String>) -> Self {
        let mut binary_rules: HashMap<(String, String), Vec<PcfgRule>> = HashMap::new();
        let mut unary_rules: HashMap<String, Vec<PcfgRule>> = HashMap::new();

        for rule in rules {
            if rule.rhs.len() == 2 {
                let key = (rule.rhs[0].clone(), rule.rhs[1].clone());
                binary_rules.entry(key).or_default().push(rule);
            } else if rule.rhs.len() == 1 {
                let key = rule.rhs[0].clone();
                unary_rules.entry(key).or_default().push(rule);
            }
        }

        Self { binary_rules, unary_rules, start_symbol: start_symbol.into() }
    }

    /// Create a parser with a simple built-in English grammar
    pub fn with_default_grammar() -> Self {
        let rules = vec![
            // Phrasal rules
            PcfgRule::new("S", vec!["NP".into(), "VP".into()], 0.9),
            PcfgRule::new("S", vec!["VP".into(), "NP".into()], 0.1),
            PcfgRule::new("VP", vec!["V".into(), "NP".into()], 0.3),
            PcfgRule::new("VP", vec!["V".into(), "PP".into()], 0.15),
            PcfgRule::new("VP", vec!["VP".into(), "PP".into()], 0.15),
            PcfgRule::new("VP", vec!["VP".into(), "ADVP".into()], 0.1),
            PcfgRule::new("VP", vec!["AUX".into(), "VP".into()], 0.1),
            // Unary: intransitive verb phrase
            PcfgRule::new("VP", vec!["V".into()], 0.2),
            PcfgRule::new("NP", vec!["DT".into(), "NN".into()], 0.4),
            PcfgRule::new("NP", vec!["DT".into(), "NNS".into()], 0.2),
            PcfgRule::new("NP", vec!["NP".into(), "PP".into()], 0.15),
            PcfgRule::new("NP", vec!["JJ".into(), "NN".into()], 0.1),
            PcfgRule::new("NP", vec!["NNP".into(), "NNP".into()], 0.05),
            PcfgRule::new("NP", vec!["CD".into(), "NNS".into()], 0.1),
            PcfgRule::new("PP", vec!["IN".into(), "NP".into()], 1.0),
            PcfgRule::new("ADJP", vec!["JJ".into(), "CC".into()], 0.3),
            PcfgRule::new("ADJP", vec!["RB".into(), "JJ".into()], 0.7),
            PcfgRule::new("ADVP", vec!["RB".into(), "RB".into()], 0.5),
            PcfgRule::new("ADVP", vec!["RB".into(), "ADJP".into()], 0.5),
            // Unary rules (lexical)
            PcfgRule::new("NN", vec!["NOUN".into()], 1.0),
            PcfgRule::new("NNS", vec!["NOUNS".into()], 1.0),
            PcfgRule::new("NNP", vec!["PROPN".into()], 1.0),
            PcfgRule::new("V", vec!["VERB".into()], 0.8),
            PcfgRule::new("V", vec!["AUX".into()], 0.2),
            PcfgRule::new("DT", vec!["DET".into()], 1.0),
            PcfgRule::new("JJ", vec!["ADJ".into()], 1.0),
            PcfgRule::new("RB", vec!["ADV".into()], 1.0),
            PcfgRule::new("IN", vec!["PREP".into()], 1.0),
            PcfgRule::new("CC", vec!["CONJ".into()], 1.0),
            PcfgRule::new("CD", vec!["NUM".into()], 1.0),
        ];
        Self::new(rules, "S")
    }

    /// Parse a sequence of POS-tagged tokens
    ///
    /// Tokens are (word, pos_tag) pairs where pos_tag is a coarse tag
    /// (NOUN, VERB, ADJ, etc.).  Returns the best parse tree under the PCFG.
    pub fn parse_tagged(&self, tokens: &[(String, String)]) -> Result<ParseNode> {
        let n = tokens.len();
        if n == 0 {
            return Err(TextError::InvalidInput("Empty token list for constituency parsing".to_string()));
        }

        // CYK table: cell[i][j] = HashMap<non_terminal -> (log_prob, back_pointer)>
        // back_pointer: Option<(split_k, left_label, right_label)>
        type BackPtr = Option<(usize, String, String)>;
        let mut table: Vec<Vec<HashMap<String, (f64, BackPtr)>>> =
            vec![vec![HashMap::new(); n]; n];

        // Fill diagonal (span length 1) using unary rules
        for i in 0..n {
            let pos = &tokens[i].1;
            // Direct insertion of POS tag
            table[i][i].insert(pos.clone(), (0.0, None));
            // Apply unary closure: repeatedly apply unary rules until no new
            // non-terminals are added (handles chains like VERB -> V -> VP)
            let max_unary_depth = 10;
            for _ in 0..max_unary_depth {
                let current_nts: Vec<(String, f64)> = table[i][i]
                    .iter()
                    .map(|(nt, (lp, _))| (nt.clone(), *lp))
                    .collect();
                let mut changed = false;
                for (nt, nt_lp) in &current_nts {
                    if let Some(rules) = self.unary_rules.get(nt.as_str()) {
                        for rule in rules {
                            let lp = rule.log_prob + nt_lp;
                            let entry = table[i][i]
                                .entry(rule.lhs.clone())
                                .or_insert((f64::NEG_INFINITY, None));
                            if lp > entry.0 {
                                *entry = (lp, None);
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Fill spans of length 2..n
        for span_len in 2..=n {
            for i in 0..=(n - span_len) {
                let j = i + span_len - 1;
                let mut new_entries: Vec<(String, f64, usize, String, String)> = Vec::new();

                for k in i..j {
                    // Get references to left[i][k] and right[k+1][j]
                    let left_keys: Vec<(String, f64)> = table[i][k]
                        .iter()
                        .map(|(nt, (lp, _))| (nt.clone(), *lp))
                        .collect();
                    let right_keys: Vec<(String, f64)> = table[k + 1][j]
                        .iter()
                        .map(|(nt, (lp, _))| (nt.clone(), *lp))
                        .collect();

                    for (left_nt, left_lp) in &left_keys {
                        for (right_nt, right_lp) in &right_keys {
                            let key = (left_nt.clone(), right_nt.clone());
                            if let Some(rules) = self.binary_rules.get(&key) {
                                for rule in rules {
                                    let total_lp = rule.log_prob + left_lp + right_lp;
                                    new_entries.push((
                                        rule.lhs.clone(),
                                        total_lp,
                                        k,
                                        left_nt.clone(),
                                        right_nt.clone(),
                                    ));
                                }
                            }
                        }
                    }
                }

                for (lhs, total_lp, k, left_nt, right_nt) in new_entries {
                    let entry = table[i][j].entry(lhs).or_insert((f64::NEG_INFINITY, None));
                    if total_lp > entry.0 {
                        *entry = (total_lp, Some((k, left_nt, right_nt)));
                    }
                }

                // Apply unary closure on this cell
                let max_unary_depth = 10;
                for _ in 0..max_unary_depth {
                    let current_nts: Vec<(String, f64)> = table[i][j]
                        .iter()
                        .map(|(nt, (lp, _))| (nt.clone(), *lp))
                        .collect();
                    let mut changed = false;
                    for (nt, nt_lp) in &current_nts {
                        if let Some(rules) = self.unary_rules.get(nt.as_str()) {
                            for rule in rules {
                                let lp = rule.log_prob + nt_lp;
                                let entry = table[i][j]
                                    .entry(rule.lhs.clone())
                                    .or_insert((f64::NEG_INFINITY, None));
                                if lp > entry.0 {
                                    *entry = (lp, None);
                                    changed = true;
                                }
                            }
                        }
                    }
                    if !changed {
                        break;
                    }
                }
            }
        }

        // Extract best parse starting from start symbol
        if !table[0][n - 1].contains_key(&self.start_symbol) {
            // Fall back to the highest-probability constituent at root span
            let best = table[0][n - 1]
                .iter()
                .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.clone());
            let root_label = best.ok_or_else(|| {
                TextError::ProcessingError("CYK: no parse found".to_string())
            })?;
            return self.build_tree(&table, tokens, &root_label, 0, n - 1);
        }

        self.build_tree(&table, tokens, &self.start_symbol, 0, n - 1)
    }

    fn build_tree(
        &self,
        table: &[Vec<HashMap<String, (f64, Option<(usize, String, String)>)>>],
        tokens: &[(String, String)],
        label: &str,
        i: usize,
        j: usize,
    ) -> Result<ParseNode> {
        let (log_prob, back_ptr) = table[i][j]
            .get(label)
            .cloned()
            .ok_or_else(|| TextError::ProcessingError(format!("Missing entry {label} at [{i},{j}]")))?;

        if i == j {
            // Leaf
            let word = &tokens[i].0;
            return Ok(ParseNode {
                label: label.to_string(),
                span: (i, j + 1),
                children: vec![ParseNode::leaf(word.clone(), i, 0.0)],
                log_prob,
            });
        }

        match back_ptr {
            None => {
                // Unary or fallback – just wrap with a single child leaf
                let word = &tokens[i].0;
                Ok(ParseNode {
                    label: label.to_string(),
                    span: (i, j + 1),
                    children: vec![ParseNode::leaf(word.clone(), i, 0.0)],
                    log_prob,
                })
            }
            Some((k, left_label, right_label)) => {
                let left_child = self.build_tree(table, tokens, &left_label, i, k)?;
                let right_child = self.build_tree(table, tokens, &right_label, k + 1, j)?;
                Ok(ParseNode {
                    label: label.to_string(),
                    span: (i, j + 1),
                    children: vec![left_child, right_child],
                    log_prob,
                })
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Semantic Role Labeling
// ────────────────────────────────────────────────────────────────────────────

/// A semantic role annotation
#[derive(Debug, Clone)]
pub struct SemanticRole {
    /// Predicate token index (1-based)
    pub predicate_idx: usize,
    /// Predicate word
    pub predicate: String,
    /// Argument role label (A0 = agent, A1 = theme, AM-LOC, AM-TMP, etc.)
    pub role: String,
    /// Span of the argument (start, end) token indices (1-based, inclusive)
    pub span: (usize, usize),
    /// Text of the argument
    pub text: String,
}

/// Simple Semantic Role Labeler using dependency parse + heuristic patterns
pub struct SemanticRoleLabeler {
    /// Verb indicators beyond the coarse POS VERB
    extra_predicate_words: Vec<String>,
}

impl Default for SemanticRoleLabeler {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticRoleLabeler {
    /// Create a new SRL
    pub fn new() -> Self {
        Self {
            extra_predicate_words: Vec::new(),
        }
    }

    /// Add extra words that should be treated as predicates
    pub fn with_predicate_words(mut self, words: Vec<String>) -> Self {
        self.extra_predicate_words = words;
        self
    }

    /// Label semantic roles given a dependency parse
    pub fn label(&self, parse: &DependencyParse) -> Result<Vec<SemanticRole>> {
        let mut roles = Vec::new();

        // Find predicates (tokens with VERB or AUX POS)
        let predicates: Vec<usize> = parse
            .tokens
            .iter()
            .filter(|t| t.pos == "VERB" || t.pos == "AUX" || self.extra_predicate_words.contains(&t.word))
            .map(|t| t.index)
            .collect();

        for pred_idx in predicates {
            let pred_token = match parse.tokens.get(pred_idx - 1) {
                Some(t) => t,
                None => continue,
            };

            // A0 (agent): nsubj dependent of predicate
            for dep in parse.dependents_of(pred_idx) {
                let role_label = match dep.dep_rel {
                    DepRel::NSubj => "A0",
                    DepRel::DObj => "A1",
                    DepRel::IObj => "A2",
                    DepRel::Prep | DepRel::PObj => "AM-LOC",
                    DepRel::AdvMod => "AM-MNR",
                    DepRel::AdjMod => "AM-ADV",
                    _ => continue,
                };

                // Collect the entire sub-tree of this dependent as the span
                let (span_start, span_end) = self.subtree_span(parse, dep.index);
                let text = parse
                    .tokens
                    .iter()
                    .filter(|t| t.index >= span_start && t.index <= span_end)
                    .map(|t| t.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                roles.push(SemanticRole {
                    predicate_idx: pred_idx,
                    predicate: pred_token.word.clone(),
                    role: role_label.to_string(),
                    span: (span_start, span_end),
                    text,
                });
            }
        }

        Ok(roles)
    }

    /// Compute the span (min_idx, max_idx) of the subtree rooted at `idx`
    fn subtree_span(&self, parse: &DependencyParse, idx: usize) -> (usize, usize) {
        let mut min_idx = idx;
        let mut max_idx = idx;
        let mut stack = vec![idx];
        while let Some(cur) = stack.pop() {
            for dep in parse.dependents_of(cur) {
                min_idx = min_idx.min(dep.index);
                max_idx = max_idx.max(dep.index);
                stack.push(dep.index);
            }
        }
        (min_idx, max_idx)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AMRLite – simplified Abstract Meaning Representation
// ────────────────────────────────────────────────────────────────────────────

/// A node in an AMR-lite graph
#[derive(Debug, Clone)]
pub struct AmrNode {
    /// Variable name (e.g. "e1", "x1")
    pub variable: String,
    /// Concept (e.g. "walk-01", "person")
    pub concept: String,
}

/// An edge in an AMR-lite graph
#[derive(Debug, Clone)]
pub struct AmrEdge {
    /// Source variable
    pub source: String,
    /// Relation label (e.g. ":ARG0", ":location")
    pub relation: String,
    /// Target variable or constant
    pub target: String,
}

/// Simplified Abstract Meaning Representation graph
#[derive(Debug, Clone)]
pub struct AmrGraph {
    /// Root variable
    pub root: String,
    /// Nodes indexed by variable
    pub nodes: HashMap<String, AmrNode>,
    /// Edges
    pub edges: Vec<AmrEdge>,
}

impl AmrGraph {
    /// Create an empty AMR graph
    pub fn new(root_var: impl Into<String>) -> Self {
        Self {
            root: root_var.into(),
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node
    pub fn add_node(&mut self, variable: impl Into<String>, concept: impl Into<String>) {
        let var = variable.into();
        self.nodes.insert(
            var.clone(),
            AmrNode { variable: var, concept: concept.into() },
        );
    }

    /// Add an edge
    pub fn add_edge(&mut self, source: impl Into<String>, relation: impl Into<String>, target: impl Into<String>) {
        self.edges.push(AmrEdge {
            source: source.into(),
            relation: relation.into(),
            target: target.into(),
        });
    }

    /// Render the AMR graph as a PENMAN notation string
    pub fn to_penman(&self) -> String {
        fn render(
            graph: &AmrGraph,
            var: &str,
            visited: &mut std::collections::HashSet<String>,
            indent: usize,
        ) -> String {
            if visited.contains(var) {
                return var.to_string();
            }
            visited.insert(var.to_string());
            let concept = graph
                .nodes
                .get(var)
                .map(|n| n.concept.as_str())
                .unwrap_or("unknown");
            let pad = " ".repeat(indent * 4);
            let mut s = format!("({} / {}", var, concept);
            for edge in &graph.edges {
                if edge.source == var {
                    let target_str = render(graph, &edge.target, visited, indent + 1);
                    s.push_str(&format!("\n{}      {} {}", pad, edge.relation, target_str));
                }
            }
            s.push(')');
            s
        }
        let mut visited = std::collections::HashSet::new();
        render(self, &self.root, &mut visited, 0)
    }
}

/// Builder that constructs an AMR-lite graph from a dependency parse
pub struct AMRLite {
    /// Counter for variable naming
    counter: usize,
}

impl Default for AMRLite {
    fn default() -> Self {
        Self::new()
    }
}

impl AMRLite {
    /// Create a new AMRLite builder
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Build an AMR graph from a dependency parse
    pub fn build_from_parse(&mut self, parse: &DependencyParse) -> Result<AmrGraph> {
        if parse.tokens.is_empty() {
            return Err(TextError::InvalidInput("Empty parse for AMR building".to_string()));
        }

        let root_token = parse.root().ok_or_else(|| {
            TextError::ProcessingError("No root token in dependency parse".to_string())
        })?;

        let root_var = self.new_var(root_token);
        let mut graph = AmrGraph::new(root_var.clone());
        self.add_subtree(parse, root_token.index, &root_var, &mut graph);

        Ok(graph)
    }

    fn new_var(&mut self, token: &DepToken) -> String {
        self.counter += 1;
        let prefix = token.word.chars().next().unwrap_or('x').to_lowercase().next().unwrap_or('x');
        format!("{}{}", prefix, self.counter)
    }

    fn concept_of(token: &DepToken) -> String {
        let w = token.word.to_lowercase();
        match token.pos.as_str() {
            "VERB" | "AUX" => format!("{}-01", w),
            _ => w,
        }
    }

    fn add_subtree(
        &mut self,
        parse: &DependencyParse,
        token_idx: usize,
        var: &str,
        graph: &mut AmrGraph,
    ) {
        let token = match parse.tokens.get(token_idx.saturating_sub(1)) {
            Some(t) => t.clone(),
            None => return,
        };

        graph.add_node(var.to_string(), Self::concept_of(&token));

        for dep in parse.dependents_of(token_idx) {
            let dep_clone = dep.clone();
            let dep_var = self.new_var(&dep_clone);
            let relation = dep_rel_to_amr_relation(&dep_clone.dep_rel);
            graph.add_edge(var.to_string(), relation, dep_var.clone());
            self.add_subtree(parse, dep_clone.index, &dep_var, graph);
        }
    }
}

fn dep_rel_to_amr_relation(rel: &DepRel) -> String {
    match rel {
        DepRel::NSubj => ":ARG0".to_string(),
        DepRel::DObj => ":ARG1".to_string(),
        DepRel::IObj => ":ARG2".to_string(),
        DepRel::Prep | DepRel::PObj => ":location".to_string(),
        DepRel::AdjMod => ":mod".to_string(),
        DepRel::AdvMod => ":manner".to_string(),
        DepRel::Det => ":quant".to_string(),
        DepRel::Compound => ":name".to_string(),
        DepRel::NumMod => ":quant".to_string(),
        DepRel::Poss => ":poss".to_string(),
        _ => format!(":{}", rel),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LogicalForm – first-order logic representation
// ────────────────────────────────────────────────────────────────────────────

/// A first-order logic term
#[derive(Debug, Clone, PartialEq)]
pub enum LfTerm {
    /// A constant (e.g. a named entity)
    Constant(String),
    /// A variable
    Variable(String),
    /// A function application
    Function(String, Vec<LfTerm>),
}

impl fmt::Display for LfTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LfTerm::Constant(c) => write!(f, "{}", c),
            LfTerm::Variable(v) => write!(f, "?{}", v),
            LfTerm::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// A logical formula
#[derive(Debug, Clone)]
pub enum LogicalFormula {
    /// Atomic predicate
    Atom {
        /// The predicate name
        predicate: String,
        /// The arguments of the predicate
        args: Vec<LfTerm>,
    },
    /// Negation
    Not(Box<LogicalFormula>),
    /// Conjunction
    And(Vec<LogicalFormula>),
    /// Disjunction
    Or(Vec<LogicalFormula>),
    /// Existential quantifier
    Exists {
        /// The bound variable name
        variable: String,
        /// The body formula
        body: Box<LogicalFormula>,
    },
    /// Universal quantifier
    ForAll {
        /// The bound variable name
        variable: String,
        /// The body formula
        body: Box<LogicalFormula>,
    },
    /// Equality
    Equal(LfTerm, LfTerm),
}

impl fmt::Display for LogicalFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalFormula::Atom { predicate, args } => {
                write!(f, "{}(", predicate)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
            LogicalFormula::Not(inner) => write!(f, "¬({})", inner),
            LogicalFormula::And(conjuncts) => {
                let s: Vec<String> = conjuncts.iter().map(|c| format!("{}", c)).collect();
                write!(f, "({})", s.join(" ∧ "))
            }
            LogicalFormula::Or(disjuncts) => {
                let s: Vec<String> = disjuncts.iter().map(|c| format!("{}", c)).collect();
                write!(f, "({})", s.join(" ∨ "))
            }
            LogicalFormula::Exists { variable, body } => {
                write!(f, "∃?{} [{}]", variable, body)
            }
            LogicalFormula::ForAll { variable, body } => {
                write!(f, "∀?{} [{}]", variable, body)
            }
            LogicalFormula::Equal(a, b) => write!(f, "{} = {}", a, b),
        }
    }
}

/// First-order logic representation builder for sentences
pub struct LogicalForm {
    /// Variable counter
    var_counter: usize,
}

impl Default for LogicalForm {
    fn default() -> Self {
        Self::new()
    }
}

impl LogicalForm {
    /// Create a new LogicalForm builder
    pub fn new() -> Self {
        Self { var_counter: 0 }
    }

    /// Attempt to build a logical form from a dependency parse
    ///
    /// Implements a simplified Neo-Davidsonian semantics where events are
    /// reified as variables and arguments are attached as predicates.
    pub fn from_parse(&mut self, parse: &DependencyParse) -> Result<LogicalFormula> {
        let root = parse.root().ok_or_else(|| {
            TextError::ProcessingError("No root token found".to_string())
        })?;

        let (formula, _event_var) = self.translate_token(parse, root.index)?;
        Ok(formula)
    }

    fn new_var(&mut self, prefix: &str) -> String {
        self.var_counter += 1;
        format!("{}{}", prefix, self.var_counter)
    }

    fn translate_token(
        &mut self,
        parse: &DependencyParse,
        token_idx: usize,
    ) -> Result<(LogicalFormula, String)> {
        let token = parse
            .tokens
            .get(token_idx.saturating_sub(1))
            .ok_or_else(|| TextError::InvalidInput(format!("Token index {} out of range", token_idx)))?
            .clone();

        let event_var = self.new_var("e");
        let mut conjuncts: Vec<LogicalFormula> = Vec::new();

        // Root event predicate: e.g. run(e1)
        conjuncts.push(LogicalFormula::Atom {
            predicate: token.word.to_lowercase(),
            args: vec![LfTerm::Variable(event_var.clone())],
        });

        for dep in parse.dependents_of(token_idx) {
            let dep_clone = dep.clone();
            match dep_clone.dep_rel {
                DepRel::NSubj => {
                    let subj_const = LfTerm::Constant(dep_clone.word.clone());
                    conjuncts.push(LogicalFormula::Atom {
                        predicate: "agent".to_string(),
                        args: vec![LfTerm::Variable(event_var.clone()), subj_const],
                    });
                }
                DepRel::DObj => {
                    let obj_const = LfTerm::Constant(dep_clone.word.clone());
                    conjuncts.push(LogicalFormula::Atom {
                        predicate: "theme".to_string(),
                        args: vec![LfTerm::Variable(event_var.clone()), obj_const],
                    });
                }
                DepRel::Prep | DepRel::PObj => {
                    let loc_const = LfTerm::Constant(dep_clone.word.clone());
                    conjuncts.push(LogicalFormula::Atom {
                        predicate: "location".to_string(),
                        args: vec![LfTerm::Variable(event_var.clone()), loc_const],
                    });
                }
                DepRel::AdvMod | DepRel::AdjMod => {
                    let mod_const = LfTerm::Constant(dep_clone.word.clone());
                    conjuncts.push(LogicalFormula::Atom {
                        predicate: "manner".to_string(),
                        args: vec![LfTerm::Variable(event_var.clone()), mod_const],
                    });
                }
                _ => {}
            }
        }

        let formula = if conjuncts.len() == 1 {
            conjuncts.remove(0)
        } else {
            LogicalFormula::And(conjuncts)
        };

        let exists_formula = LogicalFormula::Exists {
            variable: event_var.clone(),
            body: Box::new(formula),
        };

        Ok((exists_formula, event_var))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_parser_basic() {
        let parser = DependencyParser::new();
        let parse = parser.parse("The cat sat on the mat").expect("parse failed");
        assert!(!parse.tokens.is_empty());
        // All tokens must have a defined dependency relation
        for t in &parse.tokens {
            assert!(!format!("{}", t.dep_rel).is_empty());
        }
    }

    #[test]
    fn test_dependency_parse_root() {
        let parser = DependencyParser::new();
        let parse = parser.parse("Dogs run fast").expect("parse failed");
        // There should be at least one token marked as root
        let root = parse.root();
        assert!(root.is_some());
    }

    #[test]
    fn test_semantic_frame() {
        let mut frame = SemanticFrame::new("run", "Motion");
        frame.add_slot(FrameSlot::new("Agent", "cat", (0, 1)));
        frame.add_slot(FrameSlot::new("Location", "mat", (4, 5)));
        assert_eq!(frame.slots_for_role("Agent").len(), 1);
        assert_eq!(frame.slots_for_role("Location").len(), 1);
        assert_eq!(frame.slots_for_role("Theme").len(), 0);
        let s = frame.to_string();
        assert!(s.contains("run"));
    }

    #[test]
    fn test_srl_on_parse() {
        let parser = DependencyParser::new();
        let parse = parser.parse("John loves Mary").expect("parse");
        let srl = SemanticRoleLabeler::new();
        let roles = srl.label(&parse).expect("label");
        // Should produce some roles for a transitive verb
        // (exact output depends on POS heuristics)
        let _ = roles;
    }

    #[test]
    fn test_amr_lite_build() {
        let parser = DependencyParser::new();
        let parse = parser.parse("The cat runs fast").expect("parse");
        let mut amr_builder = AMRLite::new();
        let graph = amr_builder.build_from_parse(&parse).expect("amr build");
        assert!(!graph.root.is_empty());
        let penman = graph.to_penman();
        assert!(!penman.is_empty());
    }

    #[test]
    fn test_logical_form_from_parse() {
        let parser = DependencyParser::new();
        let parse = parser.parse("Alice loves Bob").expect("parse");
        let mut lf = LogicalForm::new();
        let formula = lf.from_parse(&parse).expect("lf");
        let s = formula.to_string();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_constituency_parser_default_grammar() {
        let parser = ConstituencyParser::default();
        let tokens = vec![
            ("The".to_string(), "DET".to_string()),
            ("cat".to_string(), "NOUN".to_string()),
            ("ran".to_string(), "VERB".to_string()),
        ];
        let result = parser.parse_tagged(&tokens);
        assert!(result.is_ok());
        let tree = result.expect("tree");
        let bracket = tree.to_bracket_string();
        assert!(!bracket.is_empty());
    }

    #[test]
    fn test_dep_rel_roundtrip() {
        let labels = ["root", "nsubj", "dobj", "prep", "advmod", "det"];
        for label in &labels {
            let rel = DepRel::from_str(label);
            assert_eq!(format!("{}", rel), *label);
        }
    }

    #[test]
    fn test_logical_form_display() {
        let formula = LogicalFormula::And(vec![
            LogicalFormula::Atom {
                predicate: "run".to_string(),
                args: vec![LfTerm::Variable("e1".to_string())],
            },
            LogicalFormula::Atom {
                predicate: "agent".to_string(),
                args: vec![
                    LfTerm::Variable("e1".to_string()),
                    LfTerm::Constant("Alice".to_string()),
                ],
            },
        ]);
        let s = formula.to_string();
        assert!(s.contains("run"));
        assert!(s.contains("agent"));
    }
}
