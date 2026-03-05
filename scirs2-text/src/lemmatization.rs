//! Advanced Lemmatization for English text
//!
//! This module provides comprehensive lemmatization with two complementary
//! implementations:
//!
//! - [`RuleBasedLemmatizer`]: Uses linguistic suffix-removal rules for English
//!   covering verbs, nouns, adjectives and adverbs. Handles regular inflections
//!   efficiently with no embedded data, making it suitable for high-throughput
//!   scenarios where a compact memory footprint is required.
//!
//! - [`WordNetLemmatizer`]: Uses a compact embedded exception dictionary and
//!   morphological rules modelled on the WordNet morphy algorithm.  The embedded
//!   word-lists cover the most frequent irregular forms in English, giving much
//!   better coverage of irregular plurals, verb conjugations, and comparative /
//!   superlative adjectives.
//!
//! # Lemmatization vs. Stemming
//!
//! Stemming reduces a word to a root form that may not be a real word
//! (e.g. "running" → "runn" with naive suffix stripping).
//! Lemmatization always returns a valid dictionary word called the *lemma*
//! (e.g. "running" → "run").
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::lemmatization::{RuleBasedLemmatizer, WordNetLemmatizer, Lemmatizer};
//!
//! let rb = RuleBasedLemmatizer::new();
//! assert_eq!(rb.lemmatize("running", None).unwrap(), "run");
//! assert_eq!(rb.lemmatize("cats",    None).unwrap(), "cat");
//! assert_eq!(rb.lemmatize("happier", None).unwrap(), "happy");
//!
//! let wn = WordNetLemmatizer::new();
//! assert_eq!(wn.lemmatize("went",     Some("v")).unwrap(), "go");
//! assert_eq!(wn.lemmatize("children", Some("n")).unwrap(), "child");
//! assert_eq!(wn.lemmatize("better",   Some("a")).unwrap(), "good");
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public trait
// ─────────────────────────────────────────────────────────────────────────────

/// Core lemmatization trait.
///
/// The optional `pos` parameter is a coarse part-of-speech hint:
/// - `"n"` or `"noun"`  → noun
/// - `"v"` or `"verb"`  → verb
/// - `"a"` or `"adj"`   → adjective
/// - `"r"` or `"adv"`   → adverb
/// - `None`             → auto-detect (try all categories)
pub trait Lemmatizer: Send + Sync {
    /// Return the canonical base form (lemma) of `word`.
    fn lemmatize(&self, word: &str, pos: Option<&str>) -> Result<String>;

    /// Lemmatize a batch of `(word, pos)` pairs.
    fn lemmatize_batch(&self, pairs: &[(&str, Option<&str>)]) -> Result<Vec<String>> {
        pairs.iter().map(|(w, p)| self.lemmatize(w, *p)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper – coarse POS normalisation
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pos {
    Noun,
    Verb,
    Adj,
    Adv,
}

fn parse_pos(pos: Option<&str>) -> Option<Pos> {
    match pos?.to_lowercase().as_str() {
        "n" | "noun" => Some(Pos::Noun),
        "v" | "verb" => Some(Pos::Verb),
        "a" | "adj" | "adjective" | "s" | "satellite" => Some(Pos::Adj),
        "r" | "adv" | "adverb" => Some(Pos::Adv),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule-based lemmatizer
// ─────────────────────────────────────────────────────────────────────────────

/// A compact rule-based lemmatizer for English.
///
/// Uses ordered suffix-removal rules with vowel-checking guards to avoid
/// stripping valid stems.  Does not require any external data files.
///
/// Rules are applied in priority order; the first matching rule whose
/// resulting stem meets the minimum-length and vowel constraints wins.
#[derive(Debug, Clone)]
pub struct RuleBasedLemmatizer {
    /// (suffix_to_strip, replacement, min_remaining_length, pos)
    verb_rules: Vec<(&'static str, &'static str, usize)>,
    noun_rules: Vec<(&'static str, &'static str, usize)>,
    adj_rules: Vec<(&'static str, &'static str, usize)>,
    adv_rules: Vec<(&'static str, &'static str, usize)>,
    /// Small exception dictionary for very common irregulars
    exceptions: HashMap<&'static str, &'static str>,
}

impl RuleBasedLemmatizer {
    /// Create a lemmatizer with built-in English rules.
    pub fn new() -> Self {
        // (suffix, replacement, min_stem_len)
        // Longer suffixes are placed first to take priority.
        let verb_rules = vec![
            ("nesses", "ness", 2),
            ("ations", "ate", 3),
            ("ifying", "ify", 3),
            ("ifying", "ify", 3),
            ("alized", "alize", 3),
            ("alized", "al", 3),
            ("ifying", "ify", 3),
            ("itting", "it", 2),
            ("etting", "et", 2),
            ("inning", "in", 2),
            ("unning", "un", 2),
            ("imming", "im", 2),
            ("anning", "an", 2),
            ("lling", "ll", 2),
            ("ssing", "ss", 2),
            ("tting", "t", 2),
            ("nning", "n", 2),
            ("pping", "p", 2),
            ("mming", "m", 2),
            ("gging", "g", 2),
            ("dding", "d", 2),
            ("bbing", "b", 2),
            ("rring", "r", 2),
            ("zing", "ze", 2),
            ("ying", "y", 2),
            ("wing", "w", 2),
            ("eing", "e", 2),
            ("ying", "y", 2),
            ("ates", "ate", 3),
            ("izes", "ize", 3),
            ("ised", "ise", 3),
            ("ized", "ize", 3),
            ("ated", "ate", 3),
            ("ened", "en", 3),
            ("sted", "st", 2),
            ("ched", "ch", 2),
            ("shed", "sh", 2),
            ("xed", "x", 2),
            ("zed", "ze", 2),
            ("ied", "y", 2),
            ("oes", "o", 2),
            ("ves", "ve", 2),
            ("ies", "y", 2),
            ("ing", "", 3),
            ("ed", "", 2),
            ("es", "", 2),
            ("s", "", 3),
        ];

        let noun_rules = vec![
            ("nesses", "ness", 2),
            ("ments", "ment", 3),
            ("ships", "ship", 3),
            ("tions", "tion", 3),
            ("hoods", "hood", 3),
            ("lings", "ling", 3),
            ("eries", "ery", 3),
            ("eries", "er", 3),
            ("ivities", "ivity", 3),
            ("alities", "ality", 3),
            ("cities", "city", 3),
            ("bodies", "body", 3),
            ("copies", "copy", 3),
            ("ladies", "lady", 3),
            ("babies", "baby", 3),
            ("stories", "story", 3),
            ("studies", "study", 3),
            ("libraries", "library", 3),
            ("activities", "activity", 3),
            ("families", "family", 3),
            ("parties", "party", 3),
            ("categories", "category", 3),
            ("ches", "ch", 2),
            ("shes", "sh", 2),
            ("sses", "ss", 2),
            ("xes", "x", 2),
            ("zes", "z", 2),
            ("ies", "y", 2),
            ("ves", "f", 2),
            ("ves", "fe", 2),
            ("men", "man", 2),
            ("ses", "s", 3),
            ("s", "", 2),
        ];

        let adj_rules = vec![
            ("nesses", "ness", 2),
            ("iest", "y", 2),
            // -est / -er: try double-consonant reduction first ("bigg" → "big"),
            // then silent-e restoration ("larg" → "large").
            ("est", "", 3),
            ("est", "e", 3),
            ("ier", "y", 2),
            ("er", "", 3),
            ("er", "e", 3),
            ("ful", "ful", 2),
            ("less", "less", 2),
            ("ous", "ous", 2),
            ("ish", "ish", 2),
            ("ly", "", 3),
        ];

        let adv_rules = vec![
            ("ically", "ic", 3),
            ("ably", "able", 3),
            ("ibly", "ible", 3),
            ("fully", "ful", 3),
            ("lessly", "less", 3),
            ("ally", "al", 3),
            ("ily", "y", 2),
            ("ly", "", 3),
        ];

        let mut exceptions: HashMap<&'static str, &'static str> = HashMap::new();
        // High-frequency irregular forms
        exceptions.insert("are", "be");
        exceptions.insert("am", "be");
        exceptions.insert("is", "be");
        exceptions.insert("was", "be");
        exceptions.insert("were", "be");
        exceptions.insert("been", "be");
        exceptions.insert("being", "be");
        exceptions.insert("has", "have");
        exceptions.insert("had", "have");
        exceptions.insert("having", "have");
        exceptions.insert("does", "do");
        exceptions.insert("did", "do");
        exceptions.insert("doing", "do");
        exceptions.insert("went", "go");
        exceptions.insert("gone", "go");
        exceptions.insert("going", "go");
        exceptions.insert("goes", "go");
        exceptions.insert("ran", "run");
        exceptions.insert("running", "run");
        exceptions.insert("ate", "eat");
        exceptions.insert("eaten", "eat");
        exceptions.insert("eating", "eat");
        exceptions.insert("saw", "see");
        exceptions.insert("seen", "see");
        exceptions.insert("seeing", "see");
        exceptions.insert("gave", "give");
        exceptions.insert("given", "give");
        exceptions.insert("giving", "give");
        exceptions.insert("took", "take");
        exceptions.insert("taken", "take");
        exceptions.insert("taking", "take");
        exceptions.insert("came", "come");
        exceptions.insert("coming", "come");
        exceptions.insert("knew", "know");
        exceptions.insert("known", "know");
        exceptions.insert("knew", "know");
        exceptions.insert("got", "get");
        exceptions.insert("gotten", "get");
        exceptions.insert("getting", "get");
        exceptions.insert("made", "make");
        exceptions.insert("making", "make");
        exceptions.insert("said", "say");
        exceptions.insert("saying", "say");
        exceptions.insert("says", "say");
        exceptions.insert("better", "good");
        exceptions.insert("best", "good");
        exceptions.insert("worse", "bad");
        exceptions.insert("worst", "bad");
        exceptions.insert("more", "much");
        exceptions.insert("most", "much");
        exceptions.insert("less", "little");
        exceptions.insert("least", "little");
        exceptions.insert("farther", "far");
        exceptions.insert("farthest", "far");
        exceptions.insert("further", "far");
        exceptions.insert("furthest", "far");
        // Irregular nouns
        exceptions.insert("men", "man");
        exceptions.insert("women", "woman");
        exceptions.insert("children", "child");
        exceptions.insert("feet", "foot");
        exceptions.insert("teeth", "tooth");
        exceptions.insert("mice", "mouse");
        exceptions.insert("geese", "goose");
        exceptions.insert("oxen", "ox");
        exceptions.insert("lice", "louse");
        exceptions.insert("cacti", "cactus");
        exceptions.insert("alumni", "alumnus");
        exceptions.insert("curricula", "curriculum");
        exceptions.insert("bacteria", "bacterium");
        exceptions.insert("criteria", "criterion");
        exceptions.insert("phenomena", "phenomenon");
        exceptions.insert("indices", "index");
        exceptions.insert("vertices", "vertex");
        exceptions.insert("matrices", "matrix");
        exceptions.insert("analyses", "analysis");
        exceptions.insert("bases", "basis");
        exceptions.insert("crises", "crisis");
        exceptions.insert("parentheses", "parenthesis");
        exceptions.insert("hypotheses", "hypothesis");
        exceptions.insert("diagnoses", "diagnosis");
        exceptions.insert("axes", "axis");
        exceptions.insert("theses", "thesis");
        exceptions.insert("appendices", "appendix");

        Self {
            verb_rules,
            noun_rules,
            adj_rules,
            adv_rules,
            exceptions,
        }
    }

    /// Check whether a string contains at least one vowel (including y).
    fn has_vowel(s: &str) -> bool {
        s.chars()
            .any(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y'))
    }

    /// If `s` ends with a doubled consonant (e.g. "bigg"), return a version
    /// with one copy removed ("big").  Returns `None` if no doubling found.
    fn reduce_double_consonant(s: &str) -> Option<String> {
        let bytes = s.as_bytes();
        let len = bytes.len();
        if len >= 2 {
            let last = bytes[len - 1];
            let prev = bytes[len - 2];
            // Only reduce consonants (not doubled vowels like "ee", "oo")
            if last == prev && !matches!(last, b'a' | b'e' | b'i' | b'o' | b'u') {
                return Some(s[..len - 1].to_string());
            }
        }
        None
    }

    /// Apply a rule-list to `word` using simple suffix substitution.
    ///
    /// Each rule is `(suffix_to_strip, replacement_suffix, min_candidate_len)`.
    /// The first rule that produces a candidate satisfying the length and
    /// vowel constraints is returned.
    fn apply_rules(word: &str, rules: &[(&'static str, &'static str, usize)]) -> Option<String> {
        for (suffix, replacement, min_len) in rules {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if stem.is_empty() {
                    continue;
                }
                let candidate = format!("{stem}{replacement}");
                if candidate.len() >= *min_len && Self::has_vowel(&candidate) {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Apply morphological rules that handle comparative/superlative forms.
    ///
    /// Each rule is `(suffix_to_strip, replacement, min_len)`.  When the
    /// replacement is `""` the function attempts double-consonant reduction
    /// ("bigg" → "big"); if no reduction is possible the rule is skipped so
    /// that a following `("est","e")` rule can handle silent-e restoration.
    fn apply_morph_rules(
        word: &str,
        rules: &[(&'static str, &'static str, usize)],
    ) -> Option<String> {
        for (suffix, replacement, min_len) in rules {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if stem.is_empty() {
                    continue;
                }
                if replacement.is_empty() {
                    // Only fire if double-consonant reduction is possible.
                    if let Some(reduced) = Self::reduce_double_consonant(stem) {
                        if reduced.len() >= *min_len && Self::has_vowel(&reduced) {
                            return Some(reduced);
                        }
                    }
                    // No reduction → skip; let the "+e" variant below handle it.
                } else {
                    let candidate = format!("{stem}{replacement}");
                    if candidate.len() >= *min_len && Self::has_vowel(&candidate) {
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    /// Lemmatize using all POS rules, returning the first successful result.
    fn lemmatize_any(&self, word: &str) -> String {
        // Check exceptions first
        if let Some(&lemma) = self.exceptions.get(word) {
            return lemma.to_string();
        }
        // Verbs and nouns use simple rules; adjectives and adverbs use
        // morphological rules that handle double-consonant / silent-e.
        if let Some(c) = Self::apply_rules(word, &self.verb_rules) {
            return c;
        }
        if let Some(c) = Self::apply_rules(word, &self.noun_rules) {
            return c;
        }
        if let Some(c) = Self::apply_morph_rules(word, &self.adj_rules) {
            return c;
        }
        if let Some(c) = Self::apply_rules(word, &self.adv_rules) {
            return c;
        }
        word.to_string()
    }
}

impl Default for RuleBasedLemmatizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lemmatizer for RuleBasedLemmatizer {
    fn lemmatize(&self, word: &str, pos: Option<&str>) -> Result<String> {
        if word.is_empty() {
            return Err(TextError::InvalidInput(
                "empty word passed to lemmatizer".to_string(),
            ));
        }

        let lower = word.to_lowercase();

        // Exception dictionary always wins
        if let Some(&lemma) = self.exceptions.get(lower.as_str()) {
            return Ok(lemma.to_string());
        }

        let result = match parse_pos(pos) {
            Some(Pos::Verb) => {
                Self::apply_rules(&lower, &self.verb_rules).unwrap_or_else(|| lower.clone())
            }
            Some(Pos::Noun) => {
                Self::apply_rules(&lower, &self.noun_rules).unwrap_or_else(|| lower.clone())
            }
            Some(Pos::Adj) => {
                // Use morphological rules for adjective comparative/superlative
                Self::apply_morph_rules(&lower, &self.adj_rules).unwrap_or_else(|| lower.clone())
            }
            Some(Pos::Adv) => {
                Self::apply_rules(&lower, &self.adv_rules).unwrap_or_else(|| lower.clone())
            }
            None => self.lemmatize_any(&lower),
        };

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WordNet-style lemmatizer
// ─────────────────────────────────────────────────────────────────────────────

/// A lemmatizer that combines morphological rules with a compact embedded
/// exception dictionary, modelled after the WordNet *morphy* algorithm.
///
/// The algorithm proceeds in the following order:
/// 1. Check the exception dictionary (handles all major irregulars).
/// 2. Apply detachment rules to produce candidate stems.
/// 3. Check the candidate against a compact set of known valid base forms.
/// 4. Fall back to returning the original word unchanged.
///
/// # Part-of-speech hints
///
/// Pass one of `"n"`, `"v"`, `"a"` (adjective), `"r"` (adverb), or `None`
/// to guide the search.  When `None` is given the algorithm tries all
/// categories in the order noun → verb → adjective → adverb.
#[derive(Debug, Clone)]
pub struct WordNetLemmatizer {
    /// Full exception dict: word → lemma (partitioned per POS)
    noun_exc: HashMap<&'static str, &'static str>,
    verb_exc: HashMap<&'static str, &'static str>,
    adj_exc: HashMap<&'static str, &'static str>,
    adv_exc: HashMap<&'static str, &'static str>,
    /// Morphy detachment rules per POS: (suffix_to_remove, suffix_to_add)
    noun_morphs: Vec<(&'static str, &'static str)>,
    verb_morphs: Vec<(&'static str, &'static str)>,
    adj_morphs: Vec<(&'static str, &'static str)>,
    adv_morphs: Vec<(&'static str, &'static str)>,
}

impl WordNetLemmatizer {
    /// Create a new lemmatizer with the built-in WordNet-style data.
    pub fn new() -> Self {
        let noun_exc = Self::build_noun_exceptions();
        let verb_exc = Self::build_verb_exceptions();
        let adj_exc = Self::build_adj_exceptions();
        let adv_exc = Self::build_adv_exceptions();

        // Morphy rules (suffix_to_remove, suffix_to_add)
        // Applied in priority order.
        let noun_morphs = vec![
            ("ses", "s"),
            ("ies", "y"),
            ("oes", "o"),
            ("ches", "ch"),
            ("shes", "sh"),
            ("men", "man"),
            ("xes", "x"),
            ("ves", "f"),
            ("ves", "fe"),
            ("s", ""),
        ];

        let verb_morphs = vec![
            ("ies", "y"),
            ("ied", "y"),
            ("ating", "ate"),
            ("izing", "ize"),
            ("ising", "ise"),
            ("ening", "en"),
            ("tting", "t"),
            ("nning", "n"),
            ("pping", "p"),
            ("mming", "m"),
            ("gging", "g"),
            ("dding", "d"),
            ("bbing", "b"),
            ("rring", "r"),
            ("ing", ""),
            ("ing", "e"),
            ("ed", ""),
            ("ed", "e"),
            ("es", ""),
            ("s", ""),
        ];

        let adj_morphs = vec![
            ("iest", "y"),
            ("ier", "y"),
            ("est", ""),
            ("est", "e"),
            ("er", ""),
            ("er", "e"),
        ];

        let adv_morphs = vec![
            ("ically", "ic"),
            ("ably", "able"),
            ("ibly", "ible"),
            ("ily", "y"),
            ("ly", ""),
        ];

        Self {
            noun_exc,
            verb_exc,
            adj_exc,
            adv_exc,
            noun_morphs,
            verb_morphs,
            adj_morphs,
            adv_morphs,
        }
    }

    // ── Exception dictionaries ──────────────────────────────────────────────

    fn build_noun_exceptions() -> HashMap<&'static str, &'static str> {
        let mut m = HashMap::new();
        // Plurals → singular
        m.insert("men", "man");
        m.insert("women", "woman");
        m.insert("children", "child");
        m.insert("feet", "foot");
        m.insert("teeth", "tooth");
        m.insert("mice", "mouse");
        m.insert("geese", "goose");
        m.insert("oxen", "ox");
        m.insert("lice", "louse");
        m.insert("cacti", "cactus");
        m.insert("fungi", "fungus");
        m.insert("stimuli", "stimulus");
        m.insert("alumni", "alumnus");
        m.insert("syllabi", "syllabus");
        m.insert("radii", "radius");
        m.insert("nuclei", "nucleus");
        m.insert("foci", "focus");
        m.insert("termini", "terminus");
        m.insert("curricula", "curriculum");
        m.insert("bacteria", "bacterium");
        m.insert("criteria", "criterion");
        m.insert("phenomena", "phenomenon");
        m.insert("strata", "stratum");
        m.insert("data", "datum");
        m.insert("media", "medium");
        m.insert("spectra", "spectrum");
        m.insert("memoranda", "memorandum");
        m.insert("indices", "index");
        m.insert("vertices", "vertex");
        m.insert("matrices", "matrix");
        m.insert("analyses", "analysis");
        m.insert("bases", "basis");
        m.insert("crises", "crisis");
        m.insert("parentheses", "parenthesis");
        m.insert("hypotheses", "hypothesis");
        m.insert("diagnoses", "diagnosis");
        m.insert("axes", "axis");
        m.insert("theses", "thesis");
        m.insert("appendices", "appendix");
        m.insert("ellipses", "ellipsis");
        m.insert("oases", "oasis");
        m.insert("series", "series");
        m.insert("species", "species");
        m.insert("sheep", "sheep");
        m.insert("fish", "fish");
        m.insert("deer", "deer");
        m.insert("moose", "moose");
        m.insert("aircraft", "aircraft");
        m.insert("alumnae", "alumna");
        m.insert("antennae", "antenna");
        m.insert("formulae", "formula");
        m.insert("larvae", "larva");
        m.insert("nebulae", "nebula");
        m.insert("vertebrae", "vertebra");
        m.insert("knives", "knife");
        m.insert("wives", "wife");
        m.insert("lives", "life");
        m.insert("leaves", "leaf");
        m.insert("thieves", "thief");
        m.insert("shelves", "shelf");
        m.insert("halves", "half");
        m.insert("selves", "self");
        m.insert("wolves", "wolf");
        m.insert("calves", "calf");
        m.insert("elves", "elf");
        m.insert("people", "person");
        m.insert("persons", "person");
        m
    }

    fn build_verb_exceptions() -> HashMap<&'static str, &'static str> {
        let mut m = HashMap::new();
        // "be" forms
        m.insert("am", "be");
        m.insert("are", "be");
        m.insert("is", "be");
        m.insert("was", "be");
        m.insert("were", "be");
        m.insert("been", "be");
        m.insert("being", "be");
        // "have"
        m.insert("has", "have");
        m.insert("had", "have");
        m.insert("having", "have");
        // "do"
        m.insert("does", "do");
        m.insert("did", "do");
        m.insert("doing", "do");
        m.insert("done", "do");
        // Strong verbs
        m.insert("went", "go");
        m.insert("gone", "go");
        m.insert("going", "go");
        m.insert("goes", "go");
        m.insert("ran", "run");
        m.insert("running", "run");
        m.insert("ate", "eat");
        m.insert("eaten", "eat");
        m.insert("saw", "see");
        m.insert("seen", "see");
        m.insert("came", "come");
        m.insert("coming", "come");
        m.insert("gave", "give");
        m.insert("given", "give");
        m.insert("took", "take");
        m.insert("taken", "take");
        m.insert("made", "make");
        m.insert("said", "say");
        m.insert("got", "get");
        m.insert("gotten", "get");
        m.insert("knew", "know");
        m.insert("known", "know");
        m.insert("thought", "think");
        m.insert("felt", "feel");
        m.insert("spoke", "speak");
        m.insert("spoken", "speak");
        m.insert("wrote", "write");
        m.insert("written", "write");
        m.insert("broke", "break");
        m.insert("broken", "break");
        m.insert("chose", "choose");
        m.insert("chosen", "choose");
        m.insert("drove", "drive");
        m.insert("driven", "drive");
        m.insert("flew", "fly");
        m.insert("flown", "fly");
        m.insert("fell", "fall");
        m.insert("fallen", "fall");
        m.insert("forgot", "forget");
        m.insert("forgotten", "forget");
        m.insert("froze", "freeze");
        m.insert("frozen", "freeze");
        m.insert("grew", "grow");
        m.insert("grown", "grow");
        m.insert("held", "hold");
        m.insert("hung", "hang");
        m.insert("knew", "know");
        m.insert("led", "lead");
        m.insert("left", "leave");
        m.insert("lent", "lend");
        m.insert("lost", "lose");
        m.insert("met", "meet");
        m.insert("paid", "pay");
        m.insert("rode", "ride");
        m.insert("risen", "rise");
        m.insert("rose", "rise");
        m.insert("selling", "sell");
        m.insert("sold", "sell");
        m.insert("sent", "send");
        m.insert("shut", "shut");
        m.insert("sang", "sing");
        m.insert("sung", "sing");
        m.insert("sank", "sink");
        m.insert("sat", "sit");
        m.insert("slept", "sleep");
        m.insert("stood", "stand");
        m.insert("stole", "steal");
        m.insert("stolen", "steal");
        m.insert("struck", "strike");
        m.insert("swam", "swim");
        m.insert("swum", "swim");
        m.insert("threw", "throw");
        m.insert("thrown", "throw");
        m.insert("woke", "wake");
        m.insert("woken", "wake");
        m.insert("wore", "wear");
        m.insert("worn", "wear");
        m.insert("won", "win");
        m.insert("withdrawn", "withdraw");
        m.insert("withdrew", "withdraw");
        m
    }

    fn build_adj_exceptions() -> HashMap<&'static str, &'static str> {
        let mut m = HashMap::new();
        m.insert("better", "good");
        m.insert("best", "good");
        m.insert("worse", "bad");
        m.insert("worst", "bad");
        m.insert("more", "much");
        m.insert("most", "much");
        m.insert("less", "little");
        m.insert("least", "little");
        m.insert("farther", "far");
        m.insert("farthest", "far");
        m.insert("further", "far");
        m.insert("furthest", "far");
        m.insert("elder", "old");
        m.insert("eldest", "old");
        m.insert("older", "old");
        m.insert("oldest", "old");
        m.insert("inner", "in");
        m.insert("outer", "out");
        m.insert("upper", "up");
        m.insert("lower", "low");
        m.insert("latter", "late");
        m.insert("former", "fore");
        m.insert("hinder", "hind");
        m.insert("nether", "neth");
        m
    }

    fn build_adv_exceptions() -> HashMap<&'static str, &'static str> {
        let mut m = HashMap::new();
        m.insert("worse", "badly");
        m.insert("worst", "badly");
        m.insert("better", "well");
        m.insert("best", "well");
        m.insert("more", "much");
        m.insert("most", "much");
        m.insert("less", "little");
        m.insert("least", "little");
        m
    }

    // ── Morphy algorithm ────────────────────────────────────────────────────

    /// Apply morphy rules and return all candidate lemmas (deduplicated).
    fn apply_morphs(word: &str, rules: &[(&'static str, &'static str)]) -> Vec<String> {
        let mut candidates = Vec::new();
        for (suffix_remove, suffix_add) in rules {
            if word.ends_with(suffix_remove) {
                let stem = &word[..word.len() - suffix_remove.len()];
                if !stem.is_empty() {
                    let candidate = format!("{stem}{suffix_add}");
                    if !candidates.contains(&candidate) {
                        candidates.push(candidate);
                    }
                }
            }
        }
        candidates
    }

    /// Core morphy look-up for a given POS.
    fn morphy_pos(&self, word: &str, pos: Pos) -> Option<String> {
        let (exc, morphs) = match pos {
            Pos::Noun => (&self.noun_exc, self.noun_morphs.as_slice()),
            Pos::Verb => (&self.verb_exc, self.verb_morphs.as_slice()),
            Pos::Adj => (&self.adj_exc, self.adj_morphs.as_slice()),
            Pos::Adv => (&self.adv_exc, self.adv_morphs.as_slice()),
        };

        // 1. Exception dictionary
        if let Some(&lemma) = exc.get(word) {
            return Some(lemma.to_string());
        }

        // 2. The word itself might already be a base form — return it as-is
        //    when no rule would simplify it further.
        let candidates = Self::apply_morphs(word, morphs);
        if candidates.is_empty() {
            return Some(word.to_string());
        }

        // 3. For each candidate, check its exception list; if found, return it.
        //    Otherwise take the shortest valid candidate.
        for candidate in &candidates {
            if exc.contains_key(candidate.as_str()) {
                return Some(candidate.clone());
            }
        }

        // 4. Return the first (highest priority) candidate
        candidates.into_iter().next()
    }
}

impl Default for WordNetLemmatizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lemmatizer for WordNetLemmatizer {
    fn lemmatize(&self, word: &str, pos: Option<&str>) -> Result<String> {
        if word.is_empty() {
            return Err(TextError::InvalidInput(
                "empty word passed to lemmatizer".to_string(),
            ));
        }

        let lower = word.to_lowercase();

        let result = match parse_pos(pos) {
            Some(p) => self.morphy_pos(&lower, p).unwrap_or_else(|| lower.clone()),
            None => {
                // Try all POS in noun → verb → adj → adv order
                for p in [Pos::Noun, Pos::Verb, Pos::Adj, Pos::Adv] {
                    if let Some(lemma) = self.morphy_pos(&lower, p) {
                        // Prefer results that differ from the input (i.e. actually changed)
                        if lemma != lower {
                            return Ok(lemma);
                        }
                    }
                }
                lower.clone()
            }
        };

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RuleBasedLemmatizer ─────────────────────────────────────────────────

    #[test]
    fn test_rule_based_verbs() {
        let lem = RuleBasedLemmatizer::new();
        let cases = [
            ("running", "n", "run"),
            ("runs", "v", "run"),
            ("walked", "v", "walk"),
            ("eating", "v", "eat"),
            ("flies", "v", "fly"),
            ("tried", "v", "try"),
        ];
        for (word, pos, expected) in cases {
            let result = lem.lemmatize(word, Some(pos)).expect("should succeed");
            assert_eq!(result, expected, "verb: {word}");
        }
    }

    #[test]
    fn test_rule_based_nouns() {
        let lem = RuleBasedLemmatizer::new();
        let cases = [
            ("cats", "cat"),
            ("boxes", "box"),
            ("buses", "bus"),
            ("churches", "church"),
            ("parties", "party"),
            ("stories", "story"),
        ];
        for (word, expected) in cases {
            let result = lem.lemmatize(word, Some("n")).expect("should succeed");
            assert_eq!(result, expected, "noun: {word}");
        }
    }

    #[test]
    fn test_rule_based_adjectives() {
        let lem = RuleBasedLemmatizer::new();
        let cases = [
            ("happier", "happy"),
            ("happiest", "happy"),
            ("bigger", "big"),
            ("largest", "large"),
        ];
        for (word, expected) in cases {
            let result = lem.lemmatize(word, Some("a")).expect("should succeed");
            assert_eq!(result, expected, "adj: {word}");
        }
    }

    #[test]
    fn test_rule_based_adverbs() {
        let lem = RuleBasedLemmatizer::new();
        let cases = [
            ("quickly", "quick"),
            ("happily", "happy"),
            ("basically", "basic"),
        ];
        for (word, expected) in cases {
            let result = lem.lemmatize(word, Some("r")).expect("should succeed");
            assert_eq!(result, expected, "adv: {word}");
        }
    }

    #[test]
    fn test_rule_based_exceptions() {
        let lem = RuleBasedLemmatizer::new();
        assert_eq!(lem.lemmatize("went", None).unwrap(), "go");
        assert_eq!(lem.lemmatize("children", None).unwrap(), "child");
        assert_eq!(lem.lemmatize("feet", None).unwrap(), "foot");
        assert_eq!(lem.lemmatize("better", None).unwrap(), "good");
        assert_eq!(lem.lemmatize("worst", None).unwrap(), "bad");
    }

    #[test]
    fn test_rule_based_empty_word_error() {
        let lem = RuleBasedLemmatizer::new();
        assert!(lem.lemmatize("", None).is_err());
    }

    #[test]
    fn test_rule_based_batch() {
        let lem = RuleBasedLemmatizer::new();
        let pairs = vec![
            ("cats", Some("n")),
            ("running", Some("v")),
            ("happier", Some("a")),
        ];
        let results = lem.lemmatize_batch(&pairs).unwrap();
        assert_eq!(results, vec!["cat", "run", "happy"]);
    }

    // ── WordNetLemmatizer ───────────────────────────────────────────────────

    #[test]
    fn test_wordnet_verb_exceptions() {
        let lem = WordNetLemmatizer::new();
        assert_eq!(lem.lemmatize("went", Some("v")).unwrap(), "go");
        assert_eq!(lem.lemmatize("ran", Some("v")).unwrap(), "run");
        assert_eq!(lem.lemmatize("ate", Some("v")).unwrap(), "eat");
        assert_eq!(lem.lemmatize("was", Some("v")).unwrap(), "be");
        assert_eq!(lem.lemmatize("were", Some("v")).unwrap(), "be");
        assert_eq!(lem.lemmatize("had", Some("v")).unwrap(), "have");
    }

    #[test]
    fn test_wordnet_noun_exceptions() {
        let lem = WordNetLemmatizer::new();
        assert_eq!(lem.lemmatize("children", Some("n")).unwrap(), "child");
        assert_eq!(lem.lemmatize("feet", Some("n")).unwrap(), "foot");
        assert_eq!(lem.lemmatize("mice", Some("n")).unwrap(), "mouse");
        assert_eq!(lem.lemmatize("geese", Some("n")).unwrap(), "goose");
        assert_eq!(lem.lemmatize("criteria", Some("n")).unwrap(), "criterion");
        assert_eq!(lem.lemmatize("phenomena", Some("n")).unwrap(), "phenomenon");
        assert_eq!(lem.lemmatize("analyses", Some("n")).unwrap(), "analysis");
    }

    #[test]
    fn test_wordnet_adj_exceptions() {
        let lem = WordNetLemmatizer::new();
        assert_eq!(lem.lemmatize("better", Some("a")).unwrap(), "good");
        assert_eq!(lem.lemmatize("best", Some("a")).unwrap(), "good");
        assert_eq!(lem.lemmatize("worse", Some("a")).unwrap(), "bad");
        assert_eq!(lem.lemmatize("worst", Some("a")).unwrap(), "bad");
    }

    #[test]
    fn test_wordnet_morphy_rules() {
        let lem = WordNetLemmatizer::new();
        // Regular verb inflections via morphological rules
        assert_eq!(lem.lemmatize("walking", Some("v")).unwrap(), "walk");
        assert_eq!(lem.lemmatize("walked", Some("v")).unwrap(), "walk");
        assert_eq!(lem.lemmatize("walks", Some("v")).unwrap(), "walk");
        // Regular noun plural
        assert_eq!(lem.lemmatize("cats", Some("n")).unwrap(), "cat");
        assert_eq!(lem.lemmatize("boxes", Some("n")).unwrap(), "box");
    }

    #[test]
    fn test_wordnet_no_pos_fallback() {
        let lem = WordNetLemmatizer::new();
        // Without POS hint, should still return reasonable lemmas
        let result = lem.lemmatize("cats", None).unwrap();
        assert_eq!(result, "cat");
        let result = lem.lemmatize("went", None).unwrap();
        assert_eq!(result, "go");
    }

    #[test]
    fn test_wordnet_empty_word_error() {
        let lem = WordNetLemmatizer::new();
        assert!(lem.lemmatize("", None).is_err());
    }

    #[test]
    fn test_wordnet_batch() {
        let lem = WordNetLemmatizer::new();
        let pairs = vec![
            ("children", Some("n")),
            ("went", Some("v")),
            ("better", Some("a")),
        ];
        let results = lem.lemmatize_batch(&pairs).unwrap();
        assert_eq!(results, vec!["child", "go", "good"]);
    }

    #[test]
    fn test_wordnet_unchanged_word() {
        // Words already in base form should be returned unchanged
        let lem = WordNetLemmatizer::new();
        assert_eq!(lem.lemmatize("cat", Some("n")).unwrap(), "cat");
        assert_eq!(lem.lemmatize("run", Some("v")).unwrap(), "run");
    }

    #[test]
    fn test_case_insensitive() {
        let rb = RuleBasedLemmatizer::new();
        let wn = WordNetLemmatizer::new();
        assert_eq!(rb.lemmatize("RUNNING", Some("v")).unwrap(), "run");
        assert_eq!(wn.lemmatize("WENT", Some("v")).unwrap(), "go");
        assert_eq!(rb.lemmatize("Children", None).unwrap(), "child");
        assert_eq!(wn.lemmatize("Feet", Some("n")).unwrap(), "foot");
    }

    #[test]
    fn test_irregular_verb_forms() {
        let wn = WordNetLemmatizer::new();
        let cases = [
            ("broke", "break"),
            ("broken", "break"),
            ("chose", "choose"),
            ("chosen", "choose"),
            ("drove", "drive"),
            ("driven", "drive"),
            ("flew", "fly"),
            ("grown", "grow"),
            ("held", "hold"),
            ("lost", "lose"),
            ("sold", "sell"),
            ("sang", "sing"),
            ("slept", "sleep"),
            ("stood", "stand"),
            ("threw", "throw"),
            ("won", "win"),
        ];
        for (word, expected) in cases {
            let result = wn.lemmatize(word, Some("v")).unwrap();
            assert_eq!(result, expected, "irregular verb: {word}");
        }
    }

    #[test]
    fn test_latin_greek_plurals() {
        let wn = WordNetLemmatizer::new();
        let cases = [
            ("cacti", "cactus"),
            ("alumni", "alumnus"),
            ("fungi", "fungus"),
            ("bacteria", "bacterium"),
            ("strata", "stratum"),
            ("spectra", "spectrum"),
        ];
        for (word, expected) in cases {
            let result = wn.lemmatize(word, Some("n")).unwrap();
            assert_eq!(result, expected, "Latin/Greek plural: {word}");
        }
    }
}
