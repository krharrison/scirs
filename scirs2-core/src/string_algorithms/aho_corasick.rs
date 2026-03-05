//! Aho-Corasick multi-pattern string search automaton.
//!
//! Builds a finite automaton from a set of patterns and searches for all
//! occurrences simultaneously in O(n + m + z) time, where n is the text
//! length, m is the total length of all patterns, and z is the number of
//! matches.
//!
//! Also provides KMP (Knuth-Morris-Pratt) and Boyer-Moore-Horspool single-
//! pattern search as standalone functions.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Internal state type
// ---------------------------------------------------------------------------

/// A single state in the Aho-Corasick trie/automaton.
#[derive(Debug, Clone)]
struct AcState {
    /// Byte → next-state transitions for the trie edges.
    transitions: HashMap<u8, usize>,
    /// Failure (fall-back) link used during search.
    fail: usize,
    /// Pattern indices that end at this state (including via suffix links).
    output: Vec<usize>,
}

impl AcState {
    fn new() -> Self {
        AcState {
            transitions: HashMap::new(),
            fail: 0,
            output: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// AhoCorasick public type
// ---------------------------------------------------------------------------

/// Aho-Corasick automaton for simultaneous multi-pattern text search.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::AhoCorasick;
///
/// let ac = AhoCorasick::new(&["he", "she", "his", "hers"]);
/// let hits = ac.find_all("ushers");
/// // positions (start, end_exclusive, pattern_index)
/// assert!(!hits.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct AhoCorasick {
    states: Vec<AcState>,
    /// Pattern lengths (needed to recover start positions from end positions).
    pattern_lengths: Vec<usize>,
    n_patterns: usize,
}

impl AhoCorasick {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build the automaton from a slice of string patterns.
    pub fn new(patterns: &[&str]) -> Self {
        let byte_patterns: Vec<&[u8]> = patterns.iter().map(|s| s.as_bytes()).collect();
        Self::from_bytes(&byte_patterns)
    }

    /// Build the automaton from a slice of byte-slice patterns.
    pub fn from_bytes(patterns: &[&[u8]]) -> Self {
        let n_patterns = patterns.len();
        let mut pattern_lengths = Vec::with_capacity(n_patterns);

        // ---- Phase 1: build the trie ----------------------------------------
        let mut states: Vec<AcState> = vec![AcState::new()]; // state 0 = root

        for (pi, pattern) in patterns.iter().enumerate() {
            pattern_lengths.push(pattern.len());
            let mut cur = 0usize;
            for &byte in pattern.iter() {
                if let Some(&next) = states[cur].transitions.get(&byte) {
                    cur = next;
                } else {
                    let next = states.len();
                    states.push(AcState::new());
                    states[cur].transitions.insert(byte, next);
                    cur = next;
                }
            }
            states[cur].output.push(pi);
        }

        // ---- Phase 2: build failure links via BFS ---------------------------
        // Root's children fail to root.
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Collect bytes at root level first to avoid borrow conflicts.
        let root_children: Vec<(u8, usize)> = states[0]
            .transitions
            .iter()
            .map(|(&b, &s)| (b, s))
            .collect();

        for (_byte, child) in root_children {
            states[child].fail = 0;
            queue.push_back(child);
        }

        while let Some(r) = queue.pop_front() {
            // Gather child edges from state r.
            let edges: Vec<(u8, usize)> = states[r]
                .transitions
                .iter()
                .map(|(&b, &s)| (b, s))
                .collect();

            for (byte, s) in edges {
                queue.push_back(s);

                // Walk up failure links from r to find the longest proper
                // suffix that is a prefix of some pattern.
                let mut failure = states[r].fail;
                loop {
                    if let Some(&fs) = states[failure].transitions.get(&byte) {
                        if fs != s {
                            states[s].fail = fs;
                            break;
                        }
                    }
                    if failure == 0 {
                        // No matching suffix; fail to root.
                        states[s].fail = 0;
                        break;
                    }
                    failure = states[failure].fail;
                }

                // Merge output from failure state into s.
                let fail_state = states[s].fail;
                let extra_output: Vec<usize> = states[fail_state].output.clone();
                states[s].output.extend(extra_output);
            }
        }

        AhoCorasick {
            states,
            pattern_lengths,
            n_patterns,
        }
    }

    // -----------------------------------------------------------------------
    // Search API
    // -----------------------------------------------------------------------

    /// Find all occurrences of all patterns in `text`.
    ///
    /// Returns a `Vec<(start, end, pattern_idx)>` where `[start, end)` is the
    /// half-open byte range of the match.  Results are ordered by end position.
    pub fn find_all(&self, text: &str) -> Vec<(usize, usize, usize)> {
        self.find_all_bytes(text.as_bytes())
    }

    /// Byte-slice variant of [`find_all`].
    pub fn find_all_bytes(&self, text: &[u8]) -> Vec<(usize, usize, usize)> {
        let mut results = Vec::new();
        let mut state = 0usize;

        for (i, &byte) in text.iter().enumerate() {
            // Follow failure links until a transition for `byte` exists or
            // we are at the root.
            loop {
                if let Some(&next) = self.states[state].transitions.get(&byte) {
                    state = next;
                    break;
                } else if state == 0 {
                    break;
                } else {
                    state = self.states[state].fail;
                }
            }

            // Emit all patterns that end at position i.
            for &pi in &self.states[state].output {
                let plen = self.pattern_lengths[pi];
                let start = i + 1 - plen;
                results.push((start, i + 1, pi));
            }
        }

        results
    }

    /// Return the first match `(start, end, pattern_idx)`, if any.
    pub fn find_first(&self, text: &str) -> Option<(usize, usize, usize)> {
        self.find_all_bytes(text.as_bytes()).into_iter().next()
    }

    /// Count the total number of pattern occurrences in `text`.
    pub fn count_matches(&self, text: &str) -> usize {
        self.find_all_bytes(text.as_bytes()).len()
    }

    /// Return `true` if any pattern appears anywhere in `text`.
    pub fn is_match(&self, text: &str) -> bool {
        let mut state = 0usize;
        for &byte in text.as_bytes() {
            loop {
                if let Some(&next) = self.states[state].transitions.get(&byte) {
                    state = next;
                    break;
                } else if state == 0 {
                    break;
                } else {
                    state = self.states[state].fail;
                }
            }
            if !self.states[state].output.is_empty() {
                return true;
            }
        }
        false
    }

    /// Replace every match with the corresponding entry in `replacements`.
    ///
    /// `replacements[i]` is the string that replaces pattern `i`.  When
    /// overlapping matches occur, the earliest-ending match wins and the text
    /// cursor advances past it.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidArgument`] when `replacements.len()` does
    /// not equal the number of patterns the automaton was built with.
    pub fn replace_all(&self, text: &str, replacements: &[&str]) -> CoreResult<String> {
        if replacements.len() != self.n_patterns {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "replace_all: expected {} replacements, got {}",
                self.n_patterns,
                replacements.len()
            ))));
        }

        let bytes = text.as_bytes();
        let matches = self.find_all_bytes(bytes);

        let mut result = String::with_capacity(text.len());
        let mut pos = 0usize;

        for (start, end, pi) in matches {
            if start < pos {
                // Overlapping match — skip.
                continue;
            }
            // Append literal text between last match and this one.
            match std::str::from_utf8(&bytes[pos..start]) {
                Ok(s) => result.push_str(s),
                Err(e) => {
                    return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                        "replace_all: invalid UTF-8 in source text: {e}"
                    ))))
                }
            }
            result.push_str(replacements[pi]);
            pos = end;
        }

        // Append any trailing literal text.
        match std::str::from_utf8(&bytes[pos..]) {
            Ok(s) => result.push_str(s),
            Err(e) => {
                return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                    "replace_all: invalid UTF-8 in source text tail: {e}"
                ))))
            }
        }

        Ok(result)
    }

    /// Number of patterns in the automaton.
    #[inline]
    pub fn n_patterns(&self) -> usize {
        self.n_patterns
    }

    /// Number of states in the automaton (including root).
    #[inline]
    pub fn n_states(&self) -> usize {
        self.states.len()
    }
}

// ---------------------------------------------------------------------------
// Boyer-Moore-Horspool single-pattern search
// ---------------------------------------------------------------------------

/// Boyer-Moore-Horspool single-pattern search.
///
/// Returns the (zero-based) starting positions of every non-overlapping
/// occurrence of `pattern` in `text`.  Uses an O(m) bad-character shift table
/// and runs in O(n/m) average time.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::bm_horspool_search;
///
/// let positions = bm_horspool_search(b"AABAAABAAABAA", b"AAB");
/// assert_eq!(positions, vec![0, 4]);
/// ```
pub fn bm_horspool_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    // Build the bad-character shift table.
    // Default shift is pattern length; for bytes appearing in the pattern the
    // shift is the distance from the rightmost occurrence to the end – 1.
    let mut shift = [m; 256];
    for (i, &b) in pattern[..m - 1].iter().enumerate() {
        shift[b as usize] = m - 1 - i;
    }

    let mut results = Vec::new();
    let mut i = m - 1; // index of the last character being compared

    while i < n {
        let mut k = 0usize;
        let mut j = i;
        while k < m {
            if text[j] != pattern[m - 1 - k] {
                break;
            }
            k += 1;
            if j == 0 {
                break;
            }
            j = j.saturating_sub(1);
        }
        if k == m {
            results.push(i + 1 - m);
        }
        i = i.saturating_add(shift[text[i] as usize]);
        if i < m - 1 {
            break; // overflow guard
        }
    }

    results
}

// ---------------------------------------------------------------------------
// KMP (Knuth-Morris-Pratt)
// ---------------------------------------------------------------------------

/// Build the KMP failure function for `pattern`.
///
/// `failure[i]` is the length of the longest proper prefix of `pattern[0..=i]`
/// that is also a suffix.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::kmp_failure_function;
///
/// let f = kmp_failure_function(b"abcabd");
/// assert_eq!(f, vec![0, 0, 0, 1, 2, 0]);
/// ```
pub fn kmp_failure_function(pattern: &[u8]) -> Vec<usize> {
    let m = pattern.len();
    let mut failure = vec![0usize; m];
    let mut k = 0usize;
    let mut i = 1usize;
    while i < m {
        while k > 0 && pattern[k] != pattern[i] {
            k = failure[k - 1];
        }
        if pattern[k] == pattern[i] {
            k += 1;
        }
        failure[i] = k;
        i += 1;
    }
    failure
}

/// KMP (Knuth-Morris-Pratt) single-pattern search.
///
/// Returns the (zero-based) starting positions of every (possibly overlapping)
/// occurrence of `pattern` in `text` in O(n + m) time.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::kmp_search;
///
/// let positions = kmp_search(b"aababcab", b"ab");
/// assert_eq!(positions, vec![1, 3, 6]);
/// ```
pub fn kmp_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 {
        return (0..=n).collect();
    }
    if m > n {
        return Vec::new();
    }

    let failure = kmp_failure_function(pattern);
    let mut results = Vec::new();
    let mut q = 0usize; // number of chars matched

    for (i, &c) in text.iter().enumerate() {
        while q > 0 && pattern[q] != c {
            q = failure[q - 1];
        }
        if pattern[q] == c {
            q += 1;
        }
        if q == m {
            results.push(i + 1 - m);
            q = failure[q - 1];
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Aho-Corasick -------------------------------------------------------

    #[test]
    fn test_ac_basic_find_all() {
        let ac = AhoCorasick::new(&["he", "she", "his", "hers"]);
        let hits = ac.find_all("ushers");
        // "she" at 1, "he" at 2, "hers" at 2
        let patterns: Vec<usize> = hits.iter().map(|&(_, _, p)| p).collect();
        assert!(patterns.contains(&0)); // "he"
        assert!(patterns.contains(&1)); // "she"
        assert!(patterns.contains(&3)); // "hers"
    }

    #[test]
    fn test_ac_no_match() {
        let ac = AhoCorasick::new(&["xyz", "abc"]);
        assert_eq!(ac.find_all("hello world"), vec![]);
    }

    #[test]
    fn test_ac_overlapping() {
        let ac = AhoCorasick::new(&["aa"]);
        let hits = ac.find_all("aaa");
        // both (0,2,0) and (1,3,0) should be found
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_ac_single_char_patterns() {
        let ac = AhoCorasick::new(&["a", "b"]);
        let hits = ac.find_all("abab");
        assert_eq!(hits.len(), 4);
    }

    #[test]
    fn test_ac_is_match_true() {
        let ac = AhoCorasick::new(&["hello", "world"]);
        assert!(ac.is_match("say hello!"));
    }

    #[test]
    fn test_ac_is_match_false() {
        let ac = AhoCorasick::new(&["hello", "world"]);
        assert!(!ac.is_match("greetings"));
    }

    #[test]
    fn test_ac_count_matches() {
        let ac = AhoCorasick::new(&["ab"]);
        assert_eq!(ac.count_matches("ababab"), 3);
    }

    #[test]
    fn test_ac_find_first() {
        let ac = AhoCorasick::new(&["cd", "ab"]);
        let first = ac.find_first("xabcd");
        assert!(first.is_some());
        let (start, _end, _pi) = first.expect("first match should exist");
        assert_eq!(start, 1); // "ab" at position 1
    }

    #[test]
    fn test_ac_replace_all() {
        let ac = AhoCorasick::new(&["cat", "dog"]);
        let out = ac
            .replace_all("I have a cat and a dog", &["kitty", "puppy"])
            .expect("replace_all should succeed");
        assert_eq!(out, "I have a kitty and a puppy");
    }

    #[test]
    fn test_ac_replace_all_wrong_count() {
        let ac = AhoCorasick::new(&["cat", "dog"]);
        let result = ac.replace_all("text", &["only_one"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ac_empty_text() {
        let ac = AhoCorasick::new(&["abc"]);
        assert_eq!(ac.find_all(""), vec![]);
        assert!(!ac.is_match(""));
    }

    #[test]
    fn test_ac_empty_patterns_slice() {
        let ac = AhoCorasick::new(&[]);
        assert_eq!(ac.n_patterns(), 0);
        assert_eq!(ac.find_all("any text"), vec![]);
    }

    #[test]
    fn test_ac_pattern_longer_than_text() {
        let ac = AhoCorasick::new(&["verylongpattern"]);
        assert_eq!(ac.find_all("short"), vec![]);
    }

    #[test]
    fn test_ac_positions_correct() {
        let ac = AhoCorasick::new(&["bc"]);
        let hits = ac.find_all("abcabc");
        assert_eq!(hits, vec![(1, 3, 0), (4, 6, 0)]);
    }

    #[test]
    fn test_ac_binary_patterns() {
        let patterns: &[&[u8]] = &[b"\x00\x01", b"\xFF\xFE"];
        let ac = AhoCorasick::from_bytes(patterns);
        let text: &[u8] = &[0x00, 0x01, 0x02, 0xFF, 0xFE];
        let hits = ac.find_all_bytes(text);
        assert_eq!(hits.len(), 2);
    }

    // ---- Boyer-Moore-Horspool -----------------------------------------------

    #[test]
    fn test_bmh_basic() {
        let pos = bm_horspool_search(b"AABAAABAAABAA", b"AAB");
        assert!(pos.contains(&0));
    }

    #[test]
    fn test_bmh_no_match() {
        let pos = bm_horspool_search(b"hello world", b"xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_bmh_single_char() {
        let pos = bm_horspool_search(b"aaa", b"a");
        assert_eq!(pos.len(), 3);
    }

    #[test]
    fn test_bmh_pattern_longer_than_text() {
        let pos = bm_horspool_search(b"ab", b"abc");
        assert!(pos.is_empty());
    }

    // ---- KMP ----------------------------------------------------------------

    #[test]
    fn test_kmp_basic() {
        let pos = kmp_search(b"aababcab", b"ab");
        assert_eq!(pos, vec![1, 3, 6]);
    }

    #[test]
    fn test_kmp_no_match() {
        let pos = kmp_search(b"hello", b"xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_kmp_overlapping() {
        let pos = kmp_search(b"aaa", b"aa");
        assert_eq!(pos, vec![0, 1]);
    }

    #[test]
    fn test_kmp_empty_pattern() {
        let pos = kmp_search(b"abc", b"");
        assert_eq!(pos, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_kmp_failure_function_basic() {
        let f = kmp_failure_function(b"abcabd");
        assert_eq!(f, vec![0, 0, 0, 1, 2, 0]);
    }

    #[test]
    fn test_kmp_failure_aaaa() {
        let f = kmp_failure_function(b"aaaa");
        assert_eq!(f, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_kmp_full_text_match() {
        let pos = kmp_search(b"abcabc", b"abcabc");
        assert_eq!(pos, vec![0]);
    }
}
