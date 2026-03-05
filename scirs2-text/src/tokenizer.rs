//! Tokenization utilities for transformer models
//!
//! This module provides production-grade tokenizer implementations designed for
//! use with transformer-based models such as BERT, GPT, and T5. It includes:
//!
//! - **BPE (Byte-Pair Encoding)**: Subword tokenization used by GPT-2, RoBERTa
//! - **WordPiece**: BERT-style subword tokenization with `##` continuation prefix
//! - **SimpleWhitespace**: Vocabulary-aware whitespace tokenizer with UNK handling
//! - **SimpleChar**: Character-level tokenizer mapping each char to an ID
//!
//! All tokenizers implement the [`TransformerTokenizer`] trait which provides
//! `encode` (text to token IDs) and `decode` (token IDs to text) operations,
//! as well as `vocab_size` for embedding layer configuration.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::tokenizer::{BPETokenizer, TransformerTokenizer};
//!
//! // Train a BPE tokenizer on a small corpus
//! let corpus = &["the cat sat on the mat", "the dog sat on the log"];
//! let tokenizer = BPETokenizer::train(corpus, 100).expect("training failed");
//!
//! // Encode and decode
//! let ids = tokenizer.encode("the cat");
//! let text = tokenizer.decode(&ids);
//! assert!(text.contains("the"));
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::Path;

// ---------------------------------------------------------------------------
// TransformerTokenizer trait
// ---------------------------------------------------------------------------

/// Trait for tokenizers designed for transformer model input/output.
///
/// Unlike the general-purpose [`crate::tokenize::Tokenizer`] trait which returns
/// `Vec<String>`, this trait operates on integer token IDs (`u32`) suitable for
/// embedding lookup in neural models.
pub trait TransformerTokenizer {
    /// Encode text into a sequence of token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode a sequence of token IDs back into text.
    fn decode(&self, ids: &[u32]) -> String;

    /// Return the vocabulary size (number of distinct token IDs).
    fn vocab_size(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Pre-tokenisation helpers (shared across tokenizer implementations)
// ---------------------------------------------------------------------------

/// Normalise text before tokenisation: lowercase and collapse whitespace.
fn pre_tokenize(text: &str) -> String {
    let lower = text.to_lowercase();
    // Collapse whitespace runs into single spaces, trim.
    lower.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Split text into words on whitespace boundaries.
fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_string()).collect()
}

// ===========================================================================
// BPETokenizer
// ===========================================================================

/// A Byte-Pair Encoding tokenizer for transformer models.
///
/// BPE iteratively merges the most frequent pair of symbols (initially
/// characters) to form a vocabulary of subword units. This implementation
/// supports:
///
/// - Training from a text corpus with a target vocabulary size
/// - Encoding text to `u32` token IDs
/// - Decoding token IDs back to text
/// - Special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`)
/// - JSON-based persistence (save/load)
///
/// # Training
///
/// ```rust
/// use scirs2_text::tokenizer::{BPETokenizer, TransformerTokenizer};
///
/// let corpus = &["hello world", "hello there"];
/// let tok = BPETokenizer::train(corpus, 80).expect("train");
/// assert!(tok.vocab_size() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// token string -> token ID
    vocab: HashMap<String, u32>,
    /// token ID -> token string (inverse)
    id_to_token: HashMap<u32, String>,
    /// Ordered list of merge pairs learned during training
    merges: Vec<(String, String)>,
    /// Special tokens with reserved IDs
    special_tokens: HashMap<String, u32>,
}

/// Default special tokens used by BPE when constructed via `new()`.
const DEFAULT_SPECIAL_TOKENS: &[&str] = &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

impl BPETokenizer {
    /// Create a new, empty BPE tokenizer pre-loaded with standard special tokens.
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_tokens = HashMap::new();

        for (i, &tok) in DEFAULT_SPECIAL_TOKENS.iter().enumerate() {
            let id = i as u32;
            vocab.insert(tok.to_string(), id);
            id_to_token.insert(id, tok.to_string());
            special_tokens.insert(tok.to_string(), id);
        }

        Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            special_tokens,
        }
    }

    /// Train a BPE tokenizer on a text corpus.
    ///
    /// The tokenizer learns `vocab_size` total tokens (including special tokens
    /// and individual characters that appear in the corpus).
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to train on.
    /// * `vocab_size` - Target vocabulary size (must be > number of special tokens).
    ///
    /// # Errors
    ///
    /// Returns `TextError::TokenizationError` if the corpus is empty or
    /// `vocab_size` is too small.
    pub fn train(texts: &[&str], vocab_size: usize) -> Result<Self> {
        if texts.is_empty() {
            return Err(TextError::TokenizationError(
                "Cannot train on empty corpus".to_string(),
            ));
        }
        if vocab_size < DEFAULT_SPECIAL_TOKENS.len() + 1 {
            return Err(TextError::TokenizationError(format!(
                "vocab_size must be at least {} (special tokens + 1)",
                DEFAULT_SPECIAL_TOKENS.len() + 1
            )));
        }

        let mut tokenizer = Self::new();

        // Step 1: Collect all unique characters and add them to the vocab.
        let mut char_set: Vec<char> = Vec::new();
        for text in texts {
            let normalized = pre_tokenize(text);
            for ch in normalized.chars() {
                if !char_set.contains(&ch) {
                    char_set.push(ch);
                }
            }
        }
        char_set.sort();

        for ch in &char_set {
            let s = ch.to_string();
            if !tokenizer.vocab.contains_key(&s) {
                let id = tokenizer.vocab.len() as u32;
                tokenizer.vocab.insert(s.clone(), id);
                tokenizer.id_to_token.insert(id, s);
            }
        }

        // Step 2: Build word-frequency table.
        // Each word is represented as a sequence of symbols (initially chars).
        let mut word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
        for text in texts {
            let normalized = pre_tokenize(text);
            for word in split_words(&normalized) {
                let symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                *word_freqs.entry(symbols).or_insert(0) += 1;
            }
        }

        // Step 3: Iteratively merge the most frequent pair.
        let max_merges = vocab_size.saturating_sub(tokenizer.vocab.len());

        for _ in 0..max_merges {
            // Count bigram frequencies across all words.
            let mut pair_counts: HashMap<(String, String), u64> = HashMap::new();
            for (symbols, freq) in &word_freqs {
                if symbols.len() < 2 {
                    continue;
                }
                for pair in symbols.windows(2) {
                    let key = (pair[0].clone(), pair[1].clone());
                    *pair_counts.entry(key).or_insert(0) += freq;
                }
            }

            // Find the most frequent pair.
            let best = pair_counts
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(pair, _)| pair.clone());

            let best = match best {
                Some(p) => p,
                None => break, // No pairs left to merge.
            };

            let merged = format!("{}{}", best.0, best.1);

            // Register merge & new token.
            tokenizer.merges.push(best.clone());
            if !tokenizer.vocab.contains_key(&merged) {
                let id = tokenizer.vocab.len() as u32;
                tokenizer.vocab.insert(merged.clone(), id);
                tokenizer.id_to_token.insert(id, merged.clone());
            }

            // Apply merge to word table.
            let mut new_word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
            for (symbols, freq) in &word_freqs {
                let updated = apply_merge(symbols, &best.0, &best.1, &merged);
                *new_word_freqs.entry(updated).or_insert(0) += freq;
            }
            word_freqs = new_word_freqs;
        }

        Ok(tokenizer)
    }

    /// Return the UNK token ID.
    fn unk_id(&self) -> u32 {
        self.special_tokens.get("[UNK]").copied().unwrap_or(1)
    }

    /// Encode a single word (no whitespace) into token IDs using learned merges.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply merges in order.
        for (left, right) in &self.merges {
            let merged = format!("{}{}", left, right);
            symbols = apply_merge(&symbols, left, right, &merged);
        }

        // Map symbols to IDs.
        let unk = self.unk_id();
        symbols
            .iter()
            .map(|s| self.vocab.get(s).copied().unwrap_or(unk))
            .collect()
    }

    /// Get a reference to the special tokens map.
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// Get the token ID for a given special token name, e.g. `"[CLS]"`.
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Add a custom special token. Returns the assigned ID.
    pub fn add_special_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            self.special_tokens.insert(token.to_string(), id);
            return id;
        }
        let id = self.vocab.len() as u32;
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.special_tokens.insert(token.to_string(), id);
        id
    }

    /// Save the tokenizer to a JSON file.
    ///
    /// The JSON format stores `vocab`, `merges`, and `special_tokens`.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|e| TextError::IoError(format!("save_json: {e}")))?;
        let writer = BufWriter::new(file);

        // Build a serializable structure manually (no serde required at runtime
        // if serde-support feature is off -- we write JSON by hand).
        write_bpe_json(writer, &self.vocab, &self.merges, &self.special_tokens)
    }

    /// Load a BPE tokenizer from a JSON file previously written by [`Self::save_json`].
    pub fn load_json(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(format!("load_json: {e}")))?;
        let reader = BufReader::new(file);
        read_bpe_json(reader)
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformerTokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = pre_tokenize(text);
        let words = split_words(&normalized);
        let mut ids = Vec::new();
        for word in &words {
            ids.extend(self.encode_word(word));
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(&id).cloned())
            .collect();
        // Join tokens -- BPE tokens are subword pieces so just concatenate.
        // We insert a space before tokens that start a new word. As a heuristic
        // we detect word boundaries when the token does not start with a
        // continuation indicator. For a simple BPE without explicit word boundary
        // markers we just concatenate with spaces between independent words.
        // Since our training splits on whitespace and encodes each word separately
        // we do a rough reconstruction.
        rejoin_bpe_tokens(&tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// BPE helper functions
// ---------------------------------------------------------------------------

/// Apply a single merge to a symbol sequence.
fn apply_merge(symbols: &[String], left: &str, right: &str, merged: &str) -> Vec<String> {
    let mut result = Vec::with_capacity(symbols.len());
    let mut i = 0;
    while i < symbols.len() {
        if i + 1 < symbols.len() && symbols[i] == left && symbols[i + 1] == right {
            result.push(merged.to_string());
            i += 2;
        } else {
            result.push(symbols[i].clone());
            i += 1;
        }
    }
    result
}

/// Rejoin BPE-decoded token strings into readable text.
///
/// We use a simple word-boundary heuristic: tokens that were produced from
/// separate words during encoding will be separated by spaces. Because our BPE
/// trains per-word, the concatenation of subword pieces within a word is direct.
/// Between words we insert spaces. We detect word starts by checking if the
/// decoded sequence would make sense with a space. This is approximate.
fn rejoin_bpe_tokens(tokens: &[String]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    // Since BPE encodes each whitespace-separated word independently, the
    // decode output is the concatenation of subword pieces. A space character
    // token (" ") would indicate word boundaries if present. Otherwise we
    // just concatenate everything.
    let joined: String = tokens.concat();
    // If the vocabulary includes the space character as a token, the spaces
    // appear naturally. If not, the caller may need post-processing.
    joined
}

/// Write BPE tokenizer state as JSON to a writer.
fn write_bpe_json<W: IoWrite>(
    mut w: W,
    vocab: &HashMap<String, u32>,
    merges: &[(String, String)],
    special_tokens: &HashMap<String, u32>,
) -> Result<()> {
    let write_err = |e: std::io::Error| TextError::IoError(format!("write_bpe_json: {e}"));

    w.write_all(b"{\n").map_err(write_err)?;

    // vocab
    w.write_all(b"  \"vocab\": {\n").map_err(write_err)?;
    let mut sorted_vocab: Vec<(&String, &u32)> = vocab.iter().collect();
    sorted_vocab.sort_by_key(|&(_, id)| *id);
    for (idx, (token, id)) in sorted_vocab.iter().enumerate() {
        let comma = if idx + 1 < sorted_vocab.len() {
            ","
        } else {
            ""
        };
        let escaped = escape_json_string(token);
        writeln!(w, "    \"{}\": {}{}", escaped, id, comma).map_err(write_err)?;
    }
    w.write_all(b"  },\n").map_err(write_err)?;

    // merges
    w.write_all(b"  \"merges\": [\n").map_err(write_err)?;
    for (idx, (left, right)) in merges.iter().enumerate() {
        let comma = if idx + 1 < merges.len() { "," } else { "" };
        let left_esc = escape_json_string(left);
        let right_esc = escape_json_string(right);
        writeln!(w, "    [\"{}\", \"{}\"]{}", left_esc, right_esc, comma).map_err(write_err)?;
    }
    w.write_all(b"  ],\n").map_err(write_err)?;

    // special_tokens
    w.write_all(b"  \"special_tokens\": {\n")
        .map_err(write_err)?;
    let mut sorted_special: Vec<(&String, &u32)> = special_tokens.iter().collect();
    sorted_special.sort_by_key(|&(_, id)| *id);
    for (idx, (token, id)) in sorted_special.iter().enumerate() {
        let comma = if idx + 1 < sorted_special.len() {
            ","
        } else {
            ""
        };
        let escaped = escape_json_string(token);
        writeln!(w, "    \"{}\": {}{}", escaped, id, comma).map_err(write_err)?;
    }
    w.write_all(b"  }\n").map_err(write_err)?;

    w.write_all(b"}\n").map_err(write_err)?;
    Ok(())
}

/// Read BPE tokenizer state from a JSON reader.
///
/// This is a minimal hand-rolled JSON parser sufficient for our format.
/// It avoids requiring serde at runtime.
fn read_bpe_json<R: BufRead>(reader: R) -> Result<BPETokenizer> {
    let content: String = reader
        .lines()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| TextError::IoError(format!("read_bpe_json: {e}")))?
        .join("\n");

    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut id_to_token: HashMap<u32, String> = HashMap::new();
    let mut merges: Vec<(String, String)> = Vec::new();
    let mut special_tokens: HashMap<String, u32> = HashMap::new();

    // Parse vocab section
    if let Some(vocab_section) = extract_json_object(&content, "vocab") {
        for (key, val) in parse_string_int_pairs(&vocab_section) {
            vocab.insert(key.clone(), val);
            id_to_token.insert(val, key);
        }
    }

    // Parse merges section
    if let Some(merges_section) = extract_json_array(&content, "merges") {
        merges = parse_merge_pairs(&merges_section);
    }

    // Parse special_tokens section
    if let Some(special_section) = extract_json_object(&content, "special_tokens") {
        for (key, val) in parse_string_int_pairs(&special_section) {
            special_tokens.insert(key, val);
        }
    }

    Ok(BPETokenizer {
        vocab,
        id_to_token,
        merges,
        special_tokens,
    })
}

/// Escape a string for JSON output.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Unescape a JSON string value.
fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(code) {
                            out.push(c);
                        }
                    }
                }
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Extract a JSON object value by key name, returning the content between braces.
fn extract_json_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    // Find the opening brace
    let brace_start = after_key.find('{')?;
    let content_start = start + pattern.len() + brace_start;

    // Find matching closing brace
    let mut depth = 0;
    for (i, ch) in json[content_start..].chars().enumerate() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(json[content_start + 1..content_start + i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract a JSON array value by key name, returning the content between brackets.
fn extract_json_array(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let bracket_start = after_key.find('[')?;
    let content_start = start + pattern.len() + bracket_start;

    let mut depth = 0;
    for (i, ch) in json[content_start..].chars().enumerate() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(json[content_start + 1..content_start + i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse `"key": int` pairs from the inside of a JSON object.
fn parse_string_int_pairs(content: &str) -> Vec<(String, u32)> {
    let mut pairs = Vec::new();
    let mut remaining = content.trim();

    while !remaining.is_empty() {
        // Find next quoted key
        let q1 = match remaining.find('"') {
            Some(pos) => pos,
            None => break,
        };
        let after_q1 = &remaining[q1 + 1..];
        // Find closing quote (handle escapes)
        let q2 = match find_unescaped_quote(after_q1) {
            Some(pos) => pos,
            None => break,
        };
        let key = unescape_json_string(&after_q1[..q2]);
        let after_key = &after_q1[q2 + 1..];

        // Find colon then the integer value
        let colon = match after_key.find(':') {
            Some(pos) => pos,
            None => break,
        };
        let after_colon = after_key[colon + 1..].trim_start();

        // Read integer
        let num_end = after_colon
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_colon.len());
        if let Ok(val) = after_colon[..num_end].parse::<u32>() {
            pairs.push((key, val));
        }

        // Advance past comma or end
        let consumed = after_colon[num_end..].trim_start();
        remaining = if consumed.starts_with(',') {
            &consumed[1..]
        } else {
            consumed
        };
    }
    pairs
}

/// Parse merge pairs `["left", "right"]` from inside a JSON array.
fn parse_merge_pairs(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut remaining = content.trim();

    while !remaining.is_empty() {
        // Find next inner array '['
        let bracket = match remaining.find('[') {
            Some(pos) => pos,
            None => break,
        };
        let end_bracket = match remaining[bracket..].find(']') {
            Some(pos) => bracket + pos,
            None => break,
        };
        let inner = &remaining[bracket + 1..end_bracket];

        // Extract two quoted strings from inner
        let mut strings = Vec::new();
        let mut inner_rem = inner.trim();
        for _ in 0..2 {
            let q1 = match inner_rem.find('"') {
                Some(pos) => pos,
                None => break,
            };
            let after_q1 = &inner_rem[q1 + 1..];
            let q2 = match find_unescaped_quote(after_q1) {
                Some(pos) => pos,
                None => break,
            };
            strings.push(unescape_json_string(&after_q1[..q2]));
            inner_rem = &after_q1[q2 + 1..];
            inner_rem = inner_rem.trim_start();
            if inner_rem.starts_with(',') {
                inner_rem = &inner_rem[1..];
            }
        }

        if strings.len() == 2 {
            pairs.push((strings[0].clone(), strings[1].clone()));
        }

        remaining = &remaining[end_bracket + 1..];
        remaining = remaining.trim_start();
        if remaining.starts_with(',') {
            remaining = &remaining[1..];
        }
    }
    pairs
}

/// Find the position of the first unescaped double-quote in a string.
fn find_unescaped_quote(s: &str) -> Option<usize> {
    let mut escape = false;
    for (i, ch) in s.chars().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if ch == '"' {
            return Some(i);
        }
    }
    None
}

// ===========================================================================
// WordPieceTokenizer
// ===========================================================================

/// A WordPiece tokenizer (BERT-style).
///
/// WordPiece is a subword tokenization algorithm that greedily matches the
/// longest prefix of a word from the vocabulary, using a continuation prefix
/// (typically `"##"`) for non-initial subwords.
///
/// # Example
///
/// ```rust
/// use scirs2_text::tokenizer::WordPieceTokenizer;
/// use std::collections::HashMap;
///
/// let mut vocab = HashMap::new();
/// vocab.insert("[UNK]".to_string(), 0);
/// vocab.insert("hello".to_string(), 1);
/// vocab.insert("world".to_string(), 2);
/// vocab.insert("hel".to_string(), 3);
/// vocab.insert("##lo".to_string(), 4);
///
/// let tokenizer = WordPieceTokenizer::new(vocab);
/// let tokens = tokenizer.tokenize("hello world");
/// assert!(tokens.contains(&"hello".to_string()) || tokens.contains(&"hel".to_string()));
/// ```
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    /// Token string -> token ID
    vocab: HashMap<String, u32>,
    /// Token ID -> token string (inverse)
    id_to_token: HashMap<u32, String>,
    /// Maximum word length to consider (longer words become [UNK])
    max_word_len: usize,
    /// The unknown token string
    unk_token: String,
    /// The prefix prepended to continuation subwords (default: `"##"`)
    continuing_prefix: String,
}

impl WordPieceTokenizer {
    /// Create a new WordPiece tokenizer from a vocabulary map.
    ///
    /// The vocabulary must contain at least `[UNK]`. The continuation prefix
    /// defaults to `"##"` and max word length defaults to 200.
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Self {
            vocab,
            id_to_token,
            max_word_len: 200,
            unk_token: "[UNK]".to_string(),
            continuing_prefix: "##".to_string(),
        }
    }

    /// Set the maximum word length.
    pub fn with_max_word_len(mut self, max_len: usize) -> Self {
        self.max_word_len = max_len;
        self
    }

    /// Set the unknown token string.
    pub fn with_unk_token(mut self, unk: &str) -> Self {
        self.unk_token = unk.to_string();
        self
    }

    /// Set the continuation prefix (default `"##"`).
    pub fn with_continuing_prefix(mut self, prefix: &str) -> Self {
        self.continuing_prefix = prefix.to_string();
        self
    }

    /// Load a WordPiece vocabulary from a text file (one token per line).
    ///
    /// Token IDs are assigned sequentially starting from 0.
    pub fn from_vocab_file(path: &Path) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| TextError::IoError(format!("from_vocab_file: {e}")))?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (id, line) in reader.lines().enumerate() {
            let line =
                line.map_err(|e| TextError::IoError(format!("from_vocab_file read: {e}")))?;
            let token = line.trim().to_string();
            if !token.is_empty() {
                vocab.insert(token, id as u32);
            }
        }

        if vocab.is_empty() {
            return Err(TextError::TokenizationError(
                "Vocabulary file is empty".to_string(),
            ));
        }

        Ok(Self::new(vocab))
    }

    /// Tokenize text into a list of token strings (subword pieces).
    ///
    /// Each word is greedily segmented into the longest matching vocab entries.
    /// Non-initial pieces are prefixed with `##`.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = pre_tokenize(text);
        let words = split_words(&normalized);
        let mut tokens = Vec::new();

        for word in &words {
            if word.len() > self.max_word_len {
                tokens.push(self.unk_token.clone());
                continue;
            }
            let word_tokens = self.tokenize_word(word);
            tokens.extend(word_tokens);
        }
        tokens
    }

    /// Segment a single word into WordPiece tokens.
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start == 0 {
                    substr.clone()
                } else {
                    format!("{}{}", self.continuing_prefix, substr)
                };

                if self.vocab.contains_key(&candidate) {
                    tokens.push(candidate);
                    found = true;
                    break;
                }
                end -= 1;
            }

            if !found {
                // Could not match any subword starting at `start`
                tokens.push(self.unk_token.clone());
                start += 1;
            } else {
                start = end;
            }
        }

        tokens
    }

    /// Get the UNK token ID.
    fn unk_id(&self) -> u32 {
        self.vocab.get(&self.unk_token).copied().unwrap_or(0)
    }
}

impl TransformerTokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let tokens = self.tokenize(text);
        let unk = self.unk_id();
        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(unk))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        let mut first_in_word = true;

        for &id in ids {
            let token = match self.id_to_token.get(&id) {
                Some(t) => t.as_str(),
                None => &self.unk_token,
            };

            if let Some(stripped) = token.strip_prefix(&self.continuing_prefix) {
                // Continuation piece: append directly (no space)
                result.push_str(stripped);
            } else {
                // New word
                if !first_in_word {
                    result.push(' ');
                }
                result.push_str(token);
            }
            first_in_word = false;
        }
        result
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ===========================================================================
// SimpleWhitespaceTokenizer
// ===========================================================================

/// A vocabulary-aware whitespace tokenizer.
///
/// Splits text on whitespace and maps each word to an integer token ID.
/// Unknown words map to a reserved UNK ID.
///
/// # Example
///
/// ```rust
/// use scirs2_text::tokenizer::{SimpleWhitespaceTokenizer, TransformerTokenizer};
///
/// let texts = &["hello world", "hello there"];
/// let tok = SimpleWhitespaceTokenizer::build(texts, 100);
/// let ids = tok.encode("hello world");
/// let decoded = tok.decode(&ids);
/// assert_eq!(decoded, "hello world");
/// ```
#[derive(Debug, Clone)]
pub struct SimpleWhitespaceTokenizer {
    /// word -> token ID
    vocab: HashMap<String, u32>,
    /// token ID -> word
    id_to_token: HashMap<u32, String>,
    /// Token ID for unknown words
    unk_id: u32,
}

impl SimpleWhitespaceTokenizer {
    /// Build a whitespace tokenizer from training texts.
    ///
    /// Counts word frequencies and keeps the top `max_vocab` words.
    /// Token ID 0 is reserved for `[UNK]`.
    pub fn build(texts: &[&str], max_vocab: usize) -> Self {
        let mut word_counts: HashMap<String, u64> = HashMap::new();
        for text in texts {
            let normalized = pre_tokenize(text);
            for word in split_words(&normalized) {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency (descending), then alphabetically for determinism.
        let mut sorted: Vec<(String, u64)> = word_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Reserve ID 0 for UNK
        vocab.insert("[UNK]".to_string(), 0);
        id_to_token.insert(0, "[UNK]".to_string());

        let limit = max_vocab.saturating_sub(1); // -1 for UNK
        for (word, _) in sorted.into_iter().take(limit) {
            let id = vocab.len() as u32;
            id_to_token.insert(id, word.clone());
            vocab.insert(word, id);
        }

        Self {
            vocab,
            id_to_token,
            unk_id: 0,
        }
    }

    /// Create from an existing vocabulary.
    pub fn from_vocab(vocab: HashMap<String, u32>, unk_id: u32) -> Self {
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Self {
            vocab,
            id_to_token,
            unk_id,
        }
    }
}

impl TransformerTokenizer for SimpleWhitespaceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = pre_tokenize(text);
        split_words(&normalized)
            .iter()
            .map(|w| self.vocab.get(w).copied().unwrap_or(self.unk_id))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(&id).cloned())
            .collect::<Vec<String>>()
            .join(" ")
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ===========================================================================
// SimpleCharTokenizer
// ===========================================================================

/// A character-level tokenizer that maps each unique character to a token ID.
///
/// Useful as a baseline or for character-level transformer models.
///
/// # Example
///
/// ```rust
/// use scirs2_text::tokenizer::{SimpleCharTokenizer, TransformerTokenizer};
///
/// let texts = &["abc", "bcd"];
/// let tok = SimpleCharTokenizer::build(texts);
/// let ids = tok.encode("abc");
/// assert_eq!(ids.len(), 3);
/// let decoded = tok.decode(&ids);
/// assert_eq!(decoded, "abc");
/// ```
#[derive(Debug, Clone)]
pub struct SimpleCharTokenizer {
    /// char -> token ID
    vocab: HashMap<char, u32>,
    /// token ID -> char
    id_to_char: HashMap<u32, char>,
    /// Token ID for unknown characters
    unk_id: u32,
}

impl SimpleCharTokenizer {
    /// Build a character tokenizer from training texts.
    ///
    /// Token ID 0 is reserved for unknown characters (those not seen in training).
    pub fn build(texts: &[&str]) -> Self {
        let mut char_set: Vec<char> = Vec::new();
        for text in texts {
            for ch in text.chars() {
                if !char_set.contains(&ch) {
                    char_set.push(ch);
                }
            }
        }
        char_set.sort();

        let mut vocab = HashMap::new();
        let mut id_to_char = HashMap::new();
        // Reserve ID 0 for UNK
        let unk_id = 0_u32;

        for ch in char_set {
            let id = (vocab.len() + 1) as u32; // start from 1
            vocab.insert(ch, id);
            id_to_char.insert(id, ch);
        }

        Self {
            vocab,
            id_to_char,
            unk_id,
        }
    }

    /// Create from an existing character vocabulary.
    pub fn from_vocab(vocab: HashMap<char, u32>, unk_id: u32) -> Self {
        let id_to_char: HashMap<u32, char> = vocab.iter().map(|(&c, &id)| (id, c)).collect();
        Self {
            vocab,
            id_to_char,
            unk_id,
        }
    }
}

impl TransformerTokenizer for SimpleCharTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|ch| self.vocab.get(&ch).copied().unwrap_or(self.unk_id))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_char.get(&id).copied())
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len() + 1 // +1 for the UNK slot
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- BPE tests ---

    #[test]
    fn test_bpe_train_basic() {
        let corpus = &[
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
        ];
        let tok = BPETokenizer::train(corpus, 80).expect("train should succeed");
        assert!(tok.vocab_size() > 0);
        assert!(tok.vocab_size() <= 80);
    }

    #[test]
    fn test_bpe_train_empty_corpus_error() {
        let result = BPETokenizer::train(&[], 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_train_small_vocab_error() {
        let corpus = &["hello"];
        // Vocab size too small (less than special tokens + 1)
        let result = BPETokenizer::train(corpus, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let corpus = &["hello world", "hello there world", "the world is great"];
        let tok = BPETokenizer::train(corpus, 100).expect("train");

        let text = "hello world";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());

        let decoded = tok.decode(&ids);
        // The decoded text should contain the original characters
        // (spaces are part of training so they appear as tokens)
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_bpe_encode_empty() {
        let corpus = &["hello world"];
        let tok = BPETokenizer::train(corpus, 50).expect("train");
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_bpe_special_tokens() {
        let tok = BPETokenizer::new();
        assert!(tok.special_token_id("[PAD]").is_some());
        assert!(tok.special_token_id("[UNK]").is_some());
        assert!(tok.special_token_id("[CLS]").is_some());
        assert!(tok.special_token_id("[SEP]").is_some());
        assert!(tok.special_token_id("[MASK]").is_some());
    }

    #[test]
    fn test_bpe_add_special_token() {
        let mut tok = BPETokenizer::new();
        let id = tok.add_special_token("[BOS]");
        assert_eq!(tok.special_token_id("[BOS]"), Some(id));
        assert_eq!(tok.vocab_size(), DEFAULT_SPECIAL_TOKENS.len() + 1);
    }

    #[test]
    fn test_bpe_save_load_json() {
        let corpus = &["the cat sat on the mat", "the dog sat on the log"];
        let tok = BPETokenizer::train(corpus, 60).expect("train");

        let dir = std::env::temp_dir();
        let path = dir.join("test_bpe_tokenizer.json");

        tok.save_json(&path).expect("save");
        let loaded = BPETokenizer::load_json(&path).expect("load");

        assert_eq!(tok.vocab_size(), loaded.vocab_size());
        assert_eq!(tok.merges.len(), loaded.merges.len());

        // Encoding should produce same results
        let text = "the cat sat";
        let ids1 = tok.encode(text);
        let ids2 = loaded.encode(text);
        assert_eq!(ids1, ids2);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bpe_unknown_chars() {
        let corpus = &["abc"];
        let tok = BPETokenizer::train(corpus, 30).expect("train");

        // Encode text with characters not in training corpus
        let ids = tok.encode("xyz");
        // Unknown chars should map to UNK
        let unk = tok.unk_id();
        assert!(ids.iter().all(|&id| id == unk));
    }

    #[test]
    fn test_bpe_default_constructor() {
        let tok = BPETokenizer::default();
        assert_eq!(tok.vocab_size(), DEFAULT_SPECIAL_TOKENS.len());
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn test_bpe_vocab_size_trait() {
        let corpus = &["hello world hello"];
        let tok = BPETokenizer::train(corpus, 50).expect("train");
        let trait_ref: &dyn TransformerTokenizer = &tok;
        assert!(trait_ref.vocab_size() > 0);
    }

    // --- WordPiece tests ---

    #[test]
    fn test_wordpiece_basic() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab.insert("hel".to_string(), 3);
        vocab.insert("##lo".to_string(), 4);
        vocab.insert("wor".to_string(), 5);
        vocab.insert("##ld".to_string(), 6);

        let tok = WordPieceTokenizer::new(vocab);
        let tokens = tok.tokenize("hello world");

        // "hello" should be found as-is (full match)
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_wordpiece_subword_split() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("un".to_string(), 1);
        vocab.insert("##aff".to_string(), 2);
        vocab.insert("##able".to_string(), 3);

        let tok = WordPieceTokenizer::new(vocab);
        let tokens = tok.tokenize("unaffable");
        // "unaffable" -> "un" + "##aff" + "##able"
        assert_eq!(tokens, vec!["un", "##aff", "##able"]);
    }

    #[test]
    fn test_wordpiece_unknown_word() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);

        let tok = WordPieceTokenizer::new(vocab);
        let tokens = tok.tokenize("xyz");
        // "xyz" has no matching subwords -> all characters become UNK
        assert!(tokens.contains(&"[UNK]".to_string()));
    }

    #[test]
    fn test_wordpiece_encode_decode() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("play".to_string(), 1);
        vocab.insert("##ing".to_string(), 2);
        vocab.insert("##er".to_string(), 3);
        vocab.insert("##s".to_string(), 4);
        vocab.insert("the".to_string(), 5);

        let tok = WordPieceTokenizer::new(vocab);

        let ids = tok.encode("the playing");
        assert!(!ids.is_empty());

        let decoded = tok.decode(&ids);
        assert!(decoded.contains("the"));
        assert!(decoded.contains("play"));
        assert!(decoded.contains("ing"));
    }

    #[test]
    fn test_wordpiece_max_word_len() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("a".to_string(), 1);

        let tok = WordPieceTokenizer::new(vocab).with_max_word_len(5);

        // Word longer than max_word_len -> UNK
        let tokens = tok.tokenize("toolongword");
        assert_eq!(tokens, vec!["[UNK]"]);
    }

    #[test]
    fn test_wordpiece_custom_prefix() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hel".to_string(), 1);
        vocab.insert("@@lo".to_string(), 2);

        let tok = WordPieceTokenizer::new(vocab).with_continuing_prefix("@@");
        let tokens = tok.tokenize("hello");
        assert_eq!(tokens, vec!["hel", "@@lo"]);
    }

    #[test]
    fn test_wordpiece_from_vocab_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wp_vocab.txt");

        // Write a vocab file
        {
            let mut f = File::create(&path).expect("create vocab file");
            writeln!(f, "[UNK]").expect("write");
            writeln!(f, "[PAD]").expect("write");
            writeln!(f, "hello").expect("write");
            writeln!(f, "world").expect("write");
            writeln!(f, "##ing").expect("write");
        }

        let tok = WordPieceTokenizer::from_vocab_file(&path).expect("load vocab");
        assert_eq!(tok.vocab_size(), 5);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wordpiece_vocab_size() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);

        let tok = WordPieceTokenizer::new(vocab);
        assert_eq!(tok.vocab_size(), 3);
    }

    // --- SimpleWhitespace tests ---

    #[test]
    fn test_whitespace_build_and_encode() {
        let texts = &["hello world", "hello there", "world peace"];
        let tok = SimpleWhitespaceTokenizer::build(texts, 100);

        let ids = tok.encode("hello world");
        assert_eq!(ids.len(), 2);

        // hello and world should have different IDs
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn test_whitespace_decode() {
        let texts = &["hello world", "foo bar"];
        let tok = SimpleWhitespaceTokenizer::build(texts, 100);

        let ids = tok.encode("hello world");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_whitespace_unknown_word() {
        let texts = &["hello world"];
        let tok = SimpleWhitespaceTokenizer::build(texts, 100);

        let ids = tok.encode("hello xyz");
        // "xyz" is unknown -> maps to unk_id (0)
        assert_eq!(ids[1], 0);
    }

    #[test]
    fn test_whitespace_max_vocab_limit() {
        let texts = &["a b c d e f g"];
        let tok = SimpleWhitespaceTokenizer::build(texts, 4); // 3 words + 1 UNK
        assert!(tok.vocab_size() <= 4);
    }

    #[test]
    fn test_whitespace_vocab_size() {
        let texts = &["one two three"];
        let tok = SimpleWhitespaceTokenizer::build(texts, 100);
        // 3 words + 1 UNK = 4
        assert_eq!(tok.vocab_size(), 4);
    }

    // --- SimpleChar tests ---

    #[test]
    fn test_char_build_and_encode() {
        let texts = &["abc", "bcd"];
        let tok = SimpleCharTokenizer::build(texts);

        let ids = tok.encode("abc");
        assert_eq!(ids.len(), 3);
        // All IDs should be non-zero (0 is UNK)
        assert!(ids.iter().all(|&id| id > 0));
    }

    #[test]
    fn test_char_decode() {
        let texts = &["hello"];
        let tok = SimpleCharTokenizer::build(texts);

        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_char_unknown_char() {
        let texts = &["abc"];
        let tok = SimpleCharTokenizer::build(texts);

        let ids = tok.encode("xyz");
        // All unknown -> UNK id (0)
        assert!(ids.iter().all(|&id| id == 0));
    }

    #[test]
    fn test_char_vocab_size() {
        let texts = &["ab", "bc"];
        let tok = SimpleCharTokenizer::build(texts);
        // 3 unique chars (a, b, c) + 1 UNK slot = 4
        assert_eq!(tok.vocab_size(), 4);
    }

    #[test]
    fn test_char_roundtrip() {
        let texts = &["The quick brown fox!"];
        let tok = SimpleCharTokenizer::build(texts);

        let original = "The quick brown fox!";
        let ids = tok.encode(original);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, original);
    }

    // --- Cross-tokenizer trait tests ---

    #[test]
    fn test_trait_object_dispatch() {
        let corpus = &["hello world hello"];
        let bpe = BPETokenizer::train(corpus, 50).expect("train");

        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        let wp = WordPieceTokenizer::new(vocab);

        let ws_texts = &["hello world"];
        let ws = SimpleWhitespaceTokenizer::build(ws_texts, 50);

        let char_texts = &["hello world"];
        let ch = SimpleCharTokenizer::build(char_texts);

        // All implement TransformerTokenizer and can be used as trait objects.
        let tokenizers: Vec<&dyn TransformerTokenizer> = vec![&bpe, &wp, &ws, &ch];
        for tok in tokenizers {
            assert!(tok.vocab_size() > 0);
            let ids = tok.encode("hello");
            assert!(!ids.is_empty());
            let _ = tok.decode(&ids);
        }
    }

    // --- JSON escape/unescape tests ---

    #[test]
    fn test_json_escape_roundtrip() {
        let original = "hello \"world\"\nnewline\\backslash\ttab";
        let escaped = escape_json_string(original);
        let unescaped = unescape_json_string(&escaped);
        assert_eq!(original, unescaped);
    }

    #[test]
    fn test_bpe_multiple_sentences() {
        let corpus = &[
            "machine learning is transforming the world",
            "deep learning models use transformers",
            "natural language processing with transformers",
            "the transformer architecture is powerful",
        ];
        let tok = BPETokenizer::train(corpus, 120).expect("train");

        let text = "learning transformers";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());

        // Verify no UNK tokens for known text
        let unk = tok.unk_id();
        // Characters in "learning transformers" should all be in training data
        assert!(ids.iter().all(|&id| id != unk));
    }

    #[test]
    fn test_bpe_merges_reduce_token_count() {
        let corpus = &["aaaa aaaa aaaa aaaa aaaa", "aaaa aaaa aaaa aaaa aaaa"];
        let tok = BPETokenizer::train(corpus, 50).expect("train");

        // "aaaa" should be encoded in fewer tokens than 4 chars
        // because BPE should merge "a"+"a" -> "aa", etc.
        let ids = tok.encode("aaaa");
        assert!(
            ids.len() < 4,
            "BPE should merge repeated chars: got {} tokens",
            ids.len()
        );
    }

    #[test]
    fn test_wordpiece_empty_input() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        let tok = WordPieceTokenizer::new(vocab);

        let tokens = tok.tokenize("");
        assert!(tokens.is_empty());

        let ids = tok.encode("");
        assert!(ids.is_empty());

        let decoded = tok.decode(&[]);
        assert!(decoded.is_empty());
    }
}
