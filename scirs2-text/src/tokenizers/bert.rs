//! BERT-style WordPiece tokenizer with special tokens.
//!
//! This module provides a full BERT/RoBERTa-compatible tokenizer implementing:
//! - Basic tokenization (whitespace + punctuation splitting, optional lowercasing)
//! - WordPiece subword segmentation (greedy longest-match with `##` continuations)
//! - Special token management: `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`
//! - Single- and pair-sentence encoding with token type IDs
//! - Batched encoding with padding and truncation

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use unicode_normalization::UnicodeNormalization;

// ─── Constants ────────────────────────────────────────────────────────────────

const CLS_TOKEN: &str = "[CLS]";
const SEP_TOKEN: &str = "[SEP]";
const PAD_TOKEN: &str = "[PAD]";
const MASK_TOKEN: &str = "[MASK]";
const UNK_TOKEN: &str = "[UNK]";

/// Maximum characters in a single word before falling back to `[UNK]`.
const MAX_WORD_CHARS: usize = 200;

// ─── BertEncoding ─────────────────────────────────────────────────────────────

/// A single encoded BERT input sequence.
///
/// Contains the input IDs, an attention mask (1 = real token, 0 = padding),
/// and token type IDs (0 = first segment, 1 = second segment).
#[derive(Debug, Clone, PartialEq)]
pub struct BertEncoding {
    /// Token IDs including special tokens.
    pub input_ids: Vec<u32>,
    /// Attention mask: `1` for real tokens, `0` for padding.
    pub attention_mask: Vec<u32>,
    /// Segment indicator: `0` for text_a, `1` for text_b.
    pub token_type_ids: Vec<u32>,
}

impl BertEncoding {
    /// Create a new encoding with consistent lengths.
    pub fn new(input_ids: Vec<u32>, attention_mask: Vec<u32>, token_type_ids: Vec<u32>) -> Self {
        BertEncoding {
            input_ids,
            attention_mask,
            token_type_ids,
        }
    }

    /// Returns the sequence length (number of tokens including padding).
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Returns `true` if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

// ─── BatchEncoding ────────────────────────────────────────────────────────────

/// A batch of [`BertEncoding`] instances with consistent (padded) lengths.
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    /// Individual encodings, all of the same length.
    pub encodings: Vec<BertEncoding>,
}

impl BatchEncoding {
    /// Create from a vector of encodings.
    pub fn new(encodings: Vec<BertEncoding>) -> Self {
        BatchEncoding { encodings }
    }

    /// Number of sequences in the batch.
    pub fn len(&self) -> usize {
        self.encodings.len()
    }

    /// Returns `true` if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.encodings.is_empty()
    }

    /// Collect all `input_ids` into a 2-D vector `[batch, seq_len]`.
    pub fn input_ids(&self) -> Vec<Vec<u32>> {
        self.encodings.iter().map(|e| e.input_ids.clone()).collect()
    }

    /// Collect all `attention_mask` into a 2-D vector.
    pub fn attention_masks(&self) -> Vec<Vec<u32>> {
        self.encodings
            .iter()
            .map(|e| e.attention_mask.clone())
            .collect()
    }

    /// Collect all `token_type_ids` into a 2-D vector.
    pub fn token_type_ids(&self) -> Vec<Vec<u32>> {
        self.encodings
            .iter()
            .map(|e| e.token_type_ids.clone())
            .collect()
    }
}

// ─── BasicTokenizer ───────────────────────────────────────────────────────────

/// Whitespace + punctuation tokenizer (BERT pre-tokenization step).
///
/// Optionally lowercases and strips combining Unicode marks (accents).
#[derive(Debug, Clone)]
struct BasicTokenizer {
    do_lower_case: bool,
}

impl BasicTokenizer {
    fn new(do_lower_case: bool) -> Self {
        BasicTokenizer { do_lower_case }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.do_lower_case {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Strip combining characters by NFD decomposition.
        let text: String = text.nfd().filter(|c| !is_combining_mark(*c)).collect();

        // Insert spaces around punctuation and CJK characters.
        let mut spaced = String::with_capacity(text.len() + 32);
        for ch in text.chars() {
            if ch.is_whitespace() {
                spaced.push(' ');
            } else if is_punctuation_char(ch) || is_chinese_char(ch) {
                spaced.push(' ');
                spaced.push(ch);
                spaced.push(' ');
            } else {
                spaced.push(ch);
            }
        }

        spaced
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

/// Returns `true` for Unicode combining marks (NFD non-spacing marks).
fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    (0x0300..=0x036F).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
}

/// Returns `true` for ASCII punctuation and common Unicode punctuation.
fn is_punctuation_char(ch: char) -> bool {
    let cp = ch as u32;
    // ASCII control/punctuation ranges
    if cp <= 47 || (58..=64).contains(&cp) || (91..=96).contains(&cp) || (123..=126).contains(&cp) {
        return true;
    }
    // Unicode punctuation categories (approximate)
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '。' | '，'
                | '、'
                | '；'
                | '：'
                | '？'
                | '！'
                | '—'
                | '…'
                | '\u{2018}'
                | '\u{2019}'
                | '\u{201C}'
                | '\u{201D}'
        )
}

/// Returns `true` for CJK Unified Ideographs.
fn is_chinese_char(ch: char) -> bool {
    let cp = ch as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

// ─── WordPiece helper ─────────────────────────────────────────────────────────

/// Greedy longest-match WordPiece segmentation for a single `word`.
///
/// Returns a list of subword strings; continuation pieces are prefixed with
/// `"##"`.  If the word cannot be fully segmented, returns `["[UNK]"]`.
fn wordpiece_segment(word: &str, vocab: &HashMap<String, u32>) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() > MAX_WORD_CHARS {
        return vec![UNK_TOKEN.to_string()];
    }

    let n = chars.len();
    let mut sub_tokens: Vec<String> = Vec::new();
    let mut start = 0usize;

    while start < n {
        let mut end = n;
        let mut found_tok: Option<String> = None;

        while start < end {
            let substr: String = chars[start..end].iter().collect();
            let candidate = if start == 0 {
                substr.clone()
            } else {
                format!("##{}", substr)
            };

            if vocab.contains_key(&candidate) {
                found_tok = Some(candidate);
                break;
            }

            if end == start + 1 {
                // Single character not in vocab: whole word is unknown.
                return vec![UNK_TOKEN.to_string()];
            }
            end -= 1;
        }

        match found_tok {
            Some(tok) => {
                sub_tokens.push(tok);
                start = end;
            }
            None => {
                return vec![UNK_TOKEN.to_string()];
            }
        }
    }

    sub_tokens
}

// ─── BertTokenizer ────────────────────────────────────────────────────────────

/// BERT-style tokenizer combining basic tokenization and WordPiece subword
/// segmentation.
///
/// Special tokens:
/// - `[CLS]` (classification): prepended to every encoded sequence
/// - `[SEP]` (separator): appended after each segment
/// - `[MASK]` (masking): placeholder for masked-language-model pre-training
/// - `[PAD]` (padding): used to fill sequences to a target length
/// - `[UNK]` (unknown): substituted for tokens not present in the vocabulary
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use scirs2_text::tokenizers::bert::BertTokenizer;
///
/// let mut vocab: HashMap<String, u32> = HashMap::new();
/// for (i, tok) in ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]",
///                   "hello","world","##ing","play","##ed"].iter().enumerate() {
///     vocab.insert(tok.to_string(), i as u32);
/// }
/// let tokenizer = BertTokenizer::new(vocab, true);
/// let ids = tokenizer.encode("Hello World").unwrap();
/// assert_eq!(ids[0], tokenizer.cls_token_id());
/// ```
#[derive(Debug, Clone)]
pub struct BertTokenizer {
    vocab: HashMap<String, u32>,
    ids_to_tokens: HashMap<u32, String>,
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
    mask_token_id: u32,
    unk_token_id: u32,
    max_len: usize,
    lowercase: bool,
    basic: BasicTokenizer,
}

impl BertTokenizer {
    // ── Construction ──────────────────────────────────────────────────────

    /// Build a `BertTokenizer` from a `token → id` vocabulary map.
    ///
    /// All five special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`)
    /// are inserted into the vocabulary if absent.
    pub fn new(mut vocab: HashMap<String, u32>, lowercase: bool) -> Self {
        // Ensure all required special tokens exist.
        let specials = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN];
        for tok in &specials {
            if !vocab.contains_key(*tok) {
                let next_id = vocab.len() as u32;
                vocab.insert(tok.to_string(), next_id);
            }
        }

        let cls_token_id = vocab[CLS_TOKEN];
        let sep_token_id = vocab[SEP_TOKEN];
        let pad_token_id = vocab[PAD_TOKEN];
        let mask_token_id = vocab[MASK_TOKEN];
        let unk_token_id = vocab[UNK_TOKEN];

        let ids_to_tokens: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        BertTokenizer {
            vocab,
            ids_to_tokens,
            cls_token_id,
            sep_token_id,
            pad_token_id,
            mask_token_id,
            unk_token_id,
            max_len: 512,
            lowercase,
            basic: BasicTokenizer::new(lowercase),
        }
    }

    /// Load a tokenizer from a `vocab.txt` file (one token per line; line
    /// index = token ID, 0-based).
    ///
    /// Returns an error if the file cannot be read or if the resulting
    /// vocabulary is missing required special tokens after auto-insertion.
    pub fn from_vocab_file(path: &str) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (idx, line) in reader.lines().enumerate() {
            let token = line.map_err(|e| TextError::IoError(e.to_string()))?;
            let token = token.trim().to_string();
            if !token.is_empty() {
                vocab.insert(token, idx as u32);
            }
        }

        if vocab.is_empty() {
            return Err(TextError::VocabularyError(
                "Vocabulary file is empty".to_string(),
            ));
        }

        Ok(Self::new(vocab, true))
    }

    /// Override the maximum sequence length (default 512).
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Returns the `[CLS]` token ID.
    pub fn cls_token_id(&self) -> u32 {
        self.cls_token_id
    }

    /// Returns the `[SEP]` token ID.
    pub fn sep_token_id(&self) -> u32 {
        self.sep_token_id
    }

    /// Returns the `[PAD]` token ID.
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Returns the `[MASK]` token ID.
    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    /// Returns the `[UNK]` token ID.
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Return a reference to the full `token → id` vocabulary map.
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Return whether this tokenizer lowercases input text.
    pub fn lowercase(&self) -> bool {
        self.lowercase
    }

    // ── Core tokenization ─────────────────────────────────────────────────

    /// Tokenize `text` into a list of subword strings.
    ///
    /// Applies basic tokenization (whitespace + punctuation split, optional
    /// lowercasing) followed by WordPiece subword segmentation.  Unknown
    /// characters/words map to `"[UNK]"`.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let words = self.basic.tokenize(text);
        words
            .iter()
            .flat_map(|w| wordpiece_segment(w, &self.vocab))
            .collect()
    }

    /// Convert a token string to its vocabulary ID.
    fn token_to_id(&self, token: &str) -> u32 {
        self.vocab.get(token).copied().unwrap_or(self.unk_token_id)
    }

    // ── Encoding API ──────────────────────────────────────────────────────

    /// Encode a single text segment as `[CLS] tokens [SEP]`.
    ///
    /// Returns the flat sequence of token IDs.  Use `encode_pair` for
    /// two-segment inputs (e.g. question + context).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let sub_tokens = self.tokenize(text);
        let mut ids = Vec::with_capacity(sub_tokens.len() + 2);
        ids.push(self.cls_token_id);
        ids.extend(sub_tokens.iter().map(|t| self.token_to_id(t)));
        ids.push(self.sep_token_id);
        Ok(ids)
    }

    /// Encode a pair of text segments (e.g. sentence A and sentence B).
    ///
    /// Layout: `[CLS] A-tokens [SEP] B-tokens [SEP]`
    ///
    /// Returns `(token_ids, token_type_ids)` where `token_type_ids[i]` is `0`
    /// for the first segment and `1` for the second.
    pub fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        let tokens_a = self.tokenize(text_a);
        let tokens_b = self.tokenize(text_b);

        let total = 1 + tokens_a.len() + 1 + tokens_b.len() + 1; // [CLS]+A+[SEP]+B+[SEP]
        let mut ids = Vec::with_capacity(total);
        let mut type_ids = Vec::with_capacity(total);

        // Segment A: [CLS] ... [SEP]  → type 0
        ids.push(self.cls_token_id);
        type_ids.push(0u32);

        for tok in &tokens_a {
            ids.push(self.token_to_id(tok));
            type_ids.push(0);
        }

        ids.push(self.sep_token_id);
        type_ids.push(0);

        // Segment B: ... [SEP]  → type 1
        for tok in &tokens_b {
            ids.push(self.token_to_id(tok));
            type_ids.push(1);
        }

        ids.push(self.sep_token_id);
        type_ids.push(1);

        Ok((ids, type_ids))
    }

    /// Build a single [`BertEncoding`] for `text`, with optional padding and
    /// truncation to `max_length`.
    ///
    /// If `padding` is `true`, short sequences are padded with `[PAD]` to
    /// reach `max_length`.  If `truncation` is `true`, long sequences are
    /// trimmed (preserving `[CLS]` and `[SEP]`).
    pub fn encode_single(
        &self,
        text: &str,
        max_length: usize,
        padding: bool,
        truncation: bool,
    ) -> Result<BertEncoding> {
        if max_length == 0 {
            return Err(TextError::InvalidInput(
                "max_length must be greater than 0".to_string(),
            ));
        }

        let sub_tokens = self.tokenize(text);
        // Budget for content tokens: max_length - [CLS] - [SEP]
        let budget = max_length.saturating_sub(2);

        let content: Vec<u32> = if truncation && sub_tokens.len() > budget {
            sub_tokens[..budget]
                .iter()
                .map(|t| self.token_to_id(t))
                .collect()
        } else {
            sub_tokens.iter().map(|t| self.token_to_id(t)).collect()
        };

        let mut ids = Vec::with_capacity(max_length);
        ids.push(self.cls_token_id);
        ids.extend_from_slice(&content);
        ids.push(self.sep_token_id);

        let real_len = ids.len();

        if padding && ids.len() < max_length {
            let pad_count = max_length - ids.len();
            ids.extend(std::iter::repeat_n(self.pad_token_id, pad_count));
        }

        let seq_len = ids.len();
        let mut mask = vec![0u32; seq_len];
        for m in mask.iter_mut().take(real_len) {
            *m = 1;
        }
        let type_ids = vec![0u32; seq_len];

        Ok(BertEncoding::new(ids, mask, type_ids))
    }

    /// Encode a batch of texts with consistent sequence length.
    ///
    /// When `padding` is `true`, all sequences in the batch are padded to the
    /// longest (or to `max_length`, whichever is smaller).  When `truncation`
    /// is `true`, sequences exceeding `max_length` are truncated.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        max_length: usize,
        padding: bool,
        truncation: bool,
    ) -> Result<BatchEncoding> {
        if max_length == 0 {
            return Err(TextError::InvalidInput(
                "max_length must be greater than 0".to_string(),
            ));
        }

        // First pass: build raw ids for each text (before padding).
        let mut raw_encodings: Vec<(Vec<u32>, usize)> = Vec::with_capacity(texts.len());

        for text in texts {
            let sub_tokens = self.tokenize(text);
            let budget = max_length.saturating_sub(2);
            let content: Vec<u32> = if truncation && sub_tokens.len() > budget {
                sub_tokens[..budget]
                    .iter()
                    .map(|t| self.token_to_id(t))
                    .collect()
            } else {
                sub_tokens.iter().map(|t| self.token_to_id(t)).collect()
            };

            let mut ids = Vec::with_capacity(content.len() + 2);
            ids.push(self.cls_token_id);
            ids.extend_from_slice(&content);
            ids.push(self.sep_token_id);
            let real_len = ids.len();
            raw_encodings.push((ids, real_len));
        }

        // Determine target length for padding.
        let target_len = if padding {
            let max_real = raw_encodings
                .iter()
                .map(|(ids, _)| ids.len())
                .max()
                .unwrap_or(0);
            max_real.min(max_length)
        } else {
            max_length
        };

        // Second pass: pad and build BertEncoding.
        let encodings = raw_encodings
            .into_iter()
            .map(|(mut ids, real_len)| {
                if padding && ids.len() < target_len {
                    let pad_count = target_len - ids.len();
                    ids.extend(std::iter::repeat_n(self.pad_token_id, pad_count));
                }

                let seq_len = ids.len();
                let mut mask = vec![0u32; seq_len];
                for m in mask.iter_mut().take(real_len) {
                    *m = 1;
                }
                let type_ids = vec![0u32; seq_len];
                BertEncoding::new(ids, mask, type_ids)
            })
            .collect();

        Ok(BatchEncoding::new(encodings))
    }

    // ── Decoding ──────────────────────────────────────────────────────────

    /// Decode a sequence of token IDs back to a human-readable string.
    ///
    /// Special tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`) are skipped.
    /// WordPiece continuation tokens (prefixed with `##`) are merged directly
    /// onto the preceding piece without a space.
    pub fn decode(&self, ids: &[u32]) -> String {
        let special_ids: [u32; 4] = [
            self.cls_token_id,
            self.sep_token_id,
            self.pad_token_id,
            self.mask_token_id,
        ];

        let mut out = String::new();
        for &id in ids {
            if special_ids.contains(&id) {
                continue;
            }

            let tok = match self.ids_to_tokens.get(&id) {
                Some(t) => t.as_str(),
                None => UNK_TOKEN,
            };

            if tok == UNK_TOKEN {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(tok);
                continue;
            }

            if let Some(cont) = tok.strip_prefix("##") {
                // Continuation: append directly (no space).
                out.push_str(cont);
            } else {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(tok);
            }
        }
        out
    }

    /// Convert token string to its ID (exposed for testing / downstream use).
    pub fn convert_token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Convert token ID to its string representation.
    pub fn convert_id_to_token(&self, id: u32) -> Option<&str> {
        self.ids_to_tokens.get(&id).map(|s| s.as_str())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── Vocabulary helpers ──────────────────────────────────────────────

    /// Minimal vocabulary with special tokens and a handful of content tokens.
    fn base_vocab() -> HashMap<String, u32> {
        let tokens = [
            "[PAD]",   // 0
            "[UNK]",   // 1
            "[CLS]",   // 2
            "[SEP]",   // 3
            "[MASK]",  // 4
            "hello",   // 5
            "world",   // 6
            "play",    // 7
            "##ing",   // 8
            "##ed",    // 9
            "good",    // 10
            "morning", // 11
            "the",     // 12
            "quick",   // 13
            "brown",   // 14
            "fox",     // 15
            ",",       // 16
            "!",       // 17
        ];
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect()
    }

    fn make_tokenizer() -> BertTokenizer {
        BertTokenizer::new(base_vocab(), true)
    }

    // ── Test: basic tokenization ────────────────────────────────────────

    #[test]
    fn test_bert_tokenize_basic() {
        let tok = make_tokenizer();
        let tokens = tok.tokenize("Hello, World!");
        // lowercase=true → "hello" and "world"
        assert!(
            tokens.contains(&"hello".to_string()),
            "expected 'hello' in {:?}",
            tokens
        );
        assert!(
            tokens.contains(&"world".to_string()),
            "expected 'world' in {:?}",
            tokens
        );
        // punctuation should be isolated tokens
        assert!(
            tokens.contains(&",".to_string()),
            "expected ',' in {:?}",
            tokens
        );
        assert!(
            tokens.contains(&"!".to_string()),
            "expected '!' in {:?}",
            tokens
        );
    }

    // ── Test: special tokens added in encode ────────────────────────────

    #[test]
    fn test_bert_special_tokens() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world").expect("encode failed");
        // [CLS] at start, [SEP] at end
        assert_eq!(ids[0], tok.cls_token_id(), "first token should be [CLS]");
        assert_eq!(
            *ids.last().expect("non-empty"),
            tok.sep_token_id(),
            "last token should be [SEP]"
        );
    }

    // ── Test: WordPiece subword segmentation ────────────────────────────

    #[test]
    fn test_bert_wordpiece() {
        let tok = make_tokenizer();
        // "playing" → play + ##ing
        let tokens = tok.tokenize("playing");
        assert_eq!(tokens, vec!["play", "##ing"]);
    }

    // ── Test: unknown tokens ────────────────────────────────────────────

    #[test]
    fn test_bert_unknown() {
        let tok = make_tokenizer();
        // "xyzzy" is not in vocab; should map to [UNK]
        let ids = tok.encode("xyzzy").expect("encode failed");
        // [CLS] [UNK] [SEP]
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[1], tok.unk_token_id(), "OOV token should map to [UNK]");
    }

    // ── Test: pair encoding ─────────────────────────────────────────────

    #[test]
    fn test_bert_encode_pair() {
        let tok = make_tokenizer();
        let (ids, type_ids) = tok
            .encode_pair("hello", "world")
            .expect("encode_pair failed");
        // Layout: [CLS](0) hello(0) [SEP](0) world(1) [SEP](1)
        assert_eq!(ids[0], tok.cls_token_id());
        // type_id for [CLS] should be 0
        assert_eq!(type_ids[0], 0);
        // Last token should be [SEP]
        assert_eq!(*ids.last().expect("non-empty"), tok.sep_token_id());
        // Last type_id should be 1 (segment B)
        assert_eq!(*type_ids.last().expect("non-empty"), 1);

        // Verify segment boundary: find first SEP
        let first_sep_pos = ids
            .iter()
            .position(|&id| id == tok.sep_token_id())
            .expect("has SEP");
        // Everything up to and including first SEP is type 0
        for i in 0..=first_sep_pos {
            assert_eq!(type_ids[i], 0, "position {} should be type 0", i);
        }
        // Everything after first SEP is type 1
        for i in (first_sep_pos + 1)..type_ids.len() {
            assert_eq!(type_ids[i], 1, "position {} should be type 1", i);
        }
    }

    // ── Test: decode skips special tokens ───────────────────────────────

    #[test]
    fn test_bert_decode_skips_special() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world").expect("encode failed");
        let decoded = tok.decode(&ids);
        // [CLS] and [SEP] should not appear in decoded output
        assert!(
            !decoded.contains("[CLS]"),
            "decoded should not contain [CLS]: {:?}",
            decoded
        );
        assert!(
            !decoded.contains("[SEP]"),
            "decoded should not contain [SEP]: {:?}",
            decoded
        );
        assert!(
            decoded.contains("hello"),
            "decoded should contain 'hello': {:?}",
            decoded
        );
        assert!(
            decoded.contains("world"),
            "decoded should contain 'world': {:?}",
            decoded
        );
    }

    // ── Test: batch padding ─────────────────────────────────────────────

    #[test]
    fn test_bert_batch_padding() {
        let tok = make_tokenizer();
        let texts = vec!["hello", "hello world"];
        let batch = tok
            .encode_batch(&texts, 10, true, false)
            .expect("encode_batch failed");

        assert_eq!(batch.len(), 2);
        // All sequences must have the same length after padding
        let len0 = batch.encodings[0].len();
        let len1 = batch.encodings[1].len();
        assert_eq!(len0, len1, "padded lengths must be equal");

        // The shorter sequence should have padding tokens
        let short_enc = &batch.encodings[0];
        let has_pad = short_enc
            .input_ids
            .iter()
            .any(|&id| id == tok.pad_token_id());
        let longer_real = batch.encodings[1]
            .attention_mask
            .iter()
            .filter(|&&m| m == 1)
            .count();
        let shorter_real = batch.encodings[0]
            .attention_mask
            .iter()
            .filter(|&&m| m == 1)
            .count();
        assert!(
            has_pad,
            "shorter sequence should have padding; ids={:?}",
            short_enc.input_ids
        );
        assert!(
            shorter_real < longer_real,
            "shorter text should have fewer real tokens"
        );
        // Padding positions should have attention_mask = 0
        for (id, mask) in short_enc
            .input_ids
            .iter()
            .zip(short_enc.attention_mask.iter())
        {
            if *id == tok.pad_token_id() {
                assert_eq!(*mask, 0, "padding token must have mask 0");
            }
        }
    }

    // ── Test: batch truncation ──────────────────────────────────────────

    #[test]
    fn test_bert_batch_truncation() {
        let tok = make_tokenizer();
        // "the quick brown fox" has 4 real tokens; max_length=4 forces truncation
        // (budget: 4 - 2 = 2 content tokens)
        let texts = vec!["the quick brown fox"];
        let batch = tok
            .encode_batch(&texts, 4, false, true)
            .expect("encode_batch failed");

        let enc = &batch.encodings[0];
        // ids: [CLS] + up-to-2 content tokens + [SEP] = 4
        assert_eq!(enc.input_ids.len(), 4);
        assert_eq!(enc.input_ids[0], tok.cls_token_id());
        assert_eq!(
            *enc.input_ids.last().expect("non-empty"),
            tok.sep_token_id()
        );
    }

    // ── Test: lowercase folding ──────────────────────────────────────────

    #[test]
    fn test_bert_lowercase() {
        let tok_lower = BertTokenizer::new(base_vocab(), true);
        let tok_cased = BertTokenizer::new(base_vocab(), false);

        // With lowercase=true, "HELLO" should hit "hello" in vocab
        let lower_tokens = tok_lower.tokenize("HELLO");
        assert!(
            lower_tokens.contains(&"hello".to_string()),
            "lowercase should map HELLO→hello: {:?}",
            lower_tokens
        );

        // With lowercase=false, "HELLO" is not in vocab → [UNK]
        let cased_tokens = tok_cased.tokenize("HELLO");
        assert!(
            cased_tokens.contains(&"[UNK]".to_string()),
            "cased tokenizer should map HELLO to [UNK]: {:?}",
            cased_tokens
        );
    }

    // ── Test: build from in-memory vocab ─────────────────────────────────

    #[test]
    fn test_bert_from_vocab_string() {
        // Build vocab directly from a list of strings (simulating a vocab.txt load)
        let token_list: &[&str] = &[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "rust", "is", "great",
        ];
        let vocab: HashMap<String, u32> = token_list
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        let tokenizer = BertTokenizer::new(vocab, true);
        let ids = tokenizer.encode("rust is great").expect("encode failed");
        // [CLS] rust is great [SEP] = 5 tokens
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], tokenizer.cls_token_id());
    }

    // ── Test: empty input ────────────────────────────────────────────────

    #[test]
    fn test_bert_empty_input() {
        let tok = make_tokenizer();
        let ids = tok.encode("").expect("encode empty");
        // [CLS] [SEP]
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], tok.cls_token_id());
        assert_eq!(ids[1], tok.sep_token_id());
    }

    // ── Test: all-OOV input ───────────────────────────────────────────────

    #[test]
    fn test_bert_all_oov() {
        let tok = make_tokenizer();
        // Three OOV words, each → [UNK]
        let ids = tok.encode("zzz yyy xxx").expect("encode all-OOV");
        // [CLS] [UNK] [UNK] [UNK] [SEP]
        assert_eq!(ids.len(), 5);
        for &id in &ids[1..4] {
            assert_eq!(id, tok.unk_token_id());
        }
    }

    // ── Test: max_len=1 edge case ─────────────────────────────────────────

    #[test]
    fn test_bert_max_len_one_truncation() {
        let tok = make_tokenizer();
        // With max_length=1, budget=0 content tokens (1 - 2 saturates to 0).
        // The encoder always emits [CLS] + content + [SEP]; with zero content
        // budget the result is [CLS] [SEP] (length 2), which exceeds max_length=1
        // but is the minimal valid BERT sequence.  The implementation does not
        // truncate the mandatory special tokens themselves.
        let enc = tok
            .encode_single("hello world", 1, false, true)
            .expect("encode_single");
        // Always at least [CLS] + [SEP].
        assert!(
            enc.input_ids.len() >= 2,
            "must contain at least [CLS] and [SEP]"
        );
        assert_eq!(enc.input_ids[0], tok.cls_token_id());
        assert_eq!(
            *enc.input_ids.last().expect("non-empty"),
            tok.sep_token_id()
        );
        // No content tokens (budget was 0).
        assert_eq!(enc.input_ids.len(), 2, "only [CLS] and [SEP] expected");
    }

    // ── Test: decode WordPiece continuations ─────────────────────────────

    #[test]
    fn test_bert_decode_wordpiece_merge() {
        let tok = make_tokenizer();
        // play(7) + ##ing(8) → "playing"
        let decoded = tok.decode(&[7, 8]);
        assert_eq!(decoded, "playing", "expected 'playing', got '{}'", decoded);
    }

    // ── Test: from_vocab_file round-trip ─────────────────────────────────

    #[test]
    fn test_bert_from_vocab_file() {
        use std::io::Write;

        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_bert_vocab_test.txt");
        {
            let mut f = std::fs::File::create(&tmp).expect("create temp file");
            writeln!(f, "[PAD]").expect("write");
            writeln!(f, "[UNK]").expect("write");
            writeln!(f, "[CLS]").expect("write");
            writeln!(f, "[SEP]").expect("write");
            writeln!(f, "[MASK]").expect("write");
            writeln!(f, "hello").expect("write");
            writeln!(f, "world").expect("write");
        }
        let path = tmp.to_str().expect("valid path");
        let tokenizer = BertTokenizer::from_vocab_file(path).expect("from_vocab_file");
        assert_eq!(tokenizer.convert_token_to_id("[CLS]"), Some(2));
        assert_eq!(tokenizer.convert_token_to_id("hello"), Some(5));
        let _ = std::fs::remove_file(&tmp);
    }
}
