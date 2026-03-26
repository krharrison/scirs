//! RoBERTa byte-level BPE tokenizer.
//!
//! RoBERTa reuses the GPT-2 byte-level BPE tokenizer but replaces BERT-style
//! special tokens (`[CLS]`, `[SEP]`) with `<s>` (BOS) and `</s>` (EOS).
//!
//! Special token IDs follow the standard RoBERTa convention from the
//! original fairseq/HuggingFace implementation:
//! - `<s>` BOS / CLS  → ID 0
//! - `<pad>`           → ID 1
//! - `</s>` EOS / SEP → ID 2
//! - `<unk>`           → ID 3
//! - `<mask>`          → ID 50264
//!
//! # Pair-encoding layout
//!
//! Single:  `<s>` · tokens · `</s>`
//! Pair:    `<s>` · tokens_a · `</s>` · `</s>` · tokens_b · `</s>`
//!          (two consecutive `</s>` separators between segments, as in the
//!          original RoBERTa implementation)

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ─── Special-token IDs (RoBERTa defaults) ─────────────────────────────────────

/// BOS / CLS token string used by RoBERTa.
pub const BOS_TOKEN: &str = "<s>";
/// EOS / SEP token string used by RoBERTa.
pub const EOS_TOKEN: &str = "</s>";
/// Padding token string.
pub const PAD_TOKEN: &str = "<pad>";
/// Unknown token string.
pub const UNK_TOKEN: &str = "<unk>";
/// Mask token string.
pub const MASK_TOKEN: &str = "<mask>";

/// Default BOS token ID.
const DEFAULT_BOS_ID: u32 = 0;
/// Default PAD token ID.
const DEFAULT_PAD_ID: u32 = 1;
/// Default EOS token ID.
const DEFAULT_EOS_ID: u32 = 2;
/// Default UNK token ID.
const DEFAULT_UNK_ID: u32 = 3;
/// Default MASK token ID (fairseq / HF RoBERTa).
const DEFAULT_MASK_ID: u32 = 50264;

// ─── Byte → Unicode helpers (GPT-2 / RoBERTa convention) ──────────────────────

/// Build the GPT-2 / RoBERTa byte-to-Unicode mapping.
///
/// Printable ASCII and Latin-1 supplement bytes map to themselves; all other
/// bytes are shifted to a private-use area above U+0100.
fn build_byte_encoder() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut next_free: u32 = 256;

    for b in 0u16..=255 {
        let byte = b as u8;
        let is_printable = (b'!'..=b'~').contains(&byte)
            || (0xA1u8..=0xACu8).contains(&byte)
            || (0xAEu8..=0xFFu8).contains(&byte);

        if is_printable {
            map.insert(byte, byte as char);
        } else {
            if let Some(c) = char::from_u32(next_free) {
                map.insert(byte, c);
                next_free += 1;
            }
        }
    }
    map
}

/// Invert the byte encoder to get Unicode char → byte.
fn build_byte_decoder(encoder: &HashMap<u8, char>) -> HashMap<char, u8> {
    encoder.iter().map(|(&b, &c)| (c, b)).collect()
}

// ─── BPE merge application ─────────────────────────────────────────────────────

/// Find the adjacent bigram with the lowest merge rank and apply it once.
///
/// Returns `true` when a merge was performed.
fn apply_best_merge(
    symbols: &mut Vec<String>,
    merge_ranks: &HashMap<(String, String), usize>,
) -> bool {
    if symbols.len() < 2 {
        return false;
    }

    let best = symbols
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| {
            let pair = (w[0].clone(), w[1].clone());
            merge_ranks.get(&pair).map(|&rank| (i, rank))
        })
        .min_by_key(|&(_, rank)| rank);

    match best {
        None => false,
        Some((pos, _)) => {
            let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
            symbols.remove(pos + 1);
            symbols[pos] = merged;
            true
        }
    }
}

// ─── Pre-tokenizer ─────────────────────────────────────────────────────────────

/// GPT-2 / RoBERTa-style pre-tokenization: split on whitespace boundaries,
/// attaching leading spaces to the following word.
///
/// This is a simplified implementation of the GPT-2 regex pattern that handles
/// the main cases (letters, digits, punctuation, contractions).
fn pretokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0usize;

    while i < n {
        let ch = chars[i];

        // Handle English contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if ch == '\'' && i + 1 < n {
            match chars[i + 1] {
                's' | 't' | 'm' | 'd' => {
                    tokens.push(format!("'{}", chars[i + 1]));
                    i += 2;
                    continue;
                }
                'r' if i + 2 < n && chars[i + 2] == 'e' => {
                    tokens.push("'re".to_string());
                    i += 3;
                    continue;
                }
                'v' if i + 2 < n && chars[i + 2] == 'e' => {
                    tokens.push("'ve".to_string());
                    i += 3;
                    continue;
                }
                'l' if i + 2 < n && chars[i + 2] == 'l' => {
                    tokens.push("'ll".to_string());
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }

        if ch.is_whitespace() {
            // Collect whitespace run; will be merged into next token.
            let ws_start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            let ws: String = chars[ws_start..i].iter().collect();
            // If more characters remain, merge the space prefix with them.
            if i < n {
                let next_start = i;
                let next_end = advance_word(&chars, i);
                let word: String = chars[next_start..next_end].iter().collect();
                tokens.push(format!("{}{}", ws, word));
                i = next_end;
            } else {
                tokens.push(ws);
            }
        } else {
            let start = i;
            let end = advance_word(&chars, i);
            let word: String = chars[start..end].iter().collect();
            tokens.push(word);
            i = end;
        }
    }

    tokens
}

/// Advance `i` past a run of homogeneous characters (all alpha, all digit, or
/// single punctuation).
fn advance_word(chars: &[char], i: usize) -> usize {
    let n = chars.len();
    if i >= n {
        return i;
    }
    let ch = chars[i];
    if ch.is_alphabetic() {
        let mut j = i;
        while j < n && chars[j].is_alphabetic() {
            j += 1;
        }
        j
    } else if ch.is_ascii_digit() {
        let mut j = i;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        j
    } else {
        i + 1
    }
}

// ─── RobertaTokenizer ─────────────────────────────────────────────────────────

/// RoBERTa byte-level BPE tokenizer.
///
/// Encodes text at the byte level (every byte maps to a unique Unicode
/// character) then applies learned BPE merges.  Unlike BERT, RoBERTa uses
/// `<s>` and `</s>` as special tokens.
///
/// # Building a tokenizer
///
/// The simplest way to create a [`RobertaTokenizer`] is from a set of BPE
/// merges and a vocabulary.  For unit-testing you can use
/// [`RobertaTokenizer::from_corpus`] to learn merges from scratch.
///
/// ```rust
/// use scirs2_text::tokenizers::roberta::RobertaTokenizer;
///
/// let tok = RobertaTokenizer::from_corpus(
///     &["hello world", "hello there"],
///     300,
/// ).expect("training failed");
///
/// let ids = tok.encode("hello").expect("encode failed");
/// assert_eq!(ids[0], tok.bos_token_id());
/// assert_eq!(*ids.last().unwrap(), tok.eos_token_id());
/// ```
#[derive(Debug, Clone)]
pub struct RobertaTokenizer {
    /// Vocabulary: byte-level token string → ID.
    vocab: HashMap<String, u32>,
    /// Inverse vocabulary: ID → token string.
    id_to_token: HashMap<u32, String>,
    /// Ordered BPE merge rules (lower index = higher priority).
    merges: Vec<(String, String)>,
    /// Pre-built merge rank map for O(1) lookup.
    merge_ranks: HashMap<(String, String), usize>,
    /// Byte → Unicode char mapping.
    byte_encoder: HashMap<u8, char>,
    /// Unicode char → byte mapping.
    byte_decoder: HashMap<char, u8>,
    // Special token IDs
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
    unk_id: u32,
    mask_id: u32,
    /// Maximum encoded sequence length.
    max_len: usize,
}

impl RobertaTokenizer {
    // ── Construction ──────────────────────────────────────────────────────

    /// Build a `RobertaTokenizer` from a pre-built vocabulary and ordered
    /// merge rules.
    ///
    /// The vocabulary must already contain the five special tokens (`<s>`,
    /// `</s>`, `<pad>`, `<unk>`, `<mask>`).  Missing special tokens are
    /// automatically inserted.
    pub fn new(mut vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Result<Self> {
        if vocab.is_empty() {
            return Err(TextError::VocabularyError(
                "vocabulary must not be empty".to_string(),
            ));
        }

        // Ensure special tokens are present.
        let specials = [
            (BOS_TOKEN, DEFAULT_BOS_ID),
            (PAD_TOKEN, DEFAULT_PAD_ID),
            (EOS_TOKEN, DEFAULT_EOS_ID),
            (UNK_TOKEN, DEFAULT_UNK_ID),
            (MASK_TOKEN, DEFAULT_MASK_ID),
        ];
        for (tok, default_id) in &specials {
            vocab.entry(tok.to_string()).or_insert(*default_id);
        }

        let bos_id = vocab[BOS_TOKEN];
        let eos_id = vocab[EOS_TOKEN];
        let pad_id = vocab[PAD_TOKEN];
        let unk_id = vocab[UNK_TOKEN];
        let mask_id = vocab[MASK_TOKEN];

        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        let merge_ranks: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.clone(), b.clone()), rank))
            .collect();

        let byte_encoder = build_byte_encoder();
        let byte_decoder = build_byte_decoder(&byte_encoder);

        Ok(RobertaTokenizer {
            vocab,
            id_to_token,
            merges,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            mask_id,
            max_len: 512,
        })
    }

    /// Train a `RobertaTokenizer` from a text corpus.
    ///
    /// Learns byte-pair merges iteratively until `vocab_size` is reached or
    /// no more merges can be applied.
    ///
    /// This is primarily intended for testing.  Production use typically loads
    /// a pre-trained vocabulary and merge file.
    pub fn from_corpus(corpus: &[&str], vocab_size: usize) -> Result<Self> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus must not be empty".to_string(),
            ));
        }

        let byte_encoder = build_byte_encoder();

        // Collect word frequencies.
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for text in corpus {
            for token in pretokenize(text) {
                *word_freqs.entry(token).or_insert(0) += 1;
            }
        }

        if word_freqs.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus produced no tokens".to_string(),
            ));
        }

        // Convert each word to its byte-level Unicode symbol sequence.
        let mut word_splits: HashMap<String, (Vec<String>, usize)> = HashMap::new();
        for (word, freq) in &word_freqs {
            let symbols: Vec<String> = word
                .as_bytes()
                .iter()
                .map(|b| {
                    byte_encoder
                        .get(b)
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| format!("{}", b))
                })
                .collect();
            word_splits.insert(word.clone(), (symbols, *freq));
        }

        // Seed vocabulary with special tokens then all unique byte symbols.
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let special_list = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN];
        let mut next_id: u32 = 0;
        for tok in &special_list {
            vocab.insert(tok.to_string(), next_id);
            next_id += 1;
        }
        // Add byte-level single characters.
        for (symbols, _) in word_splits.values() {
            for sym in symbols {
                vocab.entry(sym.clone()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
            }
        }

        let base_vocab_size = vocab.len();
        let num_merges = vocab_size.saturating_sub(base_vocab_size);

        // Iteratively learn merges.
        let mut merges: Vec<(String, String)> = Vec::new();
        let mut current_splits = word_splits;

        for _ in 0..num_merges {
            // Count bigram frequencies.
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
            for (symbols, freq) in current_splits.values() {
                for w in symbols.windows(2) {
                    let pair = (w[0].clone(), w[1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freqs.is_empty() {
                break;
            }

            // Select most frequent pair (tie-break by lexicographic order for
            // determinism).
            let best_pair = pair_freqs
                .iter()
                .max_by(|(k1, v1), (k2, v2)| v1.cmp(v2).then_with(|| k1.cmp(k2).reverse()))
                .map(|(k, _)| k.clone());

            let (left, right) = match best_pair {
                Some(p) => p,
                None => break,
            };

            let merged = format!("{}{}", left, right);
            vocab.entry(merged.clone()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            merges.push((left.clone(), right.clone()));

            // Apply the merge to all word splits.
            for (_, (symbols, _)) in current_splits.iter_mut() {
                let mut i = 0;
                while i + 1 < symbols.len() {
                    if symbols[i] == left && symbols[i + 1] == right {
                        symbols[i] = merged.clone();
                        symbols.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Insert MASK token at its canonical position.
        vocab
            .entry(MASK_TOKEN.to_string())
            .or_insert(DEFAULT_MASK_ID);

        Self::new(vocab, merges)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Returns the BOS / `<s>` token ID.
    pub fn bos_token_id(&self) -> u32 {
        self.bos_id
    }

    /// Returns the EOS / `</s>` token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_id
    }

    /// Returns the `<pad>` token ID.
    pub fn pad_token_id(&self) -> u32 {
        self.pad_id
    }

    /// Returns the `<unk>` token ID.
    pub fn unk_token_id(&self) -> u32 {
        self.unk_id
    }

    /// Returns the `<mask>` token ID.
    pub fn mask_token_id(&self) -> u32 {
        self.mask_id
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Return a reference to the full `token → id` vocabulary map.
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Return the ordered BPE merge rules.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Override the maximum sequence length (default 512).
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }

    // ── Core encoding ─────────────────────────────────────────────────────

    /// Encode a single `word` (pre-token) to its BPE sub-tokens.
    fn bpe_encode_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }

        // Convert word bytes to byte-level Unicode chars.
        let mut symbols: Vec<String> = word
            .as_bytes()
            .iter()
            .map(|b| {
                self.byte_encoder
                    .get(b)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| format!("{}", b))
            })
            .collect();

        // Apply BPE merges greedily.
        loop {
            if !apply_best_merge(&mut symbols, &self.merge_ranks) {
                break;
            }
        }

        symbols
    }

    /// Tokenize `text` to a list of byte-level BPE token strings.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        pretokenize(text)
            .iter()
            .flat_map(|word| self.bpe_encode_word(word))
            .collect()
    }

    /// Look up a token string in the vocabulary.
    fn token_to_id(&self, token: &str) -> u32 {
        self.vocab.get(token).copied().unwrap_or(self.unk_id)
    }

    // ── Public encoding API ───────────────────────────────────────────────

    /// Encode `text` as `<s>` · BPE-tokens · `</s>`.
    ///
    /// Returns the full sequence of token IDs including special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenize(text);
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(self.bos_id);
        ids.extend(tokens.iter().map(|t| self.token_to_id(t)));
        ids.push(self.eos_id);
        Ok(ids)
    }

    /// Encode a pair of texts in RoBERTa style.
    ///
    /// Layout: `<s>` · tokens_a · `</s>` · `</s>` · tokens_b · `</s>`
    ///
    /// Returns `(token_ids, token_type_ids)`.  Per the RoBERTa paper, all
    /// token type IDs are `0` (RoBERTa does not use segment embeddings).
    pub fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        let tokens_a = self.tokenize(text_a);
        let tokens_b = self.tokenize(text_b);

        // <s> A </s> </s> B </s>
        let total = 1 + tokens_a.len() + 1 + 1 + tokens_b.len() + 1;
        let mut ids = Vec::with_capacity(total);

        ids.push(self.bos_id);
        ids.extend(tokens_a.iter().map(|t| self.token_to_id(t)));
        ids.push(self.eos_id);
        ids.push(self.eos_id); // double </s> between segments
        ids.extend(tokens_b.iter().map(|t| self.token_to_id(t)));
        ids.push(self.eos_id);

        // RoBERTa uses all-zero token type IDs.
        let type_ids = vec![0u32; ids.len()];

        Ok((ids, type_ids))
    }

    /// Encode `text` with optional truncation and padding.
    ///
    /// Returns a flat `(input_ids, attention_mask)` tuple.
    pub fn encode_padded(
        &self,
        text: &str,
        max_length: usize,
        padding: bool,
        truncation: bool,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        if max_length == 0 {
            return Err(TextError::InvalidInput(
                "max_length must be greater than 0".to_string(),
            ));
        }

        let tokens = self.tokenize(text);
        let budget = max_length.saturating_sub(2); // reserve for <s> and </s>

        let content: Vec<u32> = if truncation && tokens.len() > budget {
            tokens[..budget]
                .iter()
                .map(|t| self.token_to_id(t))
                .collect()
        } else {
            tokens.iter().map(|t| self.token_to_id(t)).collect()
        };

        let mut ids = Vec::with_capacity(max_length);
        ids.push(self.bos_id);
        ids.extend_from_slice(&content);
        ids.push(self.eos_id);

        let real_len = ids.len();

        if padding && ids.len() < max_length {
            let pad_count = max_length - ids.len();
            ids.extend(std::iter::repeat_n(self.pad_id, pad_count));
        }

        let seq_len = ids.len();
        let mut mask = vec![0u32; seq_len];
        for m in mask.iter_mut().take(real_len) {
            *m = 1;
        }

        Ok((ids, mask))
    }

    // ── Decoding ──────────────────────────────────────────────────────────

    /// Decode token IDs back to a UTF-8 string.
    ///
    /// Special tokens (`<s>`, `</s>`, `<pad>`, `<mask>`) are skipped.  The
    /// byte-level Unicode characters are converted back to their original bytes.
    pub fn decode(&self, ids: &[u32]) -> String {
        let skip_ids: [u32; 4] = [self.bos_id, self.eos_id, self.pad_id, self.mask_id];
        let mut byte_buf: Vec<u8> = Vec::new();

        for &id in ids {
            if skip_ids.contains(&id) {
                continue;
            }
            let tok = match self.id_to_token.get(&id) {
                Some(t) => t.as_str(),
                None => continue,
            };
            // Decode each Unicode char in the token back to the original byte.
            for ch in tok.chars() {
                if let Some(&b) = self.byte_decoder.get(&ch) {
                    byte_buf.push(b);
                }
            }
        }

        String::from_utf8_lossy(&byte_buf).into_owned()
    }

    /// Look up a token ID by string.
    pub fn convert_token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Look up a token string by ID.
    pub fn convert_id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ─────────────────────────────────────────────────────────────

    /// Train a small tokenizer for testing purposes.
    fn small_tokenizer() -> RobertaTokenizer {
        let corpus = &[
            "hello world",
            "hello there",
            "world peace",
            "the quick brown fox",
            "rust is fast",
            "rust programming language",
        ];
        RobertaTokenizer::from_corpus(corpus, 300).expect("training failed")
    }

    // ── Test: special tokens in single encoding ─────────────────────────

    #[test]
    fn test_roberta_special_tokens() {
        let tok = small_tokenizer();
        let ids = tok.encode("hello").expect("encode failed");
        assert!(ids.len() >= 2, "encoded ids must include at least BOS+EOS");
        assert_eq!(ids[0], tok.bos_token_id(), "first token must be <s>");
        assert_eq!(
            *ids.last().expect("non-empty"),
            tok.eos_token_id(),
            "last token must be </s>"
        );
    }

    // ── Test: pair encoding layout ──────────────────────────────────────

    #[test]
    fn test_roberta_encode_pair() {
        let tok = small_tokenizer();
        let (ids, _type_ids) = tok
            .encode_pair("hello", "world")
            .expect("encode_pair failed");

        // Layout: <s> A </s> </s> B </s>
        // First token must be <s>
        assert_eq!(ids[0], tok.bos_token_id());
        // Last token must be </s>
        assert_eq!(*ids.last().expect("non-empty"), tok.eos_token_id());

        // There must be exactly two consecutive </s> in the middle
        let eos = tok.eos_token_id();
        let double_sep = ids
            .windows(2)
            .any(|w| w[0] == eos && w[1] == eos && w[0] != ids[0]);
        assert!(
            double_sep,
            "pair encoding must contain double </s> separator: {:?}",
            ids
        );
    }

    // ── Test: pair type IDs are all zero ────────────────────────────────

    #[test]
    fn test_roberta_pair_type_ids_all_zero() {
        let tok = small_tokenizer();
        let (_ids, type_ids) = tok.encode_pair("hello", "world").expect("encode_pair");
        for (i, &tid) in type_ids.iter().enumerate() {
            assert_eq!(tid, 0, "type_id at position {} should be 0", i);
        }
    }

    // ── Test: empty string ───────────────────────────────────────────────

    #[test]
    fn test_roberta_empty_input() {
        let tok = small_tokenizer();
        let ids = tok.encode("").expect("encode empty");
        // <s> </s>
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], tok.bos_token_id());
        assert_eq!(ids[1], tok.eos_token_id());
    }

    // ── Test: padded encoding ────────────────────────────────────────────

    #[test]
    fn test_roberta_padded_encoding() {
        let tok = small_tokenizer();
        let (ids, mask) = tok
            .encode_padded("hello", 16, true, false)
            .expect("encode_padded");
        assert_eq!(ids.len(), 16);
        assert_eq!(mask.len(), 16);
        // First token is <s> (mask=1)
        assert_eq!(ids[0], tok.bos_token_id());
        assert_eq!(mask[0], 1);
        // Padding tokens should have mask=0
        for (id, m) in ids.iter().zip(mask.iter()) {
            if *id == tok.pad_token_id() {
                assert_eq!(*m, 0, "padding tokens must have mask=0");
            }
        }
    }

    // ── Test: truncation ────────────────────────────────────────────────

    #[test]
    fn test_roberta_truncation() {
        let tok = small_tokenizer();
        // "the quick brown fox" has many tokens; truncate to 4
        let (ids, _mask) = tok
            .encode_padded("the quick brown fox", 4, false, true)
            .expect("encode_padded");
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], tok.bos_token_id());
        assert_eq!(*ids.last().expect("non-empty"), tok.eos_token_id());
    }

    // ── Test: tokenize returns non-empty for known words ────────────────

    #[test]
    fn test_roberta_tokenize_non_empty() {
        let tok = small_tokenizer();
        let tokens = tok.tokenize("hello");
        assert!(
            !tokens.is_empty(),
            "tokenize should return at least one token"
        );
    }

    // ── Test: vocab_size is reasonable ──────────────────────────────────

    #[test]
    fn test_roberta_vocab_size() {
        let tok = small_tokenizer();
        // Vocabulary must contain special tokens at minimum.
        assert!(tok.vocab_size() >= 5, "vocab must include special tokens");
    }

    // ── Test: decode round-trip (approximate) ───────────────────────────

    #[test]
    fn test_roberta_decode_skips_specials() {
        let tok = small_tokenizer();
        let ids = tok.encode("hello").expect("encode");
        let decoded = tok.decode(&ids);
        // Special tokens should not appear in the decoded text.
        assert!(
            !decoded.contains("<s>"),
            "decoded must not contain <s>: {:?}",
            decoded
        );
        assert!(
            !decoded.contains("</s>"),
            "decoded must not contain </s>: {:?}",
            decoded
        );
    }

    // ── Test: convert_token_to_id for special tokens ─────────────────────

    #[test]
    fn test_roberta_convert_special_tokens() {
        let tok = small_tokenizer();
        assert_eq!(
            tok.convert_token_to_id(BOS_TOKEN),
            Some(DEFAULT_BOS_ID),
            "<s> should map to BOS_ID"
        );
        assert_eq!(
            tok.convert_token_to_id(EOS_TOKEN),
            Some(DEFAULT_EOS_ID),
            "</s> should map to EOS_ID"
        );
        assert_eq!(
            tok.convert_token_to_id(PAD_TOKEN),
            Some(DEFAULT_PAD_ID),
            "<pad> should map to PAD_ID"
        );
        assert_eq!(
            tok.convert_token_to_id(UNK_TOKEN),
            Some(DEFAULT_UNK_ID),
            "<unk> should map to UNK_ID"
        );
    }
}
